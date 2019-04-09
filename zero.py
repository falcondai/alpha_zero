import math
from typing import List

import numpy as np

from base import AlphaZeroConfig, Node, Game, Network, ReplayBuffer, SharedStorage
from backend import launch_job, train_network


def alpha_zero_train(config: AlphaZeroConfig):
    '''
    AlphaZero training is split into two independent parts: Network training and self-play data generation.
    These two parts only communicate by transferring the latest network checkpoint from the training to the self-play, and the finished games from the self-play to the training.
    '''
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    # Self play generates finished games
    for i in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    # Supervised learning on sampled finished games
    train_network(config, storage, replay_buffer)

    return storage.latest_network()


def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)


def play_game(config: AlphaZeroConfig, Game, network: Network):
    game = Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)
        print(game.child_visits[-1])
        game.ch.print_board()
    return game


def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
    '''
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of the search tree and traversing the tree according to a modified UCB formula until we reach a leaf node.

    This implementation keeps the rollout statistics in a separate tree structure.
    '''
    root = Node(0)
    evaluate(root, game, network)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        # TODO Handle draws due to infinite loops? The random behavior will likely prevent looping for too long.
        while node.expanded():
            # Tree policy, greedy with respect to modified UCB scores
            action, node = select_child(config, node)
            scratch_game.apply(action)
            # NOTE: search path is within the expanded region of nodes
            search_path.append(node)

        # Expand and evaluate (Done with a rollout policy in vanilla MCTS)
        # NOTE: Instead of running a simulation (MC evaluation) for an unexpanded node, we evaluate it by the value network.
        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.is_first_player_turn())
        # print('search path', search_path)
        # print([(ac, child.visit_count) for ac, child in root.children.items()])
    return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
    visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        # XXX I doubt this is making much difference since visit_count off by delta would translate to factor of exp(delta) in probability
        # Would be more random if we directly sample according to visit counts
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    # XXX pi_hat influences the search priority
    prior_score = pb_c * child.prior
    # XXX Should we use V_hat during testing? Maybe we should just use true terminal values. This way, value prediction is really used only for training (features).
    value_score = child.sampled_value()
    return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
    ego_rep = game.ego_board_representation()
    ego_value, ego_policy_logits = network.inference(ego_rep)
    # Transform ego-centric to absolute
    is_first_player = game.is_first_player_turn()
    value = game.ego2abs_value(is_first_player, ego_value)
    policy_logits = game.ego2abs_policy(is_first_player, ego_policy_logits)

    # Expand the node.
    node.to_play = game.is_first_player_turn()
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return value


# At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (0 - value)
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def softmax_sample(freq_actions):
    freqs, actions = zip(*freq_actions)
    min_freq = min(freqs)
    freqs = [freq - min_freq for freq in freqs]
    logits = np.exp(freqs)
    ps = logits / logits.sum()
    choice = np.random.choice(len(freq_actions), p=ps)
    return freqs[choice], actions[choice]
