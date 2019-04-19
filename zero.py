import math
import time
from typing import List
# from checkers import Checkers

import numpy as np

from base import AlphaZeroConfig, Node, Game, Network, ReplayBuffer, SharedStorage
# from backend import launch_job, train_network


# def alpha_zero_train(config: AlphaZeroConfig):
#     '''
#     AlphaZero training is split into two independent parts: Network training and self-play data generation.
#     These two parts only communicate by transferring the latest network checkpoint from the training to the self-play, and the finished games from the self-play to the training.
#     '''
#     storage = SharedStorage()
#     replay_buffer = ReplayBuffer(config)
#
#     # Self play generates finished games
#     for i in range(config.num_actors):
#         launch_job(run_selfplay, config, storage, replay_buffer)
#
#     # Supervised learning on sampled finished games
#     train_network(config, storage, replay_buffer)
#
#     return storage.latest_network()


def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)


def play_game(config: AlphaZeroConfig, Game, network: Network, discount: float = 1):
    t0 = time.time()
    game = Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = run_mcts(config, game, network, discount=discount)
        game.apply(action)
        game.store_search_statistics(root)
    if config.max_moves <= len(game.history):
        # XXX: Set the game value to draw if it continues for too long
        game.game_value = 0
    # Logging
    game.ch.print_board()
    print('value', game.game_value, 'time', '%.2f' % (time.time() - t0), 'len', len(game.history))
    return game


def run_mcts(config: AlphaZeroConfig, game: Game, network: Network, discount: float = 1, use_cpu: bool = False):
    '''
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of the search tree and traversing the tree according to a modified UCB formula until we reach a leaf node.

    This implementation keeps the rollout statistics in a separate tree structure.
    '''
    root = Node(0)
    evaluate(root, game, network, use_cpu=use_cpu)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        # NOTE: Clone a game possibly without its history. Only its game state is needed
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
        value = evaluate(node, scratch_game, network, use_cpu=use_cpu)
        backpropagate(search_path, value, discount=discount)
    # # Log
    # for ac, child in root.children.items():
    #     print(game.actions[ac], child.is_first_player == root.is_first_player, '%.2f' % child.sampled_value(), child.visit_count)
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


def select_child(config: AlphaZeroConfig, node: Node):
    '''
    Select the child with the highest modified UCB score.
    '''
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
    '''
    The score for a node is based on its value, plus an exploration bonus based on the prior.
    '''
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    # Classic UCB
    # pb_c = 1 * np.sqrt(np.log(parent.visit_count) / child.visit_count) if child.visit_count > 0 else float('inf')
    # XXX pi_hat influences the search priority
    prior_score = pb_c * child.prior
    # XXX Should we use V_hat during testing? Maybe we should just use true terminal values. This way, value prediction is really used only for training (features).
    # NOTE: we are choosing a good move for the player of the parent node and each node returns the ego-centric value
    value_score = child.sampled_value() if child.is_first_player == parent.is_first_player else (0 - child.sampled_value())
    return prior_score + value_score


def evaluate(node: Node, game: Game, network: Network, use_cpu: bool = False):
    '''
    We use the neural network to obtain a value and policy prediction.
    '''
    ego_rep = game.ego_board_representation()
    ego_value, ego_policy_logits = network.single_inference(ego_rep, use_cpu=use_cpu)
    # Transform ego-centric to absolute
    is_first_player = game.is_first_player_turn()
    value = game.ego2abs_value(is_first_player, ego_value)
    policy_logits = game.ego2abs_policy(is_first_player, ego_policy_logits)

    # Expand the node.
    node.is_first_player = is_first_player
    # print('eval', '%0.2f' % policy_logits.max())
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
        # node.children[action] = Node(0)
    return value

    # XXX: sanity check with simulated values
    # node.is_first_player = game.is_first_player_turn()
    # value = 0
    # n_trials = 10
    # max_plies = 200
    # st = game.ch.save_state()
    # for i in range(n_trials):
    #     sim = Checkers()
    #     sim.restore_state(st)
    #     # Random policy
    #     ply = 0
    #     moves = sim.legal_moves()
    #     # Check for a terminal state
    #     if len(moves) == 0:
    #         # One player wins
    #         winner = 'white' if st[1] == 'black' else 'black'
    #     else:
    #         winner = None
    #     while ply < max_plies and winner is None:
    #         from_sq, to_sq = moves[np.random.randint(len(moves))]
    #         board, turn, last_moved_piece, moves, winner = sim.move(from_sq, to_sq, skip_check=True)
    #         ply += 1
    #     # Returns the winner or None in a draw
    #     if winner == 'black':
    #         value += 1
    #     elif winner == 'white':
    #         value -= 1
    # # Expand the node
    # legal_actions = game.legal_actions()
    # for ac in legal_actions:
    #     node.children[ac] = Node(1 / len(legal_actions))
    # return value / n_trials


def backpropagate(search_path: List[Node], value: float, discount: float = 1):
    '''
    At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
    '''
    running_discount = 1
    for node in search_path[::-1]:
        # Ego-centric value at the node
        # XXX: Discount the game value by its depth. This helps remove some overly optimistic estimate due to random "mistakes". Quicker win is better.
        running_discount *= discount
        node.value_sum += running_discount * (value if node.is_first_player else (0 - value))
        node.visit_count += 1


def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    '''
    At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to explore new actions.
    '''
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
