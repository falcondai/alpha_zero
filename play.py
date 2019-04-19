from collections import defaultdict
import itertools

import numpy as np

from checkers.game import Checkers
from checkers.agents.baselines import play_a_game, RandomPlayer
from checkers.agents import Player

from example_checkers import CheckersNetwork, CheckersGame
from base import AlphaZeroConfig
from zero import run_mcts


class AlphaZaiPlayer(Player):
    '''AlphaZero player with a trained model'''

    def __init__(self, model: CheckersNetwork, color, policy_only=False, num_simulations=100, discount=0.99, seed=None, use_cpu=False):
        self.model = model
        # Which side is being played
        self.color = color
        # Fixing the random state for easy replications
        self.random = np.random.RandomState(seed=seed)
        self.game = CheckersGame()
        # Test configurations
        self.config = AlphaZeroConfig()
        self.config.num_sampling_moves = 0
        self.config.num_simulations = num_simulations
        # QUESTION: Do we need to remove the exploration noise at root as well?
        self.root_exploration_fraction = 0
        self.policy_only = policy_only
        self.use_cpu = use_cpu
        self.discount = discount

    def next_move(self, board, last_moved_piece):
        # Obtain the ego-centric view
        self.game.ch.restore_state((board, self.color, last_moved_piece))
        # Returns if there is only one legal move
        legal_moves = self.game.ch.legal_moves()
        if len(legal_moves) == 1:
            return legal_moves[0]
        ego_rep = self.game.ego_board_representation()
        # if self.policy_only:
        # Take greedy actions according to predicated MCTS search visits
        val, ego_policy_logits = self.model.single_inference(ego_rep, use_cpu=self.use_cpu)
        # Translate into a move
        policy_logits = self.game.ego2abs_policy(self.color == 'black', ego_policy_logits)
        # Best legal action according to policy prediction
        _, ac = max([(policy_logits[a], a) for a in self.game.legal_actions()])
        pol_move = self.game.actions[ac]
        print('V_hat_0(player) = %.2f' % val)
        if not self.policy_only:
            # MCTS with predicted values (instead of simulations)
            mcts_ac, root = run_mcts(self.config, self.game, self.model, discount=self.discount, use_cpu=self.use_cpu)
            mcts_move = self.game.actions[mcts_ac]
            # Statistics
            print('policy prefers', pol_move, root.children[ac].visit_count, 'mcts prefers', mcts_move, root.children[mcts_ac].visit_count)
            print('V_hat(player) = %.2f' % root.sampled_value(), root.is_first_player)
            hist = hist_tree_leaf_depth(root)
            print('search histogram', sorted(hist_tree_leaf_depth(root).items()), 'max depth', max(hist.keys()))
            return mcts_move
        return pol_move


def hist_tree_leaf_depth(root):
    counts = defaultdict(lambda: 0)
    queue = [(root, 0)]
    # Breadth-first iteration
    while 0 < len(queue):
        node, depth = queue.pop(0)
        visited_children = [child for child in node.children.values() if child.visit_count > 0]
        if len(visited_children) == 0:
            # Leaf node
            counts[depth] += 1
        else:
            # Internal node
            queue += itertools.product(visited_children, [depth + 1])
    return counts


if __name__ == '__main__':
    from functools import partial
    import torch
    from checkers.agents.mcts import MctsPlayer
    import sys
    from checkers.agents.alpha_beta import MinimaxPlayer, material_value_adv

    # Load the trained model
    # NOTE: only as good as random
    # model_path = 'logs/model-l155.pt'
    # NOTE: 0.8/0.2 with random, 0/1 against vanilla MCTS(400),
    # model_path = 'logs/model-399-l72.4.pt'
    # NOTE: 0.9/0.1 with random
    # model_path = 'logs/model-429-l56.6.pt'
    model_path = sys.argv[1]
    print('Model path:', model_path)
    # model = CheckersNetwork()
    # model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = CheckersNetwork().cuda()
    model.load_state_dict(torch.load(model_path))

    # model_early = CheckersNetwork().cuda()
    # model_early.load_state_dict(torch.load('logs/adam-0/model-789-l54.9.pt'))

    # Match setup
    n_matches = 10
    black_wins = 0
    white_wins = 0
    for i in range(n_matches):
        ch = Checkers()
        # Alpha-beta pruned minimax player
        white_player = MinimaxPlayer('white', value_func=partial(material_value_adv, 'white', 2, 1), search_depth=4, seed=1)
        # Vanilla MCTS player
        # white_player = MctsPlayer('white', exploration_coeff=1, max_rounds=400, seed=None)
        # white_player = RandomPlayer('white', seed=None)
        # white_player = AlphaZaiPlayer(model, 'white', num_simulations=100, policy_only=False, use_cpu=False, seed=None)
        # white_player = AlphaZaiPlayer(model_early, 'white', num_simulations=100, policy_only=False, use_cpu=False, seed=None)

        # AlphaZero player
        # black_player = RandomPlayer('black', seed=None)
        # black_player = AlphaZaiPlayer(model, 'black', policy_only=True, use_cpu=True)
        # black_player = AlphaZaiPlayer(model, 'black', num_simulations=100, policy_only=False, use_cpu=True)
        black_player = AlphaZaiPlayer(model, 'black', num_simulations=400, policy_only=False, use_cpu=False, seed=None)
        # black_player = MctsPlayer('black', exploration_coeff=1, max_rounds=100, seed=None)

        winner = play_a_game(ch, black_player.next_move, white_player.next_move, max_plies=300)
        black_wins += 1 if winner == 'black' else 0
        white_wins += 1 if winner == 'white' else 0
    print('Summary:', 'match', n_matches, 'black wins', black_wins / n_matches, 'white wins', white_wins / n_matches, 'draws', 1 - (black_wins + white_wins) / n_matches)
