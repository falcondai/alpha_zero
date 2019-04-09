import numpy as np


class Node(object):
    '''Node for keeping the rollout statistics. They are keyed by actions'''

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        # Successor nodes keyed by legal actions
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def sampled_value(self):
        # Empirical mean of the simulations, as evaluated by the learned value function
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Game(object):
    '''Abstraction of a turn-taking zero-sum board game between two players. Crucially, it needs to support two needs: game state updates (gameplay) and sampling of steps in finished games (training).

    In particular, actions are represented by immutable objects (convenient as indices). Given a history of legal actions, we can re-create (fast-forward) the game state.

    In addition, we take the absolute description of a game (similar to a referee's) and the two players will be identified by their turns, as the first-moving player and the second-moving player.'''

    # The starting state of a game
    initial_board = None

    def __init__(self, history=[]):
        # Rollout statistics
        self.child_visits = []
        # Action space
        self.num_actions = 0
        self.actions = []
        # Initialize game states
        self.board = self.initial_board
        # First moving player starts
        self.turn = 0
        # Terminal values for the first player
        # 1 for win
        # 0 for draw
        # -1 for loss
        self.game_value = None
        # Fast forward to the state after taking actions in history
        self.history = history
        for action in self.history:
            self.apply(action)

    def is_first_player_turn(self) -> bool:
        '''Whether this is the first player's turn'''
        return self.turn == 0

    def terminal(self) -> bool:
        '''Whether the game has ended'''
        return self.game_value is not None

    def legal_actions(self):
        '''Returns a collection of immutable actions that are legal in the current state for the playing player'''
        return []

    def apply(self, action):
        '''Apply action and advance the game state'''
        self.history.append(action)
        # Update game state
        raise NotImplementedError

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.itervalues())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits
            if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def clone(self):
        '''Make a copy of the game'''
        return Game(list(self.history))

    def ego_board_representation(self):
        '''A stacked 2D representation of the board state from the ego-centric perspective of the current player'''
        raise NotImplementedError

    def ego_sample(self, state_index: int):
        '''Return a training sample from a finished game'''
        game = self.__class__(list(self.history[:state_index]))
        # Ego-centric views of the current player
        rep = game.ego_board_representation()
        # Zero-sum game
        ego_val = self.game_value if game.is_first_player_turn() else 0 - self.game_value
        return rep, ego_val, self.child_visits[state_index]

    def terminal_value(self) -> float:
        '''The terminal value for the first player'''
        if self.game_value is None:
            raise Exception('The game has not finished.')
        return self.game_value


class Network(object):
    '''
    The model will predict the value and moves for the current player given the board represented in an ego-centric view. Need to be careful to flip the pieces and positions to make the board representation consistent for both players, e.g. the forward direction in chess.
    '''

    def inference(self, image):
        # Ego-centric value and policy
        return (0, {})

    def get_weights(self):
        # Returns the weights of this network.
        return []


class AlphaZeroConfig(object):
    '''Hyperparameters'''

    def __init__(self):
        '''Default values from the AlphaZero paper'''
        # Self-Play
        self.num_actors = 5000

        # A few starting moves are non-greedy (helps with exploring openings?)
        self.num_sampling_moves = 30
        # Maximum length of a game
        # 512 for chess and shogi, 722 for Go.
        self.max_moves = 512
        # Number of rollouts
        self.num_simulations = 800

        # Root prior exploration noise.
        # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Training
        self.training_steps = int(700e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 4096

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            100e3: 2e-2,
            300e3: 2e-3,
            500e3: 2e-4
        }


class ReplayBuffer(object):
    def __init__(self, config: AlphaZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        # Sample uniformly across positions.
        # There are one more state at the end, after applying all moves.
        move_sum = float(sum(len(g.history) + 1 for g in self.buffer))
        games = np.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[(len(g.history) + 1) / move_sum for g in self.buffer])
        game_pos = [(g, np.random.randint(len(g.history) + 1)) for g in games]
        return [g.ego_sample(i) for (g, i) in game_pos]


class SharedStorage(object):
    def __init__(self, initialize_network):
        self._networks = {
            -1: initialize_network(),
        }

    def latest_network(self) -> Network:
        return self._networks[max(self._networks.iterkeys())]

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
