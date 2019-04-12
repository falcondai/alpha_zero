import numpy as np
import torch
from torch import nn

from checkers import Checkers
from base import Game, Network


class CheckersGame(Game):
    def __init__(self, history=[]):
        # Rollout statistics
        self.child_visits = []
        # Terminal values for the first player
        # 1 for win
        # 0 for draw
        # -1 for loss
        # None for incomplete
        self.game_value = None

        # XXX Conventions:
        # - Black player moves first
        # - Ego-centric views assume the king row are at the top, i.e. starts at the bottom (Second player has the same view as absolute)
        self.ch = Checkers()

        # Action space
        self.actions = []
        # Simple moves
        for from_sq in range(self.ch.n_positions):
            for to_sq in self.ch.neighbors[from_sq]:
                if to_sq is not None:
                    simple_move = (from_sq, to_sq)
                    self.actions.append(simple_move)

        assert 98 == len(self.actions), 'There should be 98 simple moves.'
        # Jumps
        for from_sq in range(self.ch.n_positions):
            row, col = self.ch.sq2pos(from_sq)
            # For each direction
            for di, (drow, dcol) in enumerate(Checkers.dir2del):
                next_row, next_col = row + 2 * drow, col + 2 * dcol
                if 0 <= next_row < self.ch.size and 0 <= next_col < self.ch.size:
                    # Within bound
                    to_sq = self.ch.pos2sq(next_row, next_col)
                    jump = (from_sq, to_sq)
                    self.actions.append(jump)
        self.num_actions = len(self.actions)
        assert 98 + 72 == self.num_actions, 'There should be 98 simple moves and 72 jumps.'
        # Inverse dictionary
        self.action2ind = {action: ind for ind, action in enumerate(self.actions)}
        # Square mapping from absolute to first player's ego-centric (reflect through the center)
        self.abs2ego_sq = {}
        for sq in range(self.ch.n_positions):
            row, col = self.ch.sq2pos(sq)
            re_row, re_col = -row + self.ch.size - 1, -col + self.ch.size - 1
            re_sq = self.ch.pos2sq(re_row, re_col)
            self.abs2ego_sq[sq] = re_sq
        # Inverse
        self.ego2abs_sq = {re_sq: sq for sq, re_sq in self.abs2ego_sq.items()}

        # Move mapping from absolute to first player's ego-centric
        self.abs2ego_ac = {}
        for ac, (from_sq, to_sq) in enumerate(self.actions):
            ego_move = (self.abs2ego_sq[from_sq], self.abs2ego_sq[to_sq])
            ego_ac = self.action2ind[ego_move]
            self.abs2ego_ac[ac] = ego_ac
        # Inverse
        self.ego2abs_ac = {ego_ac: ac for ac, ego_ac in self.abs2ego_ac.items()}

        # Fast forward to the last state by taking actions from history
        self.history = []
        for action in history:
            self.apply(action)

    def clone(self):
        game = CheckersGame()
        state = self.ch.save_state()
        game.ch.restore_state(state)
        return game

    def apply(self, action_index):
        from_sq, to_sq = self.actions[action_index]
        board, turn, last_moved_piece, all_next_moves, winner = self.ch.move(from_sq, to_sq)

        # Terminate when one player wins
        if winner == 'black':
            self.game_value = 1
        elif winner == 'white':
            self.game_value = -1

        self.history.append(action_index)

    def legal_actions(self):
        moves = self.ch.legal_moves()
        action_idices = {self.action2ind[move] for move in moves}
        return action_idices

    def is_first_player_turn(self):
        return self.ch.turn == 'black'

    def ego_board_representation(self):
        # XXX Channels
        # 0 my men
        # 1 my kings
        # 2 opponent's men
        # 3 opponent's kings
        # 4 my last moved piece
        # QUESTION: try indicating the king row and skipping ego transform?
        rep = np.zeros((self.ch.size, self.ch.size, 5))
        if self.ch.turn == 'white':
            # Same as the absolute view
            for sq in self.ch.board['white']['men']:
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 0] = 1
            for sq in self.ch.board['white']['kings']:
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 1] = 1
            for sq in self.ch.board['black']['men']:
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 2] = 1
            for sq in self.ch.board['black']['kings']:
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 3] = 1
            if self.ch._last_moved_piece is not None:
                row, col = self.ch.sq2pos(self.ch._last_moved_piece)
                rep[row, col, 4] = 1
        else:
            # Need to invert the board
            for sq in self.ch.board['black']['men']:
                sq = self.abs2ego_sq[sq]
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 0] = 1
            for sq in self.ch.board['black']['kings']:
                sq = self.abs2ego_sq[sq]
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 1] = 1
            for sq in self.ch.board['white']['men']:
                sq = self.abs2ego_sq[sq]
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 2] = 1
            for sq in self.ch.board['white']['kings']:
                sq = self.abs2ego_sq[sq]
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 3] = 1
            if self.ch._last_moved_piece is not None:
                sq = self.abs2ego_sq[self.ch._last_moved_piece]
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 4] = 1
        return rep

    def ego_sample(self, state_index: int):
        # Fast forward
        game = CheckersGame(list(self.history[:state_index]))
        # Ego-centric views of the current player
        rep = game.ego_board_representation()
        # Zero-sum game
        ego_val = self.game_value if game.is_first_player_turn() else (0 - self.game_value)
        # Ego-centric actions
        if game.is_first_player_turn():
            # Invert actions for the first player
            visits = np.zeros(self.num_actions)
            for i in range(self.num_actions):
                visits[self.abs2ego_ac[i]] = self.child_visits[state_index][i]
        else:
            visits = np.asarray(self.child_visits[state_index])
        return rep, ego_val, visits

    def ego2abs_policy(self, is_first_player, ego_policy):
        if is_first_player:
            policy = np.zeros(self.num_actions)
            for ego_ac, pr in enumerate(ego_policy):
                policy[self.ego2abs_ac[ego_ac]] = pr
        else:
            policy = ego_policy
        return policy


class CheckersNetwork(nn.Module, Network):
    '''
    Based on the architecture of AlphaGo Zero. Convolutions with residual connections.

    Ref: _Mastering the game of Go without human knowledge_ by Silver et al.
    https://www.nature.com/articles/nature24270.pdf
    '''

    def __init__(self):
        # Checkers
        self.board_size = 8
        self.num_actions = 170
        # AlphaGo Zero uses 19 or 39
        self.num_residual_blocks = 2

        super().__init__()
        # Parameters for each layer
        # Convolution
        self.conv = nn.Conv2d(in_channels=5, out_channels=256, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(256)

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
            batch_norm1 = nn.BatchNorm2d(256)

            conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
            batch_norm2 = nn.BatchNorm2d(256)
            residual_block = nn.ModuleList((conv1, batch_norm1, conv2, batch_norm2))
            self.residual_blocks.append(residual_block)

        # Policy head
        self.policy_conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.policy_batch_norm = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(self.board_size * self.board_size * 2, self.num_actions)

        # Value head
        self.value_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.value_batch_norm = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_size * self.board_size * 1, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Visualize the model
        print(self)
        print('# of parameters', sum(param.nelement() for param in self.parameters()))

    def forward(self, im):
        # Conv
        net = self.conv(im)
        net = self.batch_norm(net)
        net = nn.functional.relu(net)

        # Residual blocks
        for conv1, batch_norm1, conv2, batch_norm2 in self.residual_blocks:
            input = net
            net = conv1(net)
            net = batch_norm1(net)
            net = nn.functional.relu(net)
            net = conv2(net)
            net = batch_norm2(net)
            # Residual connection
            net += input
            net = nn.functional.relu(net)

        # Heads
        # Policy logits
        policy_net = self.policy_conv(net)
        policy_net = self.policy_batch_norm(policy_net)
        policy_net = nn.functional.relu(policy_net)
        policy_net = self.policy_fc(policy_net.view(-1, self.board_size * self.board_size * 2))

        # Value
        value_net = self.value_conv(net)
        value_net = self.value_batch_norm(value_net)
        value_net = nn.functional.relu(value_net)
        value_net = self.value_fc1(value_net.view(-1, self.board_size * self.board_size * 1))
        value_net = nn.functional.relu(value_net)
        value_net = self.value_fc2(value_net)
        value_net = torch.tanh(value_net)

        return value_net, policy_net

    def inference(self, ego_board_rep):
        # NOTE: PyTorch channel convention, BCHW from TF convention BHWC.
        torch_rep = ego_board_rep.transpose(1, 3)
        # torch_rep = np.transpose(ego_board_rep, (0, 3, 1, 2))
        # torch_rep = np.ascontiguousarray(torch_rep, dtype=np.float32)
        return self.forward(torch_rep)

    def single_inference(self, ego_board_rep, use_cpu=False):
        # Single board, unsqueeze
        ego_board_rep = ego_board_rep[None, :]
        ego_board_rep = np.ascontiguousarray(ego_board_rep, dtype=np.float32)
        ego_board_rep = torch.from_numpy(ego_board_rep)
        if not use_cpu:
            ego_board_rep = ego_board_rep.cuda()
        self.eval()
        vals, logits = self.inference(ego_board_rep)
        return vals[0, 0].detach().cpu().numpy(), logits[0].detach().cpu().numpy()


def make_uniform_network():
    return Network(170)


if __name__ == '__main__':
    import os
    from base import AlphaZeroConfig, SharedStorage, ReplayBuffer
    from zero import play_game
    from torch import optim

    # game = CheckersGame()
    # for i, ac in enumerate(game.actions):
    #     print(i, ac)
    # game.ch.print_empty_board()
    # acs = game.legal_actions()
    # print(acs)
    #
    # while len(acs) > 0:
    #     ac = acs.pop()
    #     game.apply(ac)
    #     acs = game.legal_actions()
    # game.ch.print_board()
    # print(game.ch.turn)
    #
    # rep = game.ego_board_representation()
    # print(rep[:, :, 0])
    # print(rep[:, :, 1])
    # print(rep[:, :, 2])
    # print(rep[:, :, 3])
    # print(rep[:, :, 4])
    #
    # print(game.terminal_value())
    # print(game.history)
    # # Fake visit counts for testing
    # game.child_visits += [list(range(game.num_actions))] * len(game.history)
    # print(game.child_visits)
    # print(game.ego_sample(20))

    # # Play with MCTS
    # config = AlphaZeroConfig()
    # config.num_simulations = 100
    # # model = make_uniform_network()
    # model = CheckersNetwork()
    # ga = play_game(config, CheckersGame, model)
    # print(ga.child_visits)

    # AlphaZero
    log_dir = 'logs/adam-0-1/'
    # Train for a few steps
    config = AlphaZeroConfig()
    config.num_simulations = 100
    config.window_size = 64
    config.batch_size = 32
    config.num_sampling_moves = 20
    # A typical competitive Checkers game lasts for ~49 half-moves
    # Ref: https://boardgames.stackexchange.com/questions/34659/how-many-turns-does-an-average-game-of-checkers-draughts-go-for
    config.max_moves = 200

    storage = SharedStorage(make_uniform_network)
    buffer = ReplayBuffer(config)

    model = CheckersNetwork()
    model.cuda()
    # HACK: Continue from adam-0/
    model.load_state_dict(torch.load('logs/adam-0/model-789-l54.9.pt'))
    storage.save_network(0, model)
    # optimizer = optim.SGD(model.parameters(), lr=2e-2, momentum=config.momentum, weight_decay=config.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=config.weight_decay)
    val_loss = nn.MSELoss(reduction='sum')

    for step in range(2000):
        # Generate some games
        for i in range(1):
            actor = storage.latest_network()
            game = play_game(config, CheckersGame, actor)
            buffer.save_game(game)
        # Update model
        batch = buffer.sample_batch()
        boards = np.zeros((config.batch_size, 8, 8, 5), dtype=np.float32)
        vals = np.zeros(config.batch_size, dtype=np.float32)
        dists = np.zeros((config.batch_size, 170), dtype=np.float32)
        for i, (board, val, dist) in enumerate(batch):
            boards[i] = board
            vals[i] = val
            dists[i] = dist
        # Forward
        model.train()
        model.zero_grad()
        boards = torch.from_numpy(boards).cuda()
        vals = torch.from_numpy(vals).cuda().view(-1, 1)
        dists = torch.from_numpy(dists).cuda()
        val_hats, logits = model.inference(boards)
        # Compute loss
        val_loss = nn.functional.mse_loss(val_hats, vals, reduction='sum')
        policy_loss = (- dists * nn.functional.log_softmax(logits, 1)).sum()
        loss = val_loss + policy_loss
        print('step', step, val_loss, policy_loss, loss)
        loss.backward()
        optimizer.step()
        # Save model
        storage.save_network(step, model)
        if step % 30 == 9:
            # Commit trained model to disk
            print('Saving model...')
            torch.save(model.state_dict(), os.path.join(log_dir, 'model-%i-l%.1f.pt' % (step, loss)))
    # Last checkpoint
    print('Saving model...')
    torch.save(model.state_dict(), os.path.join(log_dir, 'model-%i-l%.1f.pt' % (step, loss)))
