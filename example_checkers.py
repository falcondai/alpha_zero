import numpy as np

from checkers import Checkers
from base import Game


class CheckersGame(Game):
    def __init__(self, history=[]):
        # Rollout statistics
        self.child_visits = []
        # Terminal values for the first player
        # 1 for win
        # 0 for draw
        # -1 for loss
        self.game_value = None

        # Conventions:
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
        self.history = history
        for action in self.history:
            self.apply(action)

    def apply(self, action_index):
        from_sq, to_sq = self.actions[action_index]
        board, turn, last_moved_piece, all_next_moves, winner = self.ch.move(from_sq, to_sq)

        # Terminate when one player wins
        if self.winner == 'black':
            self.game_value = 1
        elif self.winner == 'white':
            self.game_value = -1

        self.history.append(action_index)

    def legal_actions(self):
        moves = self.ch.legal_moves()
        action_idices = {self.action2ind[move] for move in moves}
        return action_idices

    def is_first_player_turn(self):
        return self.turn == 'black'

    def ego_board_representation(self):
        # Channels
        # 0 my men
        # 1 my kings
        # 2 opponent's men
        # 3 opponent's kings
        # 4 my last moved piece
        rep = np.zeros((self.ch.size, self.ch.size, 5))
        if self.turn == 'white':
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
            if self._last_moved_piece is not None:
                row, col = self.ch.sq2pos(self._last_moved_piece)
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
            if self._last_moved_piece is not None:
                sq = self.abs2ego_sq[self._last_moved_piece]
                row, col = self.ch.sq2pos(sq)
                rep[row, col, 4] = 1
        return rep


if __name__ == '__main__':
    game = CheckersGame()
    for i, ac in enumerate(game.actions):
        print(i, ac)
    game.ch.print_empty_board()
    acs = game.legal_actions()
    print(acs)
    for ac in acs:
        print(ac, game.abs2ego_ac[ac])
