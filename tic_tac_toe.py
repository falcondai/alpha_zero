import numpy as np

from pseudocode import Game


class TicTacToe(Game):
    def __init__(self, history=None):
        super().__init__(history)
        # There are nine possible moves, one for each square
        self.num_actions = 9
        # Current game state, the first player's squares and the second player's squares
        self.pieces = frozenset(), frozenset()

    def fast_forward(actions):
        pass

    def terminal(self):
        # If there are three connected pieces
        for player in self.pieces:
            # Rows
            if {0, 1, 2}.issubset(player):
                return True
            if {3, 4, 5}.issubset(player):
                return True
            if {6, 7, 8}.issubset(player):
                return True
            # Columns
            if {0, 3, 6}.issubset(player):
                return True
            if {1, 4, 7}.issubset(player):
                return True
            if {2, 5, 8}.issubset(player):
                return True
            # Diagonals
            if {0, 4, 8}.issubset(player):
                return True
            if {2, 4, 6}.issubset(player):
                return True
        return False

    def terminal_value(self, to_play: bool):
        # The value for the first player
        if to_play:
            pass

    def legal_actions(self):
        mine, theirs = self.pieces
        # Unoccupied squares
        unoccupied = set(range(9)).difference(mine).difference(theirs)
        return unoccupied

    def make_image(self, state_index: int):
        # Channels
        # 0 ~ empty
        # 1 ~ my piece
        # 2 ~ opponent's piece
        img = np.zeros((3, 3, 3))
        if state_index == -1:
            # Draw the current state
            mine, theirs = self.pieces
            for x in mine:
                img[x // 3, x % 3, 1] = 1
            for x in theirs:
                img[x // 3, x % 3, 2] = 1
        else:
            assert 0 <= state_index, 'state_index must be no less than -1.'
            # Draw the position from history[:state_index]
            pass
        return img

    def to_play(self):
        # Whether it is the first player's turn
        return len(self.history) % 2
