from typing import List

class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Game(object):
    '''Abstraction of a turn-taking board game between two players'''

    def __init__(self, history=None):
        self.history = history or []
        self.child_visits = []
        self.num_actions = 0

    def terminal(self) -> bool:
        '''Whether the game has ended'''
        # Game specific termination rules.
        pass

    def terminal_value(self, to_play) -> float:
        '''The terminal value for the first player'''
        pass

    def legal_actions(self):
        '''Returns a collection of immutable actions that are legal in the current state'''
        return []

    def clone(self):
        '''Make a copy of the game'''
        return Game(list(self.history))

    def apply(self, action):
        '''Take action and advance the game state'''
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(
            child.visit_count for child in root.children.itervalues())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        '''A stacked 2D representation of the board game state'''
        return []

    def make_target(self, state_index: int):
        '''Return the training targets of terminal game value given full history and the MCTS* visit counts'''
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self) -> bool:
        '''Whether this is the first player's turn'''
        return len(self.history) % 2
