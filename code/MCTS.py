import matplotlib.pyplot as plt
import numpy as np
import time
from utils import random_argmax_choice

class GameState:
    def __init__(self):
        pass

    def get_operations_options(self):
        return (1, 2, 3)

    def apply_operation(self, operation):
        return GameState()

    def is_terminal(self):
        return False

    def get_result(self):
        return 1


class MCSTNode:
    def __init__(self, game_state, parent=None, select_child_method="UCT"):
        self.game_state = game_state
        self.reward = [0]
        self.visits = 0
        self.children = []
        self.parent = parent
        self.is_expanded = False
        self.select_child_method = select_child_method

    def select_child(self, exploration_constant):
        children_visits = [child.visits for child in self.children]
        if 0 in children_visits:
            return self.children[np.argmin(children_visits)]  # return a child that has not yet been visited
        if self.select_child_method == "UCT":
            uct_scores = [self.calculate_uct_score(child, exploration_constant) for child in self.children]
            selected_child = self.children[random_argmax_choice(uct_scores)]
        elif self.select_child_method == "epsilon_greedy":
            rewards = [np.mean(child.reward) for child in self.children]
            selected_child = self.children[random_argmax_choice(rewards)]
            select_the_best_child = np.random.randint(2)  # 50% for 0 and 50% for 1
            if not select_the_best_child:  # 50% to select the best child 50% to select other random child
                if len(self.children) > 1:
                    choices = [idx for idx in range(len(self.children)) if idx != np.argmax(rewards)]
                    selected_idx = np.random.choice(choices)
                    selected_child = self.children[selected_idx]

        else:
            raise ValueError("select_child_method does not exist")
        return selected_child

    def calculate_uct_score(self, child, exploration_constant):
        exploitation = np.mean(child.reward) / child.visits
        exploration = np.sqrt(np.log(self.visits) / child.visits)
        return exploitation + (exploration_constant * exploration)

    def expand(self):
        # generate all successors
        for operation in self.game_state.get_operations_options():
            child = MCSTNode(game_state=self.game_state.apply_operation(operation), parent=self,
                             select_child_method=self.select_child_method)
            self.children.append(child)
        self.is_expanded = True

    def rollout(self):
        # rollout a game from the current game_state to a terminal state
        current_state = self.game_state
        if current_state.rollout_budget:
            for _ in range(current_state.rollout_budget):
                operations = current_state.get_operations_options()
                if current_state.is_terminal() or (operations == []):
                    break
                operation = operations[np.random.randint(len(operations))]
                current_state = current_state.apply_operation(operation)
        else:
            while not current_state.is_terminal():
                operation = np.random.choice(current_state.get_operations_options())
                current_state = current_state.apply_operation(operation)
        result = current_state.get_result()
        return result

    def back_propagate(self, result):
        self.visits += 1
        self.reward += [result]
        if self.parent:
            self.parent.back_propagate(result)

    def __repr__(self):
        return "\n" + repr(self.game_state) + f"\nvisits:{self.visits},  reward:{np.mean(self.reward)}\n"

    def __hash__(self):
        return hash(self.game_state)


class MonteCarloTreeSearch:
    def __init__(self, root_state, exploration_constant, choose_best_move_func, iterations_per_move,
                 time_per_move, display_game=False, time_limit4game=False, select_child2expand_method="UCT"):
        self.root = MCSTNode(root_state, select_child_method=select_child2expand_method)
        self.iterations_per_move = iterations_per_move
        self.time_per_move = time_per_move
        self.exploration_constant = exploration_constant
        self.root.expand()
        self.choose_best_move_func = choose_best_move_func
        # self.expanded_dict = {self.root: True}
        self.display_game = display_game
        self.time_limit4game = time_limit4game
        if time_limit4game:
            self.start_game_time = time.time()

    def its_time_to_move(self):
        start_timer = time.time()
        for _ in range(self.iterations_per_move):
            if (time.time() - start_timer) > self.time_per_move:
                break
            yield False
        yield True

    def compute_best_move(self):
        finish_thinking = self.its_time_to_move()
        while not next(finish_thinking):
            node = self.root  # start from the root

            # select the best node to expand
            while node.is_expanded:
                node = node.select_child(self.exploration_constant)

            if node.game_state.is_terminal():
                result = node.game_state.get_result()
            else:
                node.expand()
                # self.expanded_dict[node] = True
                result = np.random.choice(node.children).rollout()
                if node.game_state.has_intermidiate_results:
                    result += node.game_state.get_result()
            node.back_propagate(result)

        # if there is a winner return it
        winning_children = [child.game_state.is_a_win() for child in self.root.children]
        if True in winning_children:
            return self.root.children[np.argmax(winning_children)]
        else:  # return the best move
            best_child = self.choose_best_move_func(self.root.children)
            return best_child

    def search(self):
        moves = 0
        self.root.game_state.display_board()
        while not self.root.game_state.is_terminal():
            self.root = self.compute_best_move()
            moves += 1
            # print(f"moves: {moves}")
            if self.display_game:
                self.root.game_state.display_board()
            if self.time_limit4game:
                if (time.time() - self.start_game_time) > self.time_limit4game:
                    break
        return self.root, moves

    def get_path(self):
        current_root = self.root
        path = [current_root]
        while current_root.parent:
            current_root = current_root.parent
            path.append(current_root)
        path.reverse()
        return path

