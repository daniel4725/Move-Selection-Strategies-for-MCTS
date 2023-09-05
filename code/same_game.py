import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import PIL
import cv2
import io


def get_samegame_board(number):
    np.random.seed(number)
    return np.random.randint(1, 5, (8, 8))


class SameGameState:
    def __init__(self, board, scores=0, former_move=None):
        self.board = copy.deepcopy(board)
        self.scores = scores
        self.former_move = former_move
        self.rows, self.cols = board.shape
        self.rollout_budget = 20
        self.has_intermidiate_results = False


    def get_operations_options(self):
        # each non zero coordinate
        optional_operation = [(y, x) for y, x in zip(np.where(self.board)[0], np.where(self.board)[1])]
        return optional_operation

    def apply_operation(self, operation):
        new_board, scores = self.click_on_coordinate(operation)
        return SameGameState(new_board, scores=scores + self.scores, former_move=operation)

    def click_on_coordinate(self, operation):
        new_board = copy.deepcopy(self.board)
        self.zero_connected_component(new_board, operation)
        num_of_zeroed = ((new_board == 0).astype(int) - (self.board == 0)).astype(int).sum()
        scores = (num_of_zeroed - 1) ** 2
        return self.squeeze_board(new_board), scores

    def squeeze_to_bottom(self, board):
        for col in range(self.cols):
            stack = []
            for row in range(self.rows):
                if board[row][col] != 0:
                    stack.append(board[row][col])
                    board[row][col] = 0

            current_row = self.rows - 1
            while stack:
                board[current_row][col] = stack.pop()
                current_row -= 1
        return board

    def squeeze_to_left(self, board):
        move_cnt = 0
        for col in range(self.cols):
            board[:, col - move_cnt] = board[:, col]
            if (board[:, col] == 0).all():
                move_cnt += 1
        if move_cnt > 0:
            board[:, -move_cnt:] = 0

        return board

    def squeeze_board(self, board):
        board = self.squeeze_to_bottom(board)
        board = self.squeeze_to_left(board)
        return board

    def zero_connected_component(self, board, operation):
        target = board[operation[0], operation[1]]

        def dfs(r, c):
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return

            if board[r, c] != target:
                return

            board[r, c] = 0

            # Perform DFS on neighboring cells
            dfs(r - 1, c)  # Up
            dfs(r + 1, c)  # Down
            dfs(r, c - 1)  # Left
            dfs(r, c + 1)  # Right

        dfs(operation[0], operation[1])

    def is_terminal(self):
        all_zeros = not self.board.any()
        return all_zeros

    def is_a_win(self):
        return False

    def get_result(self):
        return self.scores

    def __repr__(self):
        board_str = "SameGameState:"
        for row in self.board:
            board_str += "\n" + str(row)
        return board_str

    def display_board(self):
        colors = {
            0: (0, 0, 0),
            1: (0, 0, 255),
            2: (255, 0, 0),
            3: (255, 255, 0),
            4: (0, 255, 0),
            5: (0, 255, 255),
            6: (255, 0, 255),
        }
        screen = np.zeros((self.board.shape[0], self.board.shape[1], 3), dtype=int)
        # Plot the matrix with color mapping

        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                screen[y, x] = colors[cell]

        fig, ax = plt.subplots()
        plt.imshow(screen)

        # Render the figure to a binary buffer
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)

        # img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("same game", img)
        cv2.waitKey(1)
        plt.close()



if __name__ == "__main__":
    board = np.random.randint(1, 5, (15, 15))

    s = SameGameState(board)
    s.click_on_coordinate((1, 0))
    s.display_board()
    s.apply_operation((1, 0)).display_board()
    print(s)
