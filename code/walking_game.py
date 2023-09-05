import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import PIL
import cv2
import io
colorbar_exists = False


class WalkingGameState:
    def __init__(self, base_board, location_x=1, location_y=1, current_board=None, living_reward=0, living_prize=0):
        self.numbers_dict = {"death": -1, "free": 0, "reward": 1, "outside": 6, "burned": 2,
                             "lose": 3, "location": 4, "win": 5}
        base_board = np.array(base_board, dtype=np.int)
        if current_board is None:
            self.base_board = np.zeros(np.array(base_board.shape) + 2, dtype=int)
            self.base_board += self.numbers_dict["outside"]
            self.base_board[1: -1, 1: -1] = base_board
            self.current_board = self.base_board.copy()
        else:
            self.current_board = current_board
            self.base_board = base_board

        self.location_x = location_x
        self.location_y = location_y

        self.current_board[self.location_y, self.location_x] += self.numbers_dict["location"]
        self.rollout_budget = 100
        self.living_reward = living_reward
        self.living_prize = living_prize
        self.has_intermidiate_results = True

    def get_operations_options(self):
        optional_operation = ["up", "down", "left", "right"]
        illegal_moves = [self.numbers_dict["burned"], self.numbers_dict["outside"]]
        if self.current_board[self.location_y, self.location_x - 1] in illegal_moves:
            optional_operation.remove("left")
        if self.current_board[self.location_y, self.location_x + 1] in illegal_moves:
            optional_operation.remove("right")
        if self.current_board[self.location_y - 1, self.location_x] in illegal_moves:
            optional_operation.remove("up")
        if self.current_board[self.location_y + 1, self.location_x] in illegal_moves:
            optional_operation.remove("down")
        return optional_operation

    def apply_operation(self, operation):
        if operation == "up":
            new_x = self.location_x
            new_y = self.location_y - 1
        elif operation == "down":
            new_x = self.location_x
            new_y = self.location_y + 1
        elif operation == "left":
            new_x = self.location_x - 1
            new_y = self.location_y
        elif operation == "right":
            new_x = self.location_x + 1
            new_y = self.location_y
        else:
            raise ValueError("operation not legal")
        new_current_board = self.current_board.copy()
        new_current_board[self.location_y, self.location_x] = self.numbers_dict["burned"]
        return WalkingGameState(self.base_board, new_x, new_y, new_current_board,
                                living_reward=self.living_reward + self.living_prize)

    def is_terminal(self):
        stuck = self.get_operations_options() == []
        goal = self.base_board[self.location_y, self.location_x] != 0
        return stuck or goal

    def is_a_win(self):
        return self.base_board[self.location_y, self.location_x] == 1

    def is_a_lose(self):
        stuck = self.get_operations_options() == []
        hall = self.base_board[self.location_y, self.location_x] == -1
        return stuck or hall

    def get_result(self):
        stuck = self.get_operations_options() == []
        if stuck:
            return -1
        else:
            return self.base_board[self.location_y, self.location_x] + self.living_reward

    @staticmethod
    def compute_bfs_till_win(start):
        Q = []
        Q.append((start, 0))
        bfs_from_start_dict = {start: None}
        while Q:
            current_state, depth = Q.pop(0)
            for operation in current_state.get_operations_options():
                child = current_state.apply_operation(operation)
                # if the children is not in the list and not in the queue
                if not (child in bfs_from_start_dict):
                    if not (child.get_result() == -1):
                        Q.append((child, depth + 1))
                        bfs_from_start_dict[child] = depth + 1
                        if child.is_a_win():
                            return child, depth + 1
        return "no possible wins here"

    def __eq__(self, other):
        board_the_same = (self.current_board == other.current_board).all()
        same_location = (self.location_x == other.location_x) and (self.location_y == other.location_y)
        return board_the_same and same_location

    def __hash__(self):
        return hash(self.current_board.tobytes() + self.location_y.to_bytes(1, "little") + self.location_x.to_bytes(1,
                                                                                                                    "little"))

    def __repr__(self):
        board_str = "WalkingGameState:"
        for row in self.current_board:
            board_str += "\n"
            for num in row:
                if num == self.numbers_dict["free"]:
                    board_str += " _ "
                if num == self.numbers_dict["location"]:
                    board_str += " @ "
                elif num == self.numbers_dict["win"]:
                    board_str += " W "
                elif num == self.numbers_dict["lose"]:
                    board_str += " L "
                elif num == self.numbers_dict["outside"]:
                    board_str += " * "
                elif num == self.numbers_dict["burned"]:
                    board_str += " * "
                elif num == -1:
                    board_str += "-1 "
                elif num == 1:
                    board_str += " 1 "
                else:
                    assert ValueError("invalid board values")
            # board_str += "]"
        return board_str

    def display_board(self):
        global colorbar_exists
        color_name = ['black', 'cyan', 'magenta', 'orange', 'red', "pink", 'green', 'gray']
        labels = ['death', 'free', 'reward', 'burned', 'lose', "location", 'win', 'outside']
        cmap = colors.ListedColormap(color_name)

        # Plot the matrix with color mapping
        fig, ax = plt.subplots()
        plt.imshow(self.current_board, cmap=cmap)

        # Add colorbar legend
        if not colorbar_exists:
            colorbar_exists = True
            cbar = plt.colorbar()
            cbar.set_ticks(np.arange(-1, 7) * 0.9 + 0.2)
            cbar.set_ticklabels(labels)

        # Render the figure to a binary buffer
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)

        # img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("walking game", img)
        cv2.waitKey(1)
        plt.close()

    def save_board(self, path):
        global colorbar_exists
        color_name = ['black', 'cyan', 'magenta', 'orange', 'red', "pink", 'green', 'gray']
        labels = ['death', 'free', 'reward', 'burned', 'lose', "location", 'win', 'outside']
        cmap = colors.ListedColormap(color_name)

        # Plot the matrix with color mapping
        fig, ax = plt.subplots()
        plt.imshow(self.current_board, cmap=cmap)



        # Render the figure to a binary buffer
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)

        # img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
        plt.close()


class WalkingGameStateLongWalks(WalkingGameState):
    def __init__(self, base_board, location_x=1, location_y=1, current_board=None, living_reward=0):
        living_prize = 0.1
        super().__init__(base_board, location_x, location_y, current_board, living_reward, living_prize=living_prize)


class WalkingGameStateShortWalks(WalkingGameState):
    def __init__(self, base_board, location_x=1, location_y=1, current_board=None, living_reward=0):
        living_prize = 0
        super().__init__(base_board, location_x, location_y, current_board, living_reward, living_prize=living_prize)


