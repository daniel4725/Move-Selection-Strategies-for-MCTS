import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import PIL
import cv2
import io

grid_x_count = 10
grid_y_count = 10

class Keys:
    def __init__(self):
        self.UP = "up"
        self.DOWN = "down"
        self.LEFT = "left"
        self.RIGHT = "right"

keys = Keys()

snake_init_position = dict(
    direction_queue=['right'],
    snake_segments=[
        {'x': 2, 'y': 0},
        {'x': 1, 'y': 0},
        {'x': 0, 'y': 0},
    ],
    food_position=None,
    snake_alive=True,
    just_ate=False,
    living_cost=0
)

class SnakeGameState:
    def __init__(self, current_state=None):
        # direction_queue, snake_segments, snake_alive, food_position
        if current_state is None:
            self.current_state = copy.deepcopy(snake_init_position)
        else:
            self.current_state = copy.deepcopy(current_state)
        self.direction_queue = self.current_state["direction_queue"]
        self.snake_segments = self.current_state["snake_segments"]
        self.snake_alive = self.current_state["snake_alive"]
        self.food_position = self.current_state["food_position"]
        self.living_cost = self.current_state["living_cost"]
        self.has_intermidiate_results = True

        self.tuple_state = tuple(tuple(row) for row in self.current_state)
        self.dead_reward = -5
        self.eat_reward = 10
        self.update_price = -0.1
        if self.current_state["just_ate"]:
            self.reward = self.eat_reward
        else:
            self.reward = 0
        self.rollout_budget = 50
        if self.food_position is None:
            self.food_position = self.move_food()

    def move_food(self):
        possible_food_positions = []

        for food_x in range(grid_x_count):
            for food_y in range(grid_y_count):
                possible = True

                for segment in self.snake_segments:
                    if food_x == segment['x'] and food_y == segment['y']:
                        possible = False

                if possible:
                    possible_food_positions.append({'x': food_x, 'y': food_y})

        food_position = random.choice(possible_food_positions)
        return food_position

    def update(self, direction_queue=None):
        if direction_queue is None:
            direction_queue = copy.deepcopy(self.direction_queue)
        snake_segments = copy.deepcopy(self.snake_segments)
        snake_alive = self.snake_alive
        food_position = copy.deepcopy(self.food_position)
        just_ate = False
        living_cost = self.living_cost


        if len(direction_queue) > 1:
            direction_queue.pop(0)

        next_x_position = snake_segments[0]['x']
        next_y_position = snake_segments[0]['y']

        if direction_queue[0] == 'right':
            next_x_position += 1
            if next_x_position >= grid_x_count:
                next_x_position = 0

        elif direction_queue[0] == 'left':
            next_x_position -= 1
            if next_x_position < 0:
                next_x_position = grid_x_count - 1

        elif direction_queue[0] == 'down':
            next_y_position += 1
            if next_y_position >= grid_y_count:
                next_y_position = 0

        elif direction_queue[0] == 'up':
            next_y_position -= 1
            if next_y_position < 0:
                next_y_position = grid_y_count - 1

        can_move = True

        for segment in snake_segments[:-1]:
            if (next_x_position == segment['x']
                    and next_y_position == segment['y']):
                can_move = False

        if can_move:
            snake_segments.insert(0, {'x': next_x_position, 'y': next_y_position})

            if (snake_segments[0]['x'] == food_position['x']
                    and snake_segments[0]['y'] == food_position['y']):
                food_position = self.move_food()
                just_ate = True
            else:
                snake_segments.pop()
        else:
            snake_alive = False

        next_state = dict(
            direction_queue=direction_queue,
            snake_segments=snake_segments,
            food_position=food_position,
            snake_alive=snake_alive,
            just_ate=just_ate,
            living_cost=living_cost + self.update_price
        )
        return next_state

    def get_operations_options(self):
        optional_operation = ["up", "down", "left", "right"]
        for operation in copy.deepcopy(optional_operation):
            if self.apply_operation(operation).current_state == self.current_state:
                optional_operation.remove(operation)
        return optional_operation


    def apply_operation(self, operation):
        if (operation == "no-op"):
            next_state = self.update()
        else:
            direction_queue = self.on_key_down(operation)
            next_state = self.update(direction_queue)
        return SnakeGameState(next_state)

    def on_key_down(self, key):
        direction_queue = copy.deepcopy(self.direction_queue)
        if (key == keys.RIGHT
                and direction_queue[-1] != 'right'
                and direction_queue[-1] != 'left'):
            direction_queue.append('right')

        elif (key == keys.LEFT
              and direction_queue[-1] != 'left'
              and direction_queue[-1] != 'right'):
            direction_queue.append('left')

        elif (key == keys.DOWN
              and direction_queue[-1] != 'down'
              and direction_queue[-1] != 'up'):
            direction_queue.append('down')

        elif (key == keys.UP
              and direction_queue[-1] != 'up'
              and direction_queue[-1] != 'down'):
            direction_queue.append('up')
        return direction_queue

    def is_terminal(self):
        return self.is_a_win() or (not self.snake_alive)

    def is_a_win(self):
        return False

    def get_result(self):
        if self.snake_alive:
            return self.reward
        else:
            return self.dead_reward + len(self.snake_segments)

    def __eq__(self, other):
        return self.current_state == other.current_state

    def __hash__(self):
        return hash(self.tuple_state)

    def __repr__(self):
        return repr(self.current_state)

    def display_board(self):
        # print(f"snake_size = {len(self.snake_segments)}")
        screen = np.zeros((grid_y_count, grid_x_count, 3), dtype=int)
        for segment in self.snake_segments:
            color = (165, 255, 81)
            if not self.snake_alive:
                color = (140, 140, 140)
            screen[segment['y'], segment['x']] = color
        screen[self.food_position['y'], self.food_position['x']] = (255, 76, 76)

        fig, ax = plt.subplots()
        im = ax.imshow(screen)

        for i, segment in enumerate(self.snake_segments):
            text = ax.text(segment['x'], segment['y'], str(i), ha='center', va='center', color="b")

        # Render the figure to a binary buffer
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("snake", img)
        cv2.waitKey(1)
        plt.close()

if __name__ == "__main__":

    s = SnakeGameState()
    s.display_board()

    for i in range(100):
        op = random.choice(s.get_operations_options())
        print(op)
        s = s.apply_operation(op)
        s.display_board()
    # win:
    s.apply_operation("left").apply_operation("left").apply_operation("left")

    # stuck
    s.apply_operation("up").apply_operation("left").apply_operation("down")


