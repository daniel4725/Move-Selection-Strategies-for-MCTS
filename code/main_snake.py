from MCTS import MonteCarloTreeSearch
from snake_game import SnakeGameState
import time
import pandas as pd
from best_move_decision_functions import *
import cv2
from tqdm import tqdm
import os

# num_games = 3
# thinking_times = [0.1, 0.5]
# time_limit4game = 10
# functions = ["max_reward", "max_visits"]

num_games = 50
thinking_times = [0.1, 0.5, 1, 2]
time_limit4game = 180
functions = ["max_reward", "max_visits", "max_visits_50_reward_50", "max_visits_25_reward_75",
             "max_visits_75_reward_25", "max_reward_optimistic", "max_reward_pessimistic", "max_reward_median"]
select_child2expand_method = "UCT"   # "UCT" or "epsilon_greedy"
os.makedirs(f"{select_child2expand_method}_results", exist_ok=True)

results_dict = {}
idx = 0
for func in tqdm(functions, "choose_best_move_func"):
    for thinking_time in tqdm(thinking_times, "thinking_time"):

        times = []
        snake_len = []
        num_moves = []
        died_count = 0
        for game in tqdm(range(num_games), "games"):
            start = SnakeGameState()
            mcts = MonteCarloTreeSearch(root_state=start, exploration_constant=1, display_game=True,
                                        choose_best_move_func=locals()[func], iterations_per_move=1000,
                                        time_per_move=thinking_time, time_limit4game=time_limit4game,
                                        select_child2expand_method=select_child2expand_method)
            start_time = time.time()
            final_state, moves = mcts.search()
            times.append(time.time() - start_time)
            snake_len.append(len(final_state.game_state.snake_segments))
            num_moves.append(moves)
            if not final_state.game_state.snake_alive:
                died_count += 1

            print(f"time: {times[-1]}")
            print(f"snake_len: {snake_len[-1]}")
            print(f"num_moves: {num_moves[-1]}")

        results_dict[idx] = {}
        results_dict[idx]["function"] = func
        results_dict[idx]["thinking_time"] = thinking_time
        results_dict[idx]["died_count"] = died_count
        results_dict[idx]["lived_count"] = num_games - died_count
        results_dict[idx]["snake_len_mean"] = np.mean(snake_len)
        results_dict[idx]["snake_len_std"] = np.std(snake_len)
        results_dict[idx]["num_moves_mean"] = np.mean(num_moves)
        results_dict[idx]["num_moves_std"] = np.std(num_moves)
        results_dict[idx]["times_mean"] = np.mean(times)
        results_dict[idx]["times_std"] = np.std(times)
        idx += 1

        df = pd.DataFrame(results_dict).T
        df.to_csv(f"{select_child2expand_method}_results/res_snake.csv", index=False)



