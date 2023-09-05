from MCTS import MonteCarloTreeSearch
from same_game import SameGameState, get_samegame_board
import time
import pandas as pd
from best_move_decision_functions import *
import cv2
from tqdm import tqdm
import os

# num_games = 3
# thinking_times = [0.1, 0.5]
# functions = ["max_reward", "max_visits"]

num_games = 100
thinking_times = [0.1, 0.5, 1, 2, 4]
functions = ["max_reward", "max_visits", "max_visits_50_reward_50", "max_visits_25_reward_75",
             "max_visits_75_reward_25", "max_reward_optimistic", "max_reward_pessimistic", "max_reward_median"]
select_child2expand_method = "epsilon_greedy"   # "UCT" or "epsilon_greedy"
os.makedirs(f"{select_child2expand_method}_results", exist_ok=True)

results_dict = {}
idx = 0
for func in tqdm(functions, "choose_best_move_func"):
    for thinking_time in tqdm(thinking_times, "thinking_time"):

        times = []
        scores = []
        num_moves = []
        for game in tqdm(range(num_games), "games"):
            start = SameGameState(get_samegame_board(game))
            mcts = MonteCarloTreeSearch(root_state=start, exploration_constant=1, display_game=True,
                                        choose_best_move_func=locals()[func], iterations_per_move=1000,
                                        time_per_move=thinking_time, select_child2expand_method=select_child2expand_method)

            start_time = time.time()
            final_state, moves = mcts.search()
            times.append(time.time() - start_time)
            scores.append(final_state.game_state.scores)
            num_moves.append(moves)

            print(f"time: {times[-1]}")
            print(f"scores: {final_state.game_state.scores}")
            print(f"num_moves: {moves}")

        results_dict[idx] = {}
        results_dict[idx]["function"] = func
        results_dict[idx]["thinking_time"] = thinking_time
        results_dict[idx]["scores_mean"] = np.mean(scores)
        results_dict[idx]["scores_std"] = np.std(scores)
        results_dict[idx]["num_moves_mean"] = np.mean(num_moves)
        results_dict[idx]["num_moves_std"] = np.std(num_moves)
        results_dict[idx]["times_mean"] = np.mean(times)
        results_dict[idx]["times_std"] = np.std(times)
        idx += 1

        df = pd.DataFrame(results_dict).T
        df.to_csv(f"{select_child2expand_method}_results/res_same_game.csv", index=False)



