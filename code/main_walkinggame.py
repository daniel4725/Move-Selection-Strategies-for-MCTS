from MCTS import MonteCarloTreeSearch
from walking_game import WalkingGameStateLongWalks, WalkingGameStateShortWalks
from walking_game_levels import walking_game_boards
import time
import pandas as pd
from best_move_decision_functions import *
import cv2
from tqdm import tqdm
import os
# num_games_per_level = 3
# levels = [len(walking_game_boards)-1]
# thinking_times = [0.2]
# walking_type = "long"   # "long"  "short"
# functions = ["max_visits_50_reward_50", "max_visits_75_reward_25", "max_visits_25_reward_75"]
# select_child2expand_method = "UCT"   # "UCT" or "epsilon_greedy"

num_games_per_level = 10
levels = list(range(len(walking_game_boards)))
thinking_times = [0.1, 0.5, 1, 2]
walking_type = "long"   # "long"  "short"
functions = ["max_reward", "max_visits", "max_visits_50_reward_50", "max_visits_25_reward_75",
             "max_visits_75_reward_25", "max_reward_optimistic", "max_reward_pessimistic", "max_reward_median"]
select_child2expand_method = "epsilon_greedy"   # "UCT" or "epsilon_greedy"
os.makedirs(f"{select_child2expand_method}_results", exist_ok=True)


if walking_type == "long":
    walking_game = WalkingGameStateLongWalks
elif walking_type == "short":
    walking_game = WalkingGameStateShortWalks
else:
    exit(0)
results_dict = {}
idx = 0
for level in tqdm(levels, "levels"):
    for thinking_time in tqdm(thinking_times, "thinking_time"):
        for func in tqdm(functions, "choose_best_move_func"):

            times = []
            scores = []
            num_moves = []
            # diff_from_minimal_moves = []
            died_count = 0
            for game in tqdm(range(num_games_per_level), "games"):
                start = walking_game(walking_game_boards[level])
                # best_win_state, minimal_moves = walking_game.compute_bfs_till_win(start)
                mcts = MonteCarloTreeSearch(root_state=start, exploration_constant=1, display_game=True,
                                            choose_best_move_func=locals()[func], iterations_per_move=10000,
                                            time_per_move=thinking_time, select_child2expand_method=select_child2expand_method)

                start_time = time.time()
                final_state, moves = mcts.search()
                times.append(time.time() - start_time)
                scores.append(final_state.game_state.get_result())
                num_moves.append(moves)
                if final_state.game_state.is_a_lose():
                    died_count += 1
                # else:
                #     diff_from_minimal_moves.append(moves - minimal_moves)
                #     print(f"diff_from_minimal_moves: {diff_from_minimal_moves[-1]}")

                print(f"time: {times[-1]}")
                print(f"scores: {scores[-1]}")
                print(f"num_moves: {moves}")

            results_dict[idx] = {}
            results_dict[idx]["function"] = func
            results_dict[idx]["thinking_time"] = thinking_time
            results_dict[idx]["level"] = level
            results_dict[idx]["died_count"] = died_count
            results_dict[idx]["win_count"] = num_games_per_level - died_count
            results_dict[idx]["scores_mean"] = np.mean(scores)
            results_dict[idx]["scores_std"] = np.std(scores)
            results_dict[idx]["num_moves_mean"] = np.mean(num_moves)
            results_dict[idx]["num_moves_std"] = np.std(num_moves)
            # results_dict[idx]["diff_from_minimal_moves_mean"] = np.mean(diff_from_minimal_moves)
            # results_dict[idx]["diff_from_minimal_moves_std"] = np.std(diff_from_minimal_moves)
            results_dict[idx]["times_mean"] = np.mean(times)
            results_dict[idx]["times_std"] = np.std(times)
            idx += 1

            df = pd.DataFrame(results_dict).T
            df.to_csv(f"{select_child2expand_method}_results/res_walking_game_{walking_type}.csv", index=False)



