import pandas as pd
import matplotlib.pyplot as plt

directory = "epsilon_greedy_results"  # UCT_results  or  epsilon_greedy_results
df = pd.read_csv(f"{directory}/res_walking_game_long.csv")
functions = df["function"].unique()


thinking_time = df[(df["level"] == 0) & (df["function"] == "max_reward")]["thinking_time"]
new_df = df[(df["level"] == 0)].copy()
for level in df.level.unique()[1:]:
    new_df["win_count"] += df[df["level"] == level]["win_count"].to_numpy()
    new_df["num_moves_mean"] += df[df["level"] == level]["num_moves_mean"].to_numpy()
new_df["win_count"] = new_df["win_count"] / len(df.level.unique()) / 10
new_df["num_moves_mean"] = new_df["num_moves_mean"] / len(df.level.unique())

for func in functions:
    wins_prct = new_df[new_df["function"] == func]["win_count"]
    plt.plot(thinking_time, wins_prct * 100)
plt.legend(functions)
plt.title("Walking game %wins w.r.t budget & function")
plt.ylabel("% wins")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/walking_game1.png", bbox_inches='tight')
plt.show()

for func in functions:
    num_moves_mean = new_df[new_df["function"] == func]["num_moves_mean"]
    plt.plot(thinking_time, num_moves_mean)
plt.legend(functions)
plt.title("Walking game average number of moves w.r.t budget & function")
plt.ylabel("Avg moves")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/walking_game2.png", bbox_inches='tight')
plt.show()


for func in functions:
    num_moves_mean = new_df[new_df["function"] == func]["num_moves_mean"]
    plt.plot(thinking_time, num_moves_mean)
plt.legend(functions)
plt.title("Walking game average number of moves w.r.t budget & function")
plt.ylabel("Avg moves")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/walking_game3.png", bbox_inches='tight')
plt.show()
