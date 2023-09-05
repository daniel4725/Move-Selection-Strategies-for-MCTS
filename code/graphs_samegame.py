import pandas as pd
import matplotlib.pyplot as plt

directory = "epsilon_greedy_results"  # UCT_results  or  epsilon_greedy_results
df = pd.read_csv(f"{directory}/res_same_game.csv")
functions = df["function"].unique()

thinking_time = df[df["function"] == "max_reward"]["thinking_time"]
for func in functions:
    scores = df[df["function"] == func]["scores_mean"]
    plt.plot(thinking_time, scores)
plt.legend(functions)
plt.title("Same Game scores w.r.t budget & function")
plt.ylabel("avg scores")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/same_game1.png", bbox_inches='tight')
plt.show()

thinking_time = df[df["function"] == "max_reward"]["thinking_time"]
for func in functions:
    scores = df[df["function"] == func]["scores_mean"]
    plt.plot(thinking_time[1:], scores[1:])
plt.legend(functions)
plt.title("Same Game scores w.r.t budget & function (zoomed)")
plt.ylabel("avg scores")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/same_game2.png", bbox_inches='tight')
plt.show()

thinking_time = df[df["function"] == "max_reward"]["thinking_time"]
for func in functions:
    scores = df[df["function"] == func]["scores_mean"]
    plt.plot(thinking_time[2:], scores[2:])
plt.legend(functions)
plt.title("Same Game scores w.r.t budget & function (zoomed)")
plt.ylabel("avg scores")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/same_game3.png", bbox_inches='tight')
plt.show()
