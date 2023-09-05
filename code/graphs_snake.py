import pandas as pd
import matplotlib.pyplot as plt


# avg game time (not died count) add std to the game time?,
# seconds per eat, length,


directory = "epsilon_greedy_results"  # UCT_results  or  epsilon_greedy_results
df = pd.read_csv(f"{directory}/res_snake.csv")
functions = df["function"].unique()


thinking_time = df[df["function"] == "max_reward"]["thinking_time"]
for func in functions:
    snake_len_mean = df[df["function"] == func]["snake_len_mean"]
    plt.plot(thinking_time, snake_len_mean)
plt.legend(functions)
plt.title("Snake length w.r.t budget & function")
plt.ylabel("avg length")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/snake1.png", bbox_inches='tight')
plt.show()

thinking_time = df[df["function"] == "max_reward"]["thinking_time"]
for func in functions:
    times_mean = df[df["function"] == func]["times_mean"]
    plt.plot(thinking_time, times_mean)
plt.legend(functions)
plt.title("Snake survival time w.r.t budget & function")
plt.ylabel("avg survival time (sec)")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/snake2.png", bbox_inches='tight')
plt.show()


thinking_time = df[df["function"] == "max_reward"]["thinking_time"]
for func in functions:
    if func in ["max_reward_pessimistic", "max_reward_median"]:
        continue
    times_mean = df[df["function"] == func]["times_mean"]
    num_moves_mean = df[df["function"] == func]["num_moves_mean"]
    snake_len_mean = df[df["function"] == func]["snake_len_mean"]

    plt.plot(thinking_time, num_moves_mean / snake_len_mean)
plt.legend(functions)
plt.title("Snake moves per eat w.r.t budget & function")
plt.ylabel("avg moves per eat")
plt.xlabel("time budget (sec)")
plt.savefig(f"{directory}/snake3.png", bbox_inches='tight')
plt.show()


