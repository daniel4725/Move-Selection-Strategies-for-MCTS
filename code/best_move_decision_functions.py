import numpy as np
from utils import random_argmax_choice



def max_reward(children_list):
    best_child_idx = random_argmax_choice([np.mean(child.reward) for child in children_list])
    return children_list[best_child_idx]


def max_reward_optimistic(children_list):
    best_child_idx = random_argmax_choice([np.mean(child.reward) + np.std(child.reward) for child in children_list])
    return children_list[best_child_idx]


def max_reward_pessimistic(children_list):
    best_child_idx = random_argmax_choice([np.mean(child.reward) - np.std(child.reward) for child in children_list])
    return children_list[best_child_idx]


def max_reward_median(children_list):
    best_child_idx = random_argmax_choice([np.median(child.reward) for child in children_list])
    return children_list[best_child_idx]


def max_visits(children_list):
    best_child_idx = random_argmax_choice([child.visits for child in children_list])
    return children_list[best_child_idx]


def max_visits_50_reward_50(children_list):
    a = 0.5
    b = 0.5
    return max_visits_reward(a, b, children_list)


def max_visits_75_reward_25(children_list):
    a = 0.75
    b = 0.25
    return max_visits_reward(a, b, children_list)


def max_visits_25_reward_75(children_list):
    a = 0.25
    b = 0.75
    return max_visits_reward(a, b, children_list)


def max_visits_reward(a, b, children_list):
    if len(children_list) < 2:
        return children_list[0]
    visits = np.array([child.visits for child in children_list])
    rewards = np.array([np.mean(child.reward) for child in children_list])

    visits = np.nan_to_num((visits - visits.min()) / (visits.max() - visits.min()))
    rewards = np.nan_to_num((rewards - rewards.min()) / (rewards.max() - rewards.min()))

    best_child_idx = random_argmax_choice([(a * v) + (b * r) for r, v in zip(rewards, visits)])
    return children_list[best_child_idx]


