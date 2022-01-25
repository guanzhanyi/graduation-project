import numpy as np


def single_feedback(time_slot, pred_prob, trans_prob, rates, bests, circle, change_time, best_arms):
    arm_num = len(rates)
    pred_right_count = 0
    choices = np.array([0]*arm_num)
    regret = np.array([0.0] * time_slot)
    S_n = np.array([0] * arm_num)
    F_n = np.array([0] * arm_num)
    y_n = np.array([0.0] * arm_num)
    index = 0
    k = len(trans_prob)
    for t in range(time_slot):
        t_remainder = t % circle
        for r in range(arm_num):
            y_n[r] = np.random.beta(S_n[r] + 1, F_n[r] + 1)
        if t_remainder % change_time == 0:
            index = (index + 1) % k
        y_n = y_n * y_n
        choice = np.argmax(y_n * rates)
        choices[choice] += 1
        X = np.random.choice([1, 0], size=1, p=[
                             pred_prob[choice], 1 - pred_prob[choice]])[0]
        Y = np.random.choice([1, 0], size=1, p=[
                             trans_prob[index][choice], 1 - trans_prob[index][choice]])[0]
        if X * Y == 1:
            S_n[choice] += 1
        else:
            F_n[choice] += 1
        if t == 0:
            regret[t] = bests[index] - X * Y * rates[choice]
        else:
            regret[t] = regret[t - 1] + bests[index] - X * Y * rates[choice]
        if best_arms[index] == choice:
            pred_right_count += 1
    return regret, pred_right_count/time_slot, choices
