import numpy as np

'''
bests: t时隙最好的吞吐量
big_cycle: 一个周期
small_cycle: 一个周期内一组传输概率持续的时间
piancha: 置零的时间与small_cycle的偏差，0时表示small_cycle发生改变时进行切换
best_arms: t时隙最好的arm
-----------
regret: 遗憾
accuracy: 准确率
choices: 各个手臂被选中的次数
'''


def d_set_zero(time_slot, pred_prob, trans_prob, rates, bests, big_cycle, small_circle, piancha, best_arms):

    arm_num = len(rates)
    pred_right_count = 0
    choices = np.array([0]*arm_num)
    regret = np.array([0.0] * time_slot)
    S_n_1 = np.array([0] * arm_num)
    F_n_1 = np.array([0] * arm_num)
    S_n_2 = np.array([0] * arm_num)
    F_n_2 = np.array([0] * arm_num)
    a_n = np.array([0.0] * arm_num)
    b_n = np.array([0.0] * arm_num)
    index = 0
    k = big_cycle // small_circle
    for t in range(time_slot):
        t_remainder = t % big_cycle
        # 周期内的传输概率发生变化
        if t_remainder % (small_circle + piancha) == 0:
            S_n_2 = np.array([0] * arm_num)
            F_n_2 = np.array([0] * arm_num)
        if t_remainder % small_circle == 0:
            index = (index + 1) % k
        for r in range(arm_num):
            a_n[r] = np.random.beta(S_n_1[r] + 1, F_n_1[r] + 1)
            b_n[r] = np.random.beta(S_n_2[r] + 1, F_n_2[r] + 1)
        cho = np.argmax(a_n * b_n * rates)
        choices[cho] += 1
        X = np.random.choice([1, 0], size=1, p=[
                             pred_prob[cho], 1 - pred_prob[cho]])[0]
        Y = np.random.choice([1, 0], size=1, p=[
                             trans_prob[index][cho], 1 - trans_prob[index][cho]])[0]

        if X == 1:
            S_n_1[cho] += 1
        else:
            F_n_1[cho] += 1
        if Y == 1:
            S_n_2[cho] += 1
        else:
            F_n_2[cho] += 1
        if t == 0:
            regret[t] = bests[index] - X * Y * rates[cho]
        else:
            regret[t] = regret[t - 1] + bests[index] - X * Y * rates[cho]
        if best_arms[index] == cho:
            pred_right_count += 1
    accuracy = pred_right_count/time_slot
    return regret, accuracy, choices
