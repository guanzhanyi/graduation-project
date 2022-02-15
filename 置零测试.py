'''
Author: guanzhanyi
Date: 2022-01-25 23:32:05
LastEditors: guanzhanyi
LastEditTime: 2022-02-15 18:29:17
FilePath: \graduation-project\置零测试.py
Description: 

Copyright (c) 2022 by guanzhanyi/xxx, All Rights Reserved. 
'''
import set_zero
import setting
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys
sys.stdout = open('_置零测试.txt', 'a',encoding='utf8')
np.set_printoptions(threshold=100000000000)

print(datetime.now())

if __name__ == '__main__':
    Time = setting.Time
    # 实验次数
    trial = setting.trial
    # 大周期
    big_circle = setting.big_circle
    # 置零偏差
    pianchas = [0.1, 0.3, 0.5, 0.7, 0.9, 0]

    rates = setting.rates
    # 两个
    trans_prob = setting.trans_prob
    # 滑动窗口大小
    small_circle = big_circle // len(trans_prob)
    print("trans_prob:\n", trans_prob)
    # 预测概率分布
    pred_prob = setting.pred_prob

    bests = []
    best_arms = []
    for i in range(len(trans_prob)):
        choice = np.argmax(rates * pred_prob * trans_prob[i])
        best = rates[choice] * pred_prob[choice] * trans_prob[i][choice]
        bests.append(best)
        best_arms.append(choice)
    bests = np.array(bests)
    best_arms = np.array(best_arms)
    for piancha in pianchas:
        set_zero_accumulation = np.array([0.0] * Time)
        sz_accuracy_acc = 0
        print("偏差:",piancha)
        for i in range(trial):
            set_zero_regret,  sz_accuracy, choices = set_zero.d_set_zero(Time, pred_prob, trans_prob,rates, bests, big_circle, small_circle, piancha, best_arms)
            set_zero_accumulation += set_zero_regret
            sz_accuracy_acc+=sz_accuracy
            # print(choices)
        
        plt.plot(np.array(range(1, Time + 1)), set_zero_accumulation / trial,label='偏差:'+str(piancha))
        print(str(piancha), set_zero_accumulation / trial)
        print(str(piancha), sz_accuracy_acc / trial)
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle))
    plt.legend()
    plt.savefig('置零',dpi=300)
