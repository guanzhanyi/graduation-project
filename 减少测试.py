'''
Author: guanzhanyi
Date: 2022-01-25 23:34:05
LastEditors: guanzhanyi
LastEditTime: 2022-02-08 12:39:46
FilePath: \graduation-project\减少测试.py
Description: 

Copyright (c) 2022 by guanzhanyi/xxx, All Rights Reserved. 
'''
import ss_window
import setting
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys
sys.stdout = open('_减少测试.txt', 'a',encoding='utf8')

print(datetime.now())

if __name__ == '__main__':
    # 生成50个trans_prob
    # datetime slot
    Time = setting.Time
    # 实验次数
    trial = setting.trial
    # 大周期
    big_circle = setting.big_circle
    # 折扣系数
    
    succ_count = setting.succ_count
    fail_counts = [1,2,3,4,5,6,7,8,9,10]

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
    discount_factor = 1
    slide_window_side = setting.slide_window_side
    

    for fail_count in fail_counts:
        slide_window_accumulation = np.array([0.0] * Time)
        sli_accuracy_acc = 0
        for i in range(trial):
            slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, succ_count, fail_count, best_arms)
            print(fail_count,datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            slide_window_accumulation = slide_window_accumulation + slide_window
            sli_accuracy_acc+=sli_accuracy

    # 画图
        print(" slide:",sli_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial,
                label='滑动窗口双反馈'+' fail_count:' + str(fail_count))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真结果' + ' 仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    # plt.show()
    plt.savefig('减少',dpi=300)