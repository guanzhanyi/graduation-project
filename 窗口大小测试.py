'''
Author: guanzhanyi
Date: 2022-01-25 23:33:30
LastEditors: guanzhanyi
LastEditTime: 2022-02-08 16:14:39
FilePath: \graduation-project\窗口大小测试.py
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
sys.stdout = open('_窗口大小测试.txt', 'a',encoding='utf8')

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
    discount_factor_ds  = setting.discount_factor_ds 
    discount_factor_d  = setting.discount_factor_d
    
    succ_count = setting.succ_count
    fail_count = setting.fail_count

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
    slide_window_sides = [100,250,350,500,800,1000,2000]
    for slide_window_side in slide_window_sides:
        print(slide_window_side)
        slide_window_accumulation = np.array([0.0] * Time)
        discount_accumulation = np.array([0.0] * Time)
        discount_slide_window_accumulation = np.array([0.0] * Time)

        ds_accuracy_acc = 0
        sli_accuracy_acc = 0
        dis_accuracy_acc = 0
        for i in range(trial):
            print(slide_window_side,trial,flush=True)
            discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, succ_count, fail_count, best_arms)
            print("discount slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            discount_slide_window_accumulation = discount_slide_window_accumulation + discount_slide_window

            ds_accuracy_acc+=ds_accuracy

    # 画图
        print(" discount_slide:",ds_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), discount_slide_window_accumulation / trial,
                label=' 容量:' + str(slide_window_side))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.savefig('窗口',dpi=300)
