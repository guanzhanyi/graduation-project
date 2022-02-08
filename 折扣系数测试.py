'''
Author: guanzhanyi
Date: 2022-01-25 23:33:43
LastEditors: guanzhanyi
LastEditTime: 2022-02-08 16:14:26
FilePath: \graduation-project\折扣系数测试.py
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
sys.stdout = open('_折扣系数测试.txt', 'a',encoding='utf8')

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
    discount_factors = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 1]
    slide_window_side = setting.slide_window_side
    for discount_factor in discount_factors:
        discount_factor_ds = discount_factor_d = discount_factor

        discount_accumulation = np.array([0.0] * Time)
        discount_slide_window_accumulation = np.array([0.0] * Time)

        ds_accuracy_acc = 0
        dis_accuracy_acc = 0
        
        for i in range(trial):
            print(discount_factor,trial,flush=True)
            discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, succ_count, fail_count, best_arms)
            print("discount slide finished",datetime.now().strftime("%H:%M:%S"), discount_factor, choices, flush=True)
            discount, dis_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, discount_factor_d, Time, succ_count, fail_count, best_arms)
            print("discount finished",datetime.now().strftime("%H:%M:%S"), discount_factor, choices, flush=True)
            
            discount_slide_window_accumulation = discount_slide_window_accumulation + discount_slide_window
            discount_accumulation = discount_accumulation + discount

            ds_accuracy_acc+=ds_accuracy
            dis_accuracy_acc+=dis_accuracy

    # 画图
        print(" discount_slide:",ds_accuracy_acc/trial, " discount:",dis_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), discount_accumulation / trial,
                label='折扣系数' + ' 折扣系数:' + str(discount_factor_d))
        plt.plot(np.array(range(1, Time + 1)), discount_slide_window_accumulation / trial,
                label='折扣滑动' + ' 折扣系数:' + str(discount_factor_ds))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真结果' + ' 仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.savefig('折扣',dpi=300)
