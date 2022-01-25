import ss_window

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys
sys.stdout = open('折扣系数测试.txt', 'a',encoding='utf8')

print(datetime.now())

if __name__ == '__main__':
    # 生成50个trans_prob
    # datetime slot
    Time = 10000
    # 实验次数
    trial = 3
    # 大周期
    big_circle = 3600
    # 折扣系数
    
    add = 1
    sub = 1

    rates = np.array([2, 3, 5, 6, 9])
    # 两个
    trans_prob = [[0.9, 0.8, 0.65, 0.63, 0.1], [0.99, 0.85, 0.7, 0.15, 0.01]]
    # 滑动窗口大小
    small_circle = big_circle // len(trans_prob)
    print("trans_prob:\n", trans_prob)
    # 预测概率分布
    pred_prob = np.array([0.1, 0.3, 0.6, 0.7, 0.9])
    bests = []
    best_arms = []
    for i in range(len(trans_prob)):
        choice = np.argmax(rates * pred_prob * trans_prob[i])
        best = rates[choice] * pred_prob[choice] * trans_prob[i][choice]
        bests.append(best)
        best_arms.append(choice)
    bests = np.array(bests)
    best_arms = np.array(best_arms)
    discount_factors = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95]
    slide_window_side = 300
    for discount_factor in discount_factors:
        discount_factor_ds = discount_factor_d = discount_factor

        slide_window_accumulation = np.array([0.0] * Time)
        discount_accumulation = np.array([0.0] * Time)
        discount_slide_window_accumulation = np.array([0.0] * Time)

        ds_accuracy_acc = 0
        sli_accuracy_acc = 0
        dis_accuracy_acc = 0
        for i in range(trial):
            discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, add, sub, best_arms)
            print("discount slide finished",datetime.now().strftime("%H:%M:%S"), discount_factor, choices, flush=True)
            slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, add, sub, best_arms)
            print("slide finished",datetime.now().strftime("%H:%M:%S"), discount_factor, choices, flush=True)
            discount, dis_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, discount_factor_d, Time, add, sub, best_arms)
            print("discount finished",datetime.now().strftime("%H:%M:%S"), discount_factor, choices, flush=True)
            
            discount_slide_window_accumulation = discount_slide_window_accumulation + discount_slide_window
            discount_accumulation = discount_accumulation + discount
            slide_window_accumulation = slide_window_accumulation + slide_window

            ds_accuracy_acc+=ds_accuracy
            sli_accuracy_acc+=sli_accuracy
            dis_accuracy_acc+=dis_accuracy

    # 画图
        print(" discount_slide:",ds_accuracy_acc/trial, " slide:",sli_accuracy_acc/trial, " discount:",dis_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial,
                label='滑动窗口双反馈'+' 窗口容量:' + str(slide_window_side))
        plt.plot(np.array(range(1, Time + 1)), discount_accumulation / trial,
                label='折扣系数双反馈' + ' 折扣系数:' + str(discount_factor_d))
        plt.plot(np.array(range(1, Time + 1)), discount_slide_window_accumulation / trial,
                label='折扣系数滑动窗口双反馈' + ' 容量:' + str(slide_window_side)+ ' 折扣系数:' + str(discount_factor_ds))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真结果' + ' 仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()
