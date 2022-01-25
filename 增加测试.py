import set_zero
import single_feedback
import double_feedback
import ss_window

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys
sys.stdout = open('增加测试.txt', 'a',encoding='utf8')

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
    
    adds = [1,2,3,4,5,6,7,8,9,10]
    sub = 2

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
    discount_factor = 1
    slide_window_side = 250

    for add in adds:
        slide_window_accumulation = np.array([0.0] * Time)
        sli_accuracy_acc = 0
        for i in range(trial):
            slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, add, sub, best_arms)
            print(add,datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            slide_window_accumulation = slide_window_accumulation + slide_window
            sli_accuracy_acc+=sli_accuracy

    # 画图
        print(" slide:",sli_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial,
                label='滑动窗口双反馈'+' add:' + str(add))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真结果' + ' 仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()
