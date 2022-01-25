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
sys.stdout = open('总测试.txt', 'a',encoding='utf8')

print(datetime.now())

# 总测试
if __name__ == '__main__':
    # 生成50个trans_prob
    # datetime slot
    Time = 10000
    # 实验次数
    trial = 10
    # 大周期
    big_circle = 600
    # 置零偏差
    piancha = 0
    # 折扣系数
    discount_factor_ds = 1
    discount_factor_d = 1
    
    add = 2
    sub = 2

    rates = np.array([2, 3, 5, 6, 9])
    # 两个
    trans_prob = [[0.9, 0.8, 0.65, 0.63, 0.1], [0.99, 0.85, 0.7, 0.15, 0.01]]
    # 滑动窗口大小
    small_circle = big_circle // len(trans_prob)
    slide_window_side = 250
    print("trans_prob:\n", trans_prob)
    # 预测概率分布
    pred_prob = np.array([0.1, 0.3, 0.6, 0.7, 0.9])
    # 用于求回归的最佳情况
    # 用于画图，累加每一次实验的结果
    double_regret_accumulation = np.array([0.0] * Time)
    single_regret_accumulation = np.array([0.0] * Time)
    set_zero_accumulation = np.array([0.0] * Time)
    slide_window_accumulation = np.array([0.0] * Time)
    discount_accumulation = np.array([0.0] * Time)
    discount_slide_window_accumulation = np.array([0.0] * Time)

    sz_accuracy_acc = 0
    dou_accuracy_acc= 0
    sin_accuracy_acc= 0
    ds_accuracy_acc = 0
    sli_accuracy_acc = 0 
    dis_accuracy_acc = 0

    bests = []
    best_arms = []
    for i in range(len(trans_prob)):
        choice = np.argmax(rates * pred_prob * trans_prob[i])
        best = rates[choice] * pred_prob[choice] * trans_prob[i][choice]
        bests.append(best)
        best_arms.append(choice)
    bests = np.array(bests)
    best_arms = np.array(best_arms)

    for i in range(trial):
        print("trial: ", i, flush=True)
        set_zero_regret,  sz_accuracy, choices = set_zero.d_set_zero(Time, pred_prob, trans_prob,rates, bests, big_circle, small_circle, piancha, best_arms)
        print("set zero finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True,)
        double_regret , dou_accuracy, choices= double_feedback.double_feedback(Time, pred_prob, trans_prob, rates, bests,big_circle, small_circle,best_arms)
        print("double finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        single_regret, sin_accuracy, choices = single_feedback.single_feedback(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, best_arms)
        print("single finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, add, sub, best_arms)
        print("discount slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, add, sub, best_arms)
        print("slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        discount, dis_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, discount_factor_d, Time, add, sub, best_arms)
        print("discount finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        
        double_regret_accumulation += double_regret
        set_zero_accumulation += set_zero_regret
        single_regret_accumulation += single_regret
        discount_slide_window_accumulation += discount_slide_window
        discount_accumulation += discount
        slide_window_accumulation += slide_window

        sz_accuracy_acc+=sz_accuracy
        dou_accuracy_acc+=dou_accuracy
        sin_accuracy_acc+=sin_accuracy
        ds_accuracy_acc+=ds_accuracy
        sli_accuracy_acc+=sli_accuracy
        dis_accuracy_acc+=dis_accuracy

    # 画图
    print("set_zero:",sz_accuracy_acc/trial," double:", dou_accuracy_acc/trial, " single:",sin_accuracy_acc/trial, " discount_slide:",ds_accuracy_acc/trial, " slide:",sli_accuracy_acc/trial, " discount:",dis_accuracy_acc/trial, flush=True)

    plt.plot(np.array(range(1, Time + 1)), double_regret_accumulation / trial, label='双反馈')
    plt.plot(np.array(range(1, Time + 1)), set_zero_accumulation / trial,
             label='置零双反馈')
    plt.plot(np.array(range(1, Time + 1)), single_regret_accumulation / trial, label='单反馈')
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
