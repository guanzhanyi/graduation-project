import set_zero
import single_feedback
import double_feedback
import ss_window
import setting
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys
sys.stdout = open('_总测试.txt', 'a',encoding='utf8')

print(datetime.now())
np.set_printoptions(threshold=100000000000)

# 总测试
if __name__ == '__main__':
    # 生成50个trans_prob
    # datetime slot
    Time = setting.Time
    # 实验次数
    trial = setting.trial
    # 大周期
    big_circle = setting.big_circle
    # 置零偏差
    piancha = setting.piancha
    # 折扣系数
    discount_factor_ds  = setting.discount_factor_ds 
    discount_factor_d  = setting.discount_factor_d
    
    succ_count = setting.succ_count
    fail_count = setting.fail_count

    rates = setting.rates
    trans_prob =  setting.trans_prob
    # 滑动窗口大小
    small_circle = big_circle // len(trans_prob)
    slide_window_side = setting.slide_window_side
    print("trans_prob:\n", trans_prob)
    # 预测概率分布
    pred_prob = setting.pred_prob
    # 用于求回归的最佳情况
    # 用于画图，累加每一次实验的结果
    double_regret_accumulation = np.array([0.0] * Time)
    single_regret_accumulation = np.array([0.0] * Time)
    set_zero_accumulation = np.array([0.0] * Time)
    slide_window_accumulation = np.array([0.0] * Time)
    discount_accumulation = np.array([0.0] * Time)
    discount_slide_window_accumulation = np.array([0.0] * Time)
    succ_count_discount_slide_window_accumulation = np.array([0.0] * Time)

    sz_accuracy_acc = 0
    dou_accuracy_acc= 0
    sin_accuracy_acc= 0
    ds_accuracy_acc = 0
    sli_accuracy_acc = 0 
    dis_accuracy_acc = 0
    succ_count_accuracy_acc = 0
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
        #print("trial: ", i, flush=True)
        set_zero_regret,  sz_accuracy, choices = set_zero.d_set_zero(Time, pred_prob, trans_prob,rates, bests, big_circle, small_circle, piancha, best_arms)
        #print("set zero finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        double_regret , dou_accuracy, choices= double_feedback.double_feedback(Time, pred_prob, trans_prob, rates, bests,big_circle, small_circle,best_arms)
        #print("double finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        single_regret, sin_accuracy, choices = single_feedback.single_feedback(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, best_arms)
        #print("single finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, 1, 1, best_arms)
        #print("discount slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, 1, 1, best_arms)
        #print("slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        discount, dis_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, discount_factor_d, Time, 1, 1, best_arms)
        #print("discount finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        succ_count_regret, succ_count_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, succ_count, fail_count, best_arms)
        #print("succ_count finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)

        double_regret_accumulation += double_regret
        set_zero_accumulation += set_zero_regret
        single_regret_accumulation += single_regret
        discount_slide_window_accumulation += discount_slide_window
        discount_accumulation += discount
        slide_window_accumulation += slide_window
        succ_count_discount_slide_window_accumulation += succ_count_regret

        sz_accuracy_acc += sz_accuracy
        dou_accuracy_acc += dou_accuracy
        sin_accuracy_acc += sin_accuracy
        ds_accuracy_acc += ds_accuracy
        sli_accuracy_acc += sli_accuracy
        dis_accuracy_acc += dis_accuracy
        succ_count_accuracy_acc += succ_count_accuracy

    # 画图
    print("set_zero:",sz_accuracy_acc/trial," double:", dou_accuracy_acc/trial, " single:",sin_accuracy_acc/trial, " discount_slide:",ds_accuracy_acc/trial, " slide:",sli_accuracy_acc/trial, " discount:",dis_accuracy_acc/trial, " succ_count:",succ_count_accuracy_acc/trial,flush=True)
    print("set_zero:",set_zero_accumulation/trial," double:", double_regret_accumulation/trial, " single:",single_regret_accumulation/trial, " discount_slide:",discount_slide_window_accumulation/trial, " slide:",slide_window_accumulation/trial, " discount:",discount_accumulation/trial, " succ_count:",succ_count_discount_slide_window_accumulation/trial,flush=True)

    plt.plot(np.array(range(1, Time + 1)), double_regret_accumulation / trial, label='双反馈')
    plt.plot(np.array(range(1, Time + 1)), set_zero_accumulation / trial,
             label='置零')
    plt.plot(np.array(range(1, Time + 1)), single_regret_accumulation / trial, label='单反馈')
    plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial,
             label='滑动窗口'+' 容量:' + str(slide_window_side))
    plt.plot(np.array(range(1, Time + 1)), discount_accumulation / trial,
             label='折扣系数' + ' 折扣系数:' + str(discount_factor_d))
    plt.plot(np.array(range(1, Time + 1)), discount_slide_window_accumulation / trial,
             label='折扣系数滑动窗口' + ' 容量:' + str(slide_window_side)+ ' 折扣系数:' + str(discount_factor_ds))
    plt.plot(np.array(range(1, Time + 1)), succ_count_discount_slide_window_accumulation / trial,
             label='增加折扣系数滑动窗口' + ' 容量:' + str(slide_window_side)+ ' 折扣系数:' + str(discount_factor_ds)+' 增加:'+str(succ_count)+' 减少:'+str(fail_count))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    #plt.show()
    plt.savefig('总',dpi=300)
            