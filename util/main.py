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
sys.stdout = open('log.txt', 'a',encoding='utf8')

print(datetime.now())

# 总测试
if __name__ == '__main_ _':
    # 生成50个trans_prob
    # datetime slot
    Time = setting.Time
    # 实验次数
    trial = 13
    # 大周期
    big_circle = 600
    # 置零偏差
    piancha = setting.piancha 
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
        discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, succ_count, fail_count, best_arms)
        print("discount slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, succ_count, fail_count, best_arms)
        print("slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        discount, dis_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, discount_factor_d, Time, succ_count, fail_count, best_arms)
        print("discount finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        
        double_regret_accumulation = double_regret_accumulation + double_regret
        set_zero_accumulation = set_zero_accumulation + set_zero_regret
        single_regret_accumulation = single_regret_accumulation + single_regret
        discount_slide_window_accumulation = discount_slide_window_accumulation + discount_slide_window
        discount_accumulation = discount_accumulation + discount
        slide_window_accumulation = slide_window_accumulation + slide_window

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
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()


# set zero 偏移测试
if __name__ == '__main_ _':
    Time = setting.Time
    # 实验次数
    trial = 10
    # 大周期
    big_circle = 600
    # 置零偏差
    pianchas = [0.1, 0.3, 0.5, 0.7, 0.9, 0]

    rates = setting.rates
    # 两个
    trans_prob = setting.trans_prob
    # 滑动窗口大小
    small_circle = big_circle // len(trans_prob)
    slide_window_side = setting.slide_window_side
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
            print("trial: ", i, flush=True)
            set_zero_regret,  sz_accuracy, choices = set_zero.d_set_zero(Time, pred_prob, trans_prob,rates, bests, big_circle, small_circle, piancha, best_arms)
            set_zero_accumulation += set_zero_regret
            sz_accuracy_acc+=sz_accuracy
            print(choices)
        plt.plot(np.array(range(1, Time + 1)), set_zero_accumulation / trial,label='偏差:'+str(piancha))

    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()
# double 与 single 比较
if __name__ == '__main_ _':
    Time = setting.Time
    # 实验次数
    trial = 50
    # 大周期
    rates = setting.rates
    big_circle = setting.big_circle
    # 两个
    trans_prob = setting.trans_prob
    # 滑动窗口大小
    small_circle = big_circle // len(trans_prob)
    print("trans_prob:\n", trans_prob)
    # 预测概率分布
    pred_prob = setting.pred_prob
    # 用于求回归的最佳情况
    # 用于画图，累加每一次实验的结果
    double_regret_accumulation = np.array([0.0] * Time)
    single_regret_accumulation = np.array([0.0] * Time)

    dou_accuracy_acc= 0
    sin_accuracy_acc= 0

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
        double_regret , dou_accuracy, choices= double_feedback.double_feedback(Time, pred_prob, trans_prob, rates, bests,big_circle, small_circle,best_arms)
        print("double finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        single_regret, sin_accuracy, choices = single_feedback.single_feedback(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, best_arms)
        print("single finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        
        double_regret_accumulation = double_regret_accumulation + double_regret
        single_regret_accumulation = single_regret_accumulation + single_regret

        dou_accuracy_acc+=dou_accuracy
        sin_accuracy_acc+=sin_accuracy

    # 画图
    print(" double:", dou_accuracy_acc/trial, " single:",sin_accuracy_acc/trial,flush=True)

    plt.plot(np.array(range(1, Time + 1)), double_regret_accumulation / trial, label='双反馈')
    plt.plot(np.array(range(1, Time + 1)), single_regret_accumulation / trial, label='单反馈')

    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()

# double 与 slide_window 比较
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
    slide_window_side = setting.slide_window_side
    print("trans_prob:\n", trans_prob)
    # 预测概率分布
    pred_prob = setting.pred_prob
    # 用于求回归的最佳情况
    # 用于画图，累加每一次实验的结果
    double_regret_accumulation = np.array([0.0] * Time)
    slide_window_accumulation = np.array([0.0] * Time)
    discount_accumulation = np.array([0.0] * Time)
    discount_slide_window_accumulation = np.array([0.0] * Time)

    dou_accuracy_acc= 0
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
        double_regret , dou_accuracy, choices = double_feedback.double_feedback(Time, pred_prob, trans_prob, rates, bests,big_circle, small_circle,best_arms)
        print("double finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, succ_count, fail_count, best_arms)
        print("discount slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, succ_count, fail_count, best_arms)
        print("slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        discount, dis_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, discount_factor_d, Time, succ_count, fail_count, best_arms)
        print("discount finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        
        double_regret_accumulation = double_regret_accumulation + double_regret
        discount_slide_window_accumulation = discount_slide_window_accumulation + discount_slide_window
        discount_accumulation = discount_accumulation + discount
        slide_window_accumulation = slide_window_accumulation + slide_window

        dou_accuracy_acc+=dou_accuracy
        ds_accuracy_acc+=ds_accuracy
        sli_accuracy_acc+=sli_accuracy
        dis_accuracy_acc+=dis_accuracy

    # 画图
    print(" double:", dou_accuracy_acc/trial," discount_slide:",ds_accuracy_acc/trial, " slide:",sli_accuracy_acc/trial, " discount:",dis_accuracy_acc/trial, flush=True)

    plt.plot(np.array(range(1, Time + 1)), double_regret_accumulation / trial, label='双反馈')
    plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial,
             label='滑动窗口双反馈'+' 窗口容量:' + str(slide_window_side))
    plt.plot(np.array(range(1, Time + 1)), discount_accumulation / trial,
             label='折扣系数双反馈' + ' 折扣系数:' + str(discount_factor_d))
    plt.plot(np.array(range(1, Time + 1)), discount_slide_window_accumulation / trial,
             label='折扣系数滑动窗口双反馈' + ' 容量:' + str(slide_window_side)+ ' 折扣系数:' + str(discount_factor_ds))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()

# 窗口大小比较
if __name__ == '__main_ _':
    # 生成50个trans_prob
    # datetime slot
    Time = setting.Time
    # 实验次数
    trial = 5
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
    slide_window_sides = [0.1, 0.2, 0.3, 0.5, 1, 3, 5]
    for slide_window_side in slide_window_sides:
        slide_window_accumulation = np.array([0.0] * Time)
        discount_accumulation = np.array([0.0] * Time)
        discount_slide_window_accumulation = np.array([0.0] * Time)

        ds_accuracy_acc = 0
        sli_accuracy_acc = 0
        dis_accuracy_acc = 0
        for i in range(trial):
            discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side* small_circle, succ_count, fail_count, best_arms)
            print("discount slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            discount_slide_window_accumulation = discount_slide_window_accumulation + discount_slide_window

            ds_accuracy_acc+=ds_accuracy

    # 画图
        print(" discount_slide:",ds_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), discount_slide_window_accumulation / trial,
                label='折扣系数滑动窗口双反馈' + ' 容量:' + str(slide_window_side)+ ' 折扣系数:' + str(discount_factor_ds))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()

# 折扣系数比较
if __name__ == '__main_ _':
    # 生成50个trans_prob
    # datetime slot
    Time = setting.Time
    # 实验次数
    trial = 10
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
    discount_factors = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95]
    slide_window_side = setting.slide_window_side
    for discount_factor in discount_factors:
        discount_factor_ds = discount_factor_d = discount_factor

        slide_window_accumulation = np.array([0.0] * Time)
        discount_accumulation = np.array([0.0] * Time)
        discount_slide_window_accumulation = np.array([0.0] * Time)

        ds_accuracy_acc = 0
        sli_accuracy_acc = 0
        dis_accuracy_acc = 0
        for i in range(trial):
            discount_slide_window, ds_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle,discount_factor_ds, slide_window_side, succ_count, fail_count, best_arms)
            print("discount slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, succ_count, fail_count, best_arms)
            print("slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            discount, dis_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, discount_factor_d, Time, succ_count, fail_count, best_arms)
            print("discount finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            
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
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()

# slide_window 测试
if __name__ == '__main_ _':
    # 生成50个trans_prob
    # datetime slot
    Time = setting.Time
    # 实验次数
    trial = 1
    # 大周期
    big_circle = setting.big_circle
    # 折扣系数
    
    succ_count = setting.succ_count
    fail_count = 5

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
    slide_window_accumulation = np.array([0.0] * Time)
    sli_accuracy_acc = 0

    for i in range(trial):
        slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, succ_count, fail_count, best_arms)
        print("slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        slide_window_accumulation = slide_window_accumulation + slide_window
        sli_accuracy_acc+=sli_accuracy

    # 画图
        print(" slide:",sli_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial,
                label='滑动窗口双反馈'+' 窗口容量:' + str(slide_window_side))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()

# succ_count测试
if __name__ == '__main_ _':
    # 生成50个trans_prob
    # datetime slot
    Time = setting.Time
    # 实验次数
    trial = setting.trial
    # 大周期
    big_circle = setting.big_circle
    # 折扣系数
    
    succ_counts = [1,2,3,4,5,6,7,8,9,10]
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
    discount_factor = 1
    slide_window_side = setting.slide_window_side
    slide_window_accumulation = np.array([0.0] * Time)
    sli_accuracy_acc = 0

    for succ_count in succ_counts:
        for i in range(trial):
            slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, succ_count, fail_count, best_arms)
            print("slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            slide_window_accumulation = slide_window_accumulation + slide_window
            sli_accuracy_acc+=sli_accuracy

    # 画图
        print(" slide:",sli_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial,
                label='滑动窗口双反馈'+' succ_count:' + str(succ_count))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()


# fail_count测试
if __name__ == '__main_ _':
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
    slide_window_accumulation = np.array([0.0] * Time)
    sli_accuracy_acc = 0

    for fail_count in fail_counts:
        for i in range(trial):
            slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, succ_count, fail_count, best_arms)
            print("slide finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
            slide_window_accumulation = slide_window_accumulation + slide_window
            sli_accuracy_acc+=sli_accuracy

    # 画图
        print(" slide:",sli_accuracy_acc/trial, flush=True)
        plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial,
                label='滑动窗口双反馈'+' succ_count:' + str(fail_count))
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title('仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.show()