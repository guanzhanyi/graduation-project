import setting
import ss_window
import double_feedback
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys
sys.stdout = open('_多组概率测试.txt', 'a',encoding='utf8')
print(datetime.now())

if __name__ == '__main__':
    # 生成50个trans_prob
    # datetime slot
    Time = 100000
    # 实验次数
    trial = setting.trial
    # 大周期
    big_circle = 9000
    # 折扣系数
    
    succ_count = setting.succ_count
    fail_count = setting.fail_count

    rates = setting.rates
    trans_prob = [[9.17232238e-01, 7.73003868e-01, 3.40853342e-01, 2.76486244e-01, 2.69665310e-01],
                  [9.42666589e-01, 7.94025527e-01, 1.05878116e-01, 1.01868742e-01, 9.98424323e-02],
                  [8.92913452e-01, 6.36422405e-01, 1.25317991e-01, 5.74160723e-02, 1.11644921e-02],
                  [9.17258079e-01, 8.57131912e-01, 8.13363729e-01, 3.06526305e-01, 2.08246941e-01],
                  [8.93332814e-01, 8.67544448e-01, 1.81616840e-02, 7.98680622e-03, 7.93849118e-03],
                  [9.45620682e-01, 6.88380586e-01, 4.79303152e-01, 4.77523542e-01, 4.27387721e-01],
                  [9.37241626e-01, 8.09231857e-01, 2.49947996e-01, 6.63134219e-02, 6.25369120e-02],
                  [8.62805221e-01, 7.44243470e-01, 3.40131766e-01, 3.21244253e-01, 2.32680729e-01],
                  [9.55516494e-01, 6.93126374e-01, 1.97086029e-01, 1.84617003e-01, 1.46633451e-01],
                  [9.20380506e-01, 6.53191389e-01, 2.75594457e-01, 8.54699160e-02, 6.52521115e-02],
                  [8.63661967e-01, 6.50762793e-01, 4.65076433e-01, 4.20228294e-01, 3.07252906e-02],
                  [9.14272276e-01, 5.69970768e-01, 1.93930214e-01, 1.27077544e-01, 3.69351471e-02],
                  [9.11609405e-01, 7.39642893e-01, 1.65055041e-02, 6.84033653e-03, 4.76220973e-04],
                  [9.42029756e-01, 7.06859427e-01, 5.97262271e-01, 5.77849331e-01, 5.10984428e-01],
                  [9.88055492e-01, 7.38314103e-01, 5.57487853e-01, 3.12110580e-01, 2.15128305e-01],
                  [9.62069825e-01, 8.48599191e-01, 2.96524767e-01, 2.45468200e-01, 1.17151704e-01],
                  [9.80951859e-01, 6.64149527e-01, 2.98710571e-01, 1.01841884e-01, 4.00836055e-02],
                  [9.14428320e-01, 8.59751735e-01, 5.71535773e-01, 4.44697946e-01, 1.78279806e-01],
                  [9.62124368e-01, 7.35565191e-01, 1.34541127e-01, 1.23933141e-01, 9.22558348e-02],
                  [8.98080730e-01, 6.50597784e-01, 1.37555005e-01, 4.02373388e-03, 2.07315560e-03],
                  [8.61754380e-01, 7.78119149e-01, 1.78596734e-01, 1.54861080e-01, 1.47282941e-01],
                  [9.21387183e-01, 8.45149269e-01, 6.61989589e-01, 2.77418070e-01, 2.25688497e-01],
                  [8.76874794e-01, 7.56230569e-01, 4.68098295e-01, 1.01478985e-02, 1.34871658e-03],
                  [9.67023236e-01, 8.39809256e-01, 6.15634738e-01, 4.89838333e-01, 3.20775813e-01],
                  [8.75997285e-01, 7.97829910e-01, 4.97421155e-01, 3.22137200e-01, 1.77837822e-02],
                  [8.78074214e-01, 8.52445430e-01, 7.57878426e-01, 1.38205612e-01, 4.89849757e-03],
                  [9.19596559e-01, 6.21464045e-01, 3.46530613e-01, 3.08327084e-02, 1.04960788e-02],
                  [9.46205769e-01, 7.28329300e-01, 1.18263790e-01, 6.83496873e-02, 5.00572934e-02],
                  [8.62978127e-01, 8.15794358e-01, 5.04972807e-01, 4.30019005e-01, 1.79629035e-01],
                  [8.84561234e-01, 7.11013286e-01, 9.53635217e-02, 7.71825928e-02, 4.04199910e-02],
                  [9.39299106e-01, 9.09940774e-01, 7.37923098e-01, 6.68029039e-01, 5.28288198e-01],
                  [8.58984237e-01, 7.86394779e-01, 2.58818456e-01, 1.03198248e-01, 8.12090307e-02],
                  [8.72863200e-01, 8.39169246e-01, 1.91977643e-01, 1.01190425e-01, 9.84272985e-02],
                  [9.88949877e-01, 7.67994173e-01, 2.02360567e-01, 1.82551094e-02, 1.40816729e-02],
                  [8.56483078e-01, 5.82045604e-01, 4.20425476e-01, 9.33663144e-02, 2.07907446e-04],
                  [9.17186949e-01, 7.37143401e-01, 5.63627719e-01, 1.09121521e-01, 1.03751496e-01],
                  [9.38159923e-01, 6.86610987e-01, 5.52330057e-01, 2.52580157e-01, 9.81698171e-02],
                  [9.74222094e-01, 9.03586909e-01, 5.68291250e-01, 4.69387916e-01, 1.61010598e-01],
                  [9.15572669e-01, 6.63973769e-01, 2.19822881e-01, 1.70425000e-01, 3.92638846e-02],
                  [8.67039265e-01, 6.71363593e-01, 2.03903847e-01, 2.01492713e-01, 1.83328901e-01],
                  [9.45385006e-01, 6.29053331e-01, 5.38632491e-01, 3.55220852e-01, 9.33596900e-02],
                  [9.61890573e-01, 7.69305362e-01, 3.94121695e-01, 1.45470288e-01, 1.04882250e-01],
                  [9.83138579e-01, 8.16153334e-01, 6.64771100e-01, 2.14756397e-01, 4.66763128e-02],
                  [8.76527634e-01, 8.14775919e-01, 6.89438846e-01, 3.52871602e-01, 1.03814174e-01],
                  [9.38581231e-01, 5.53431816e-01, 1.61972394e-01, 3.55644289e-03, 1.15302172e-03],
                  [9.20068593e-01, 8.88722104e-01, 7.88617456e-02, 7.32866092e-02, 4.18926207e-03],
                  [9.58478676e-01, 8.09547000e-01, 5.10249086e-01, 7.69205758e-02, 6.62751342e-02],
                  [9.34476491e-01, 6.35244921e-01, 3.97340834e-02, 8.99635588e-03, 8.84033407e-03],
                  [8.95891477e-01, 5.08359884e-01, 2.30129731e-01, 2.28853000e-01, 2.16522608e-01],
                  [9.28148365e-01, 5.62885342e-01, 3.73058381e-01, 1.60635734e-02, 9.51663211e-03]]
    # 滑动窗口大小
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
    double_regret_accumulation = np.array([0.0] * Time)

    small_circle = big_circle // len(trans_prob)
    slide_window_accumulation = np.array([0.0] * Time)
    double_regret_accumulation = np.array([0.0] * Time)

    sli_accuracy_acc = 0
    dou_accuracy_acc= 0

    for i in range(trial):
        double_regret , dou_accuracy, choices = double_feedback.double_feedback(Time, pred_prob, trans_prob, rates, bests,big_circle, small_circle,best_arms)
        print("double finished",datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        slide_window, sli_accuracy, choices = ss_window.d_slide_window(Time, pred_prob, trans_prob, rates, bests, big_circle, small_circle, 1,slide_window_side, succ_count, fail_count, best_arms)
        print(big_circle,datetime.now().strftime("%H:%M:%S"), choices, flush=True)
        double_regret_accumulation = double_regret_accumulation + double_regret
        slide_window_accumulation = slide_window_accumulation + slide_window

        dou_accuracy_acc+=dou_accuracy
        sli_accuracy_acc+=sli_accuracy

# 画图
    print("double:", dou_accuracy_acc/trial," slide:",sli_accuracy_acc/trial, flush=True)
    plt.plot(np.array(range(1, Time + 1)), double_regret_accumulation / trial, label='双反馈')
    plt.plot(np.array(range(1, Time + 1)), slide_window_accumulation / trial, label='滑动窗口')
    plt.xlabel('时隙')
    plt.ylabel('遗憾')
    plt.title(' 仿真次数:' + str(trial) + ' 周期:' + str(big_circle) + ' 传输概率组数:' + str(len(trans_prob)))
    plt.legend()
    plt.savefig('多组')
    # plt.show()
