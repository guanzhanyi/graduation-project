import numpy as np


class SlideWindow(object):
    def __init__(self, arm_num, size, discount_factor, add, sub):
        if size < 1:
            raise Exception('size不能小于1。x 的值为: {}'.format(size))
        self.cap = size
        self.is_discount = True
        self.discount_factor = discount_factor
        self.sums = [[0.0, 0.0] for i in range(arm_num)]
        self.queue = []
        self.arm_counts = [0 for i in range(arm_num)]
        self.add = add
        self.sub = sub
        self.arm_num = arm_num

    def is_full(self):
        return len(self.queue) == self.cap + 1

    def is_any_arm_full(self, cho):
        for arm in self.arm_counts:
            if arm > self.cap//2 and cho != arm:
                return True
        return False

    def is_cho_full(self, cho):
        if self.arm_counts[cho] > self.cap//2:
            return True
        return False

    # 添加元素
    def append(self, elems):
        self.queue.append(elems)
        if self.is_full():
            self.arm_counts[self.queue[0][0]] -= 1
            self.queue = self.queue[1:]
        cho = elems[0]
        self.arm_counts[cho] += 1
        # 如果cho超过一半且cho失败了，失败次数增加
        # 如果有出现超过一半但不是cho，而且cho成功了，超过次数增加
        if elems[1] == 1 and self.is_any_arm_full(cho):
            elems[1] = self.add
        if elems[0] == 0 and self.is_cho_full(cho):
            elems[2] = self.sub
        self.sums = [[0.0, 0.0] for i in range(self.arm_num)]
        for i in range(0, len(self.queue)):
            index = self.queue[i][0]
            if self.queue[i][1] > 0:
                self.sums[index][0] += (self.queue[i][1] *
                                      self.discount_factor ** (len(self.queue) - i))
            else:
                self.sums[index][1] += (self.queue[i][2] *
                                      self.discount_factor ** (len(self.queue) - i))
        
    def clear(self):
        self.queue = []
        self.sums = [[0.0, 0.0] for i in range(self.arm_num)]

    def getSum(self, arm_num):
        return self.sums[arm_num]


def d_slide_window(time_slot, pred_prob, trans_prob, rates, bests, circle, change_time, discount_factor, silde_size, add, sub, best_arms):
    TEST = 0
    arm_num = len(rates)
    pred_right_count = 0
    choices = np.array([0]*arm_num)

    regret = np.array([0.0] * time_slot)
    S_n_1 = np.array([0] * arm_num)
    F_n_1 = np.array([0] * arm_num)
    a_n = np.array([0.0] * arm_num)
    b_n = np.array([0.0] * arm_num)
    # 滑动窗口
    sw = SlideWindow(arm_num, int(silde_size), discount_factor, add, sub)
    index = 0
    k = len(trans_prob)
    for t in range(time_slot):
        t_remainder = t % circle
        if t_remainder % change_time == 0:
            index = (index + 1) % k
        for r in range(arm_num):
            a_n[r] = np.random.beta(S_n_1[r] + 1, F_n_1[r] + 1)
            sum = sw.getSum(r)
            b_n[r] = np.random.beta(sum[0] + 1.0, sum[1] + 1.0)
            if TEST:
                print(r, sum[0], sum[1])
        cho = np.argmax(a_n * b_n * rates)
        choices[cho] += 1
        if TEST:
            print('size: ',len(sw.queue))
        X = np.random.choice([1, 0], size=1, p=[
                             pred_prob[cho], 1 - pred_prob[cho]])[0]
        Y = np.random.choice([1, 0], size=1, p=[
                             trans_prob[index][cho], 1 - trans_prob[index][cho]])[0]
        if X == 1:
            S_n_1[cho] += 1
        else:
            F_n_1[cho] += 1
        if Y == 1:
            sw.append([cho, 1, 0])
        else:
            sw.append([cho, 0, 1])
        if t == 0:
            regret[t] = bests[index] - X * Y * rates[cho]
        else:
            regret[t] = regret[t - 1] + bests[index] - X * Y * rates[cho]
        if best_arms[index] == cho:
            pred_right_count += 1
    return regret, pred_right_count/time_slot, choices
