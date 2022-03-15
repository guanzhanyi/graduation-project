'''
Author: guanzhanyi
Date: 2022-01-26 12:56:14
LastEditors: guanzhanyi
LastEditTime: 2022-03-15 21:52:26
FilePath: \graduation-project\setting.py
Description: 

Copyright (c) 2022 by guanzhanyi/xxx, All Rights Reserved. 
'''
import numpy as np
trans_prob =  np.array([[0.9, 0.8, 0.65, 0.5, 0.1],[0.99, 0.85, 0.75, 0.15, 0.01]])

pred_prob = np.array([0.1, 0.3, 0.6, 0.7, 0.9])

rates = np.array([2, 3, 5, 7, 9])

# 滑动窗口
slide_window_side = 300
# 周期
big_circle = 3600
# 时长
Time = 10000
# 成功增加次数
succ_count = 2
# 失败增加次数
fail_count = 2
# 置零偏差时机
piancha = 0 
# 置零滑动窗口算法折扣系数
discount_factor_ds = 0.95
# 折扣TS算法折扣系数
discount_factor_d = 0.95
# 实验次数
trial = 5
# print(trans_prob[0]*pred_prob*rates)
# print(trans_prob[1]*pred_prob*rates)  