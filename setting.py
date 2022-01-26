import numpy as np
trans_prob =  np.array([[0.9, 0.8, 0.65, 0.63, 0.1],[0.99, 0.85, 0.75, 0.15, 0.01]])

pred_prob = np.array([0.1, 0.3, 0.6, 0.7, 0.9])

rates = np.array([2, 3, 5, 7, 9])

slide_window_side = 300

big_circle = 3600

Time = 10000

add = 2

sub = 2

piancha = 0 

discount_factor_ds = 1
discount_factor_d = 1

# print(trans_prob[0]*pred_prob*rates)
# print(trans_prob[1]*pred_prob*rates)