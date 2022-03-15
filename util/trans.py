import numpy as np
trans_prob =  np.array([[0.9, 0.8, 0.65, 0.5, 0.1],[0.99, 0.85, 0.75, 0.15, 0.01]])

pred_prob = np.array([0.1, 0.3, 0.6, 0.7, 0.9])

rates = np.array([2, 3, 5, 7, 9])

print(trans_prob[0]*pred_prob*rates)
print(trans_prob[1]*pred_prob*rates)