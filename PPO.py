import tensorflow as tf
import copy

'''
PPO의 학습 방법을 사용하기 위해서 policy network (Old params, Current params) 를 받아와 학습을 위한 loss function 계산하고 train 하는 알고리즘이 담긴 파일입니다.
(1) value function loss : MSE
(2) policy function loss : GAE (Generalized Advantage Estimation) with entropy  (i.e., diff(params){ log(prob) } * GAE + entropy
(3) value function loss + policy function loss + entropy
'''

'''
Generalized Advantage Estimation (GAE) is important factor in recent reinforcement learning algorithm for stabilized fast training.
Therefore, you need to understand mathmatically about it. 
'''

