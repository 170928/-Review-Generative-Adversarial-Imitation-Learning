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
> High-Dimensional Continuous Control Using Generalized Advatage Estimation 논문에서 볼 수 있습니다.
'''

class PPO():
    def __int__(self, OldPolicy, CurPolicy, gamma=0.99, clip_value=0.2, c1 = 1, c2 = 0.01 ):
        '''
        :param OldPolicy: Old Policy Network made by policy_network.py 가 들어옵니다.
        :param CurPolicy:  Current Policy Network made by policy_network.py
        :param gamma: Discounted reward factor
        :param clip_value: PPO에서 사용하는 r(theta)를 clip하기 위한 변수
        > PPO 논문을 참조하세요
        :param c1: value network의 결과값이 보여주는 loss 값을 고려하는 비중을 조절하는 파라미터
        :param c2: policy entropy의 값에 대한 loss 값의 비중을 조절하는 파라미터
        :return:
        '''

        self.curPolicy = CurPolicy
        self.oldPolicy = OldPolicy
        self.gamma = gamma
        self.clip_value = clip_value
        self.c1 = c1
        self.c2 = c2

        '''
        Gradient 계산시에 사용하기 위해서 policy network의 trainable_variable들을 가져옵니다. 
        '''
        self.cur_params = self.curPolicy.get_params()
        self.old_params = self.oldPolicy.get_params()

        '''
        old network의 variable들을 cur network의 variable들로 대체합니다.
        '''
        with tf.variable_scope('Assign_Cur2Old'):
            self.assign_op = []
            for var, old in zip(self.cur_params, self.old_params):
                self.assign_op.append(tf.assign(old, var))

        '''
        PPO 방식의 Train을 위해서 필요한 값들을 받아오는 placeholder를 선언합니다. 
        '''
        with tf.variable_scope('PPOph'):
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None], name='actions')
            '''
            self.rewards는 Discriminator가 (s,a) pair 에 대해서 판단한 확률값이 나오게 됩니다. 
            '''
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.GAE = tf.placeholder(dtype=tf.float32, shape=[None], name='Generalized Advantage Estimation')
            self.nextV = tf.placeholder(dtype=tf.float32, shape=[None], name='Estimated Value')

        action_prob = self.curPolicy.action_probs
        old_action_prob = self.oldPolicy.action_probs

        '''
        Important Sampling 기법의 Ratio를 구한다. 
        '''
        self.action_probs = tf.reduce_sum(action_prob * tf.one_hot(action_prob, depth = action_prob.shape[1]), axis=1)
        self.old_action_probs = tf.reduce_sum(old_action_prob * tf.one_hot(old_action_prob, depth = old_action_prob.shape[1]), axis=1)

        with tf.variable_scope('Loss'):
            '''
            위에서 계산한 Important Sampling Ratio의 경우
            tf.divide(self.action_probs, self.old_action_probs)로 구하는 식이 PPO 논문에서 사용 
            그러나, 성능을 위해서 exponentail( log (prob/prob) ) 형태를 사용 
            '''
            self.ratios = tf.exp(tf.log(tf.clip_by_value(self.action_probs, 1e-10, 1.0)) - tf.log(tf.clip_by_value(self.old_action_probs, 1e-10, 1.0)))

            '''
            PPO 논문에서 제안하는 loss function에서의 Important Sampling Loss를 Clip 하는 operation
            PPO 논문 equation (7)을 확신하시면 됩니다. 
            '''
            self.clipped_ratio = tf.clip_by_value(self.ratios, clip_value_min=self.ratios-self.clip_value, clip_value_max=self.ratios+self.clip_value)
            loss_clip = tf.minimum(tf.multiply(self.GAE, self.ratios), tf.multiply(self.GAE, self.clipped_ratio))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

            












