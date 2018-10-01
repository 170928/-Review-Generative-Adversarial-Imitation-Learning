import tensorflow as tf

class discriminator:

    def __init__(self, state_dim, action_dim, name, action_type):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.action_type = action_type

        with tf.variable_scope(self.name):
            # discriminaor estimates whether the (s, a) is the pair of learner or not.

            with tf.variable_scope('expert'):
                '''
                we just know the state and selected action of expert
                This means that we need action placeholder that has [None] shape
                Using expert_a placeholder, we have to recover it to [action_dim] tensor.
                '''
                self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_dim.shape))
                self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])

                with tf.variable_scope('recover'):
                    self.expert_one_hot = tf.one_hot(indices = self.expert_a, depth = self.action_dim)
                    self.expert_one_hot = tf.to_float(self.expert_one_hot)
                    '''
                    For stabilise training, we need to add noise to recovered one_hot tensor.
                    i.e, [0 0 1 0] => [0.2331, 0.1313, 1, 0.4131]
                    '''
                    self.expert_one_hot += tf.random_normal(tf.shape(self.expert_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2

                '''
                The neural network receive (state, action) pair.
                Therefore, we need to concate the expert_s and expert_one_hot tensor.
                '''
                self.expert_input = tf.concat([self.expert_s, self.expert_one_hot], axis=1)

            with tf.variable_scope('learner'):
                self.learner_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_dim.shape))
                self.learner_a = tf.placeholder(dtype=tf.int32, shape=[None])

                with tf.variable_scope('recover'):
                    self.learner_one_hot = tf.one_hot(indices = self.learner_a, depth = self.action_dim)
                    self.learner_one_hot = tf.to_float(self.learner_one_hot)
                    self.learner_one_hot += tf.random_normal(tf.shape(self.learner_one_hot), mean=0.2, stddev=0.1, dtype = tf.float32)/1.2

                self.learner_input = tf.concat([self.learner_s, self.learner_one_hot], axis=1)

            with tf.variable_scope('Discriminator') as network:
                '''
                Discriminator network
                Input : (state , action) pair
                Output : Probability whether the (state, action) pair is learner's or not.
                '''
                self.expert_action_probs = self.build_network(self.expert_input)
                '''
                같은 모델을 사용해야 하므로, 다음과 같이 scope 내의 변수를 선언하여 네트워크를 구성 함수를 호출한다.
                '''
                network.reuse_variables()
                self.learner_action_probs = self.build_network(self.learner_input)

            with tf.variable_scope('Loss'):
                self.D_loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(self.expert_action_probs, 0.01, 1)))
                self.D_loss_learner = tf.reduce_mean(tf.log(tf.clip_by_value(self.learner_action_probs, 0.01, 1)))
                self.loss = -( self.D_loss_expert + self.D_loss_learner)
                tf.summary.scalar('discriminator_Loss', self.loss)


            with tf.variable_scope('Optimizer'):
                self.optimizer = tf.train.AdamOptimizer()
                self.train = self.optimizer.minimize(self.loss)

            with tf.variable_scope('D_Reward_Connection'):
                self.D_reward = tf.log(tf.clip_by_value(self.learner_action_probs, 1e-10, 1))

    def build_network(self, input):
        h1 = tf.layers.dense(inputs=input, units=60, activation=tf.nn.leaky_relu, name='layer1')
        h2 = tf.layers.dense(inputs=h1, units=60, activation=tf.nn.leaky_relu, name='layer2')
        h3 = tf.layers.dense(inputs=h2, units=60, activation=tf.nn.leaky_relu, name='layer3')
        '''
        디스크리미네이터의 결과 값은 0~1 사이의 학습자 혹은 전문가의 결과여부에 대한 확률을 의미하므로
        다음과 같이 시그모이드를 거쳐서 나온 결과 값을 사용합니다 
        '''
        probs = tf.layers.dense(inputs=h3, units=1, activation=tf.sigmoid, name='prob')
        return probs

    def update(self, expert_s, expert_a, learner_s, learner_a):
        return tf.get_default_session().run(self.train, feed_dict={self.expert_s : expert_s, self.expert_a : expert_a,
                                                                    self.learner_s : learner_s, self.learner_a : learner_a})

    def get_reward(self, learner_s, learner_a):
        return tf.get_default_session().run(self.D_reward, feed_dict = {self.learner_s : learner_s, self.learner_a : learner_a})
