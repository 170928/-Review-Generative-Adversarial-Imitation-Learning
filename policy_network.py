import tensorflow as tf

class PolicyNet:
    def __init__(self, state_dim, action_dim, name, action_type):

        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_type = action_type

        with tf.variable_scope(self.name):
            # 예제 코드를 따라서 observation space 는 2 인 경우로 만들고 있습니다.
            self.observation = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_dim.shape), name='observations')
            #print(self.observation)

            # High Dimensional Continuous Control Using Generalized Advantage Estimation Paper
            # Proximal Policy Optimization Algorithm Paper
            # Actor Critic Style PPO Optimization is used
            # Therefore, we use two neural network models
            # (1) Actor model == policy estimation network
            # (2) Critic model == value estimation network

            '''
            학습시에 레이어의 유닛 수를 모두 60으로 하는 경우 
            제대로 학습이 안되는 경우가 발생 
            오버피팅 문제일까요 
            '''

            with tf.variable_scope('policy'):
                h1 = tf.layers.dense(inputs=self.observation, units=20, activation=tf.tanh)
                h2 = tf.layers.dense(inputs=h1, units=20, activation=tf.tanh)
                h3 = tf.layers.dense(inputs=h2, units=self.action_dim, activation=tf.tanh)
                self.action_probs = tf.layers.dense(inputs=h3, units=self.action_dim, activation=tf.nn.softmax)
                #self.policy_scope = tf.get_variable_scope().name
                #print(self.action_probs)

            with tf.variable_scope('value'):
                v1 = tf.layers.dense(inputs=self.observation, units=20, activation=tf.tanh)
                v2 = tf.layers.dense(inputs=v1, units=20, activation=tf.tanh)
                self.value_estimates = tf.layers.dense(inputs=v2, units=1, activation = None)
                #self.value_scope = tf.get_variable_scope().name

            #self.policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.policy_scope)
            #self.value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.value_scope)

            if action_type == 'stochastic':
                self.action_stochastic = tf.multinomial(tf.log(self.action_probs), num_samples=1)
                self.actions = tf.reshape(self.action_stochastic, [-1])

                #print(self.action_stochastic)
                #print(self.actions)

            else:
                self.actions = tf.argmax(self.action_probs, axis = 1)

            self.scope = tf.get_variable_scope().name


    def estimate(self, obs):
        return tf.get_default_session().run([self.actions, self.value_estimates], feed_dict={self.observation: obs})

    def get_probs(self, obs):
        return tf.get_default_session().run(self.action_probs, feed_dict={self.observation : obs})

    def get_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.scope)
