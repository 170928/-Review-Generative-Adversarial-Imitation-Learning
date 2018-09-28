import tensorflow as tf

class discriminaotr:
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
                self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
                self.expert_a = tf.placeholder(dtype=tf.float32, shape=[None])

                with tf.variable_scope('recover'):
                    self.expert_one_hot = tf.one_hot(indices = self.expert_a, depth = self.action_dim, on_value = 1, off_value = 0)
                    '''
                    For stabilise training, we need to add noise to recovered one_hot tensor.
                    i.e, [0 0 1 0] => [0.2331, 0.1313, 1, 0.4131]
                    '''
                    self.expert_one_hot += tf.random_normal(tf.shape(self.expert_one_hot), mean=0.2, stddev=0.1,dtype=tf.float32)/1.2

                '''
                The neural network receive (state, action) pair.
                Therefore, we need to concate the expert_s and expert_one_hot tensor.
                '''
                self.expert_input = tf.concat(self.expert_s, self.expert_one_hot)

            with tf.variable_scope('learner'):
                



