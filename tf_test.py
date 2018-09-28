import tensorflow as tf
import numpy as np




if __name__ =="__main__":


    with tf.Session() as sess:

        probs = tf.convert_to_tensor(np.array([[0.1, 0.15, 0.5,  0.25], [0.5, 0.2, 0.05, 0.25]]))

        # index return
        #
#        res = tf.multinomial(tf.log(probs), num_samples=1)
#        res = tf.reshape(res, [-1])
        res = tf.argmax(probs, axis= 1)

        print(sess.run(res))


        with tf.variable_scope('A'):

            with tf.variable_scope('B'):
                b = tf.get_variable_scope().name

            with tf.variable_scope('C'):
                c = tf.get_variable_scope().name


        print(b, c)

        test = tf.random_normal(tf.shape([0, 0, 1, 0]), mean=0.2, stddev=0.1, dtype=tf.float32) / 1.2

        print(sess.run(test))