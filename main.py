# Reference Code
# https://github.com/chagmgang/gail/blob/master/gail_move2beacon/policy_net.py
# 차금강님의 코드를 참조하였습니다.
# https://github.com/uidilr/gail_ppo_tf/blob/master/run_gail.py
# Thank you.


import os
import argparse
import gym
import numpy as np
import tensorflow as tf
from PPO import PPOalgorithm
from policy_network import PolicyNet
from discriminator_network import discriminator

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='./log/train/')
    parser.add_argument('--savedir', help='save directory', default='./save_model/')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1000000))
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')
    env.seed(0)

    state_dim = env.observation_space
    action_dim = env.action_space.n
    '''
    PolicyNet(state_dim, action_dim, name, action_type)
    '''
    curPolicy = PolicyNet(state_dim, action_dim, 'currentPolicy', 'stochastic')
    oldPolicy = PolicyNet(state_dim, action_dim, 'oldPolicy', 'stochastic')

    PPO = PPOalgorithm( oldPolicy, curPolicy,  gamma=args.gamma)
    Discriminator = discriminator(state_dim, action_dim, 'Discriminator', 'stochastic')


    expert_observations = np.genfromtxt('trajectory/observations.csv')
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    saver = tf.train.Saver()
    checkpoint_path = os.path.join(args.savedir, "model")
    ckpt = tf.train.get_checkpoint_state(args.savedir)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        if ckpt and ckpt.model_checkpoint_path:
            print("[Restore Model]")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("[Initialzie Model]")
            sess.run(tf.global_variables_initializer())



        obs = env.reset()
        reward = 0
        success_num = 0

        for iteration in range(args.iteration):

            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0

            while True:
                env.render()
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                '''
                현재 정책에 따라서 observation을 보고 action과 V(s)를 예측합니다. 
                '''
                act, v_pred = curPolicy.estimate(obs=obs)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                next_obs, reward, done, info = env.step(act)

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            print(iteration, sum(rewards), success_num)

            if sum(rewards) >= 190:
                success_num += 1
                if success_num >= 1:
                    saver.save(sess, args.savedir + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            '''
            tf.placholder에 데이터를 넣기 위해서 list를 numpy array로 변환 합니다. 
            '''
            observations = np.reshape(observations, newshape=[-1] + list(state_dim.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            '''
            Discriminator를 업데이트 합니다. 
            '''
            for i in range(2):
                sample_indices = (np.random.randint(expert_observations.shape[0], size=observations.shape[0]))
                inp = [expert_observations, expert_actions]
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                Discriminator.update(sampled_inp[0], sampled_inp[1], observations, actions)
            '''
            GAIL 의 논문을 참조하세요.
            GAIL 논문의 reward 는 Discriminator의 (s,a) pair 에 대한 learner의 policy에서 생성된 것인지를 판단하는 
            확률 입니다. 
            '''
            d_rewards = Discriminator.get_reward(learner_s=observations, learner_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_next_preds=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            '''
            Policy network들을  update합니다. 
            '''
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.update()


            for epoch in range(15):
                '''
                MiniBatch 
                '''
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)

                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          GAEs=sampled_inp[2],
                          rewards=sampled_inp[3],
                          estimated_v=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      GAEs=inp[2],
                                      rewards=inp[3],
                                      estimated_v=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()




if __name__=="__main__":
    args = argparser()
    main(args)








