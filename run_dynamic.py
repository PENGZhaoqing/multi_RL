# -*- coding: utf-8 -*-

import argparse
import logging
from envs_wrapper import Env
from config import Config
from utils import *
from method.model_factory import ModelFactory


# file_name = os.path.basename(__file__)
# kernprof -l script_to_profile.py
# python -m line_profiler script_to_profile.py.lprof

# @profile
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--t-max', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--steps-per-epoch', type=int, default=50000)
    parser.add_argument('--testing', type=int, default=1)
    parser.add_argument('--continue-training', type=int, default=0)
    parser.add_argument('--epoch-num', type=int, default=30)
    parser.add_argument('--start-epoch', type=int, default=20)
    parser.add_argument('--testing-epoch', type=int, default=24)
    parser.add_argument('--max-agent-num', type=int, default=20)
    parser.add_argument('--method', type=str, default='AMP')
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--game-name', type=str, default='traffic')
    parser.add_argument('--mode', type=str, default='hard')
    parser.add_argument('--entropy-wt', type=float, default=0.05)
    parser.add_argument('--dynamic', type=bool, default=False)
    parser.add_argument('--vis', type=bool, default=False)
    parser.add_argument('--gated', type=bool, default=False)
    parser.add_argument('--use-attend', type=int, default=0)
    parser.add_argument('--message-hidden', type=int, default=100)

    args = parser.parse_args()
    args.dir = os.path.dirname(__file__)
    print args.dir
    config = Config(args)
    t_max = args.t_max
    gamma = config.gamma
    steps_per_epoch = args.steps_per_epoch
    testing_epoch = args.testing_epoch
    epoch_num = args.epoch_num
    epoch_range = range(epoch_num)

    max_agent_num = args.max_agent_num
    num_envs = args.num_envs
    vis = args.vis

    testing = args.testing
    testing = True if testing == 1 else False

    # Init envs and actions
    envs = []
    for i in range(args.num_envs):
        envs.append(Env(id=i, config=config))

    # Init actions and state_dim (for one agent)
    state_dim = envs[0].ple.get_states().shape[1]
    action_set = envs[0].ple.get_action_set()
    action_maps = []
    for idx in action_set:
        tmp = []
        for action in action_set[idx]:
            tmp.append(action_set[idx][action])
        action_maps.append(tmp)
    action_num = len(action_maps[0])

    config.act_space = action_num
    config.state_dim = state_dim

    '''
    初始化Agent
    '''
    agents = ModelFactory.create(config)

    if testing:
        envs[0].ple.force_fps = True
        envs[0].game.draw = True
        envs[0].ple.display_screen = True
        agents.load_params(testing_epoch)
        epoch_range = range(testing_epoch, 1 + testing_epoch)
        steps_per_epoch = 5000
    else:
        assert num_envs > 1
        assert vis == False
        logging_config(logging, config)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)

    if args.method in ['DIAL', 'DIAL_rnn']:
        print_params(logging, agents.model._curr_module)
    else:
        print_params(logging, agents.model)

    import time
    time.sleep(10)
    '''
    游戏在每个周期里执行steps_per_epoch=50000步
    '''
    for epoch in epoch_range:
        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()

        for i in range(num_envs):
            agent_num, agent_mask = envs[i].reset_cirruculumn(epoch)

        while steps_left > 0:
            episode += 1
            time_episode_start = time.time()
            episode_collisions = np.zeros(len(envs), dtype=int)
            episode_rewards = np.zeros(num_envs, dtype=np.float)
            episode_values = np.zeros(max_agent_num)
            episode_step = 0

            '''
            初始化网络输入值
            '''
            # init last_step_comms, last_hidden_states
            agents.init_variant()
            step_obs = np.zeros((num_envs, max_agent_num, state_dim))
            for i in range(num_envs):
                envs[i].reset_var()
                envs[i].ple.reset_game()
                obs = envs[i].ple.get_states()
                step_obs[i] = obs[:]

            all_done = False
            t = 1
            training_steps = 0
            sample_num = 0.0
            while not all_done:

                '''
                agents输入当前状态step_obs到网络里, 网络输出step_outputs
                step_outputs: (max_agent_num, env_num, state_dim)
                '''
                step_outputs = agents.forward(step_obs, agent_mask, is_train=False)

                '''
                解析step_outputs, 获得agents当前执行的动作和每个动作对应的Q值(未来收益的期望值)
                step_actions/step_value: (env_nums, agent_num, 1)
                '''
                step_values_np, step_actions = agents.parse_outputs(step_outputs)
                agents.last_state_update(t, step_outputs)

                '''
                执行游戏
                '''
                for i in range(num_envs):
                    if not envs[i].done:
                        '''
                        在每个环境实例中,调用act方法执行每个agent的动作, 获得下一步的状态next_ob和当前获得的reward,
                        以及环境时候结束(done)和辅助信息(info)
                        step_obs: (gent_num, state_dim), step_actions: (agent_num, 1) , step_value: (agent_num,1)
                        '''
                        action_list = [action_maps[idx][step_actions[i, idx, 0]] for idx in range(config.max_agent_num)]
                        next_ob, reward, done, info = envs[i].ple.act(action_list)

                        '''
                        将状态,动作,Q值,奖励和辅助信息缓冲到对应的envs实例中,这是因为我们有可能执行3步游戏,才更新1次网络,此时t_max=3
                        '''
                        envs[i].append(obs=step_obs[i], actions=step_actions[i], values=step_values_np[i],
                                       rewards=reward.reshape(-1, 1), alphas=envs[i].game.info2)
                        envs[i].done = done

                        '''
                        更新当前的step_obs为next_obs
                        '''
                        step_obs[i] = next_ob[:]

                        episode_rewards[i] += sum(reward)
                        episode_collisions[i] += sum(i < 0 for i in reward)
                        episode_step += 1

                '''
                当t=t_max时,开始更新网络
                '''
                if t == t_max and not testing:

                    '''
                    首先计算出下一个状态的Q值, extra_values (agent_num, envs_num, 1)
                    并转置为extra_values (envs_num, agent_num, 1)
                    '''
                    extra_outputs = agents.forward(step_obs, agent_mask, is_train=False)
                    extra_values = extra_outputs[max_agent_num:2 * max_agent_num]
                    for i in range(len(extra_values)):
                        extra_values[i] = extra_values[i].asnumpy()
                    extra_values = np.transpose(extra_values, (1, 0, 2))

                    '''将extra_values缓存至envs实例中,如果此时为最后一个状态,values全为0'''
                    for i in range(num_envs):
                        if envs[i].done:
                            extra_values = np.zeros((num_envs, max_agent_num, 1))
                            envs[i].values = np.append(envs[i].values, extra_values[i], axis=1)
                        else:
                            envs[i].values = np.append(envs[i].values, extra_values[i], axis=1)

                    '''
                    使用贝尔曼方程计算更新网络的误差advs=(r+V(t+1)-V(t))^2
                    advs (envs_num, agent_num, t_max)
                    '''
                    advs = []
                    for i in range(num_envs):
                        R = envs[i].values[:, -1].reshape(-1, 1)
                        tmp_R = [[] for _ in range(max_agent_num)]
                        for j in range(envs[i].rewards.shape[1])[::-1]:
                            R = envs[i].rewards[:, j].reshape(-1, 1) + gamma * R
                            tmp_R = np.append(tmp_R, R - envs[i].values[:, j].reshape(-1, 1), axis=1)
                        advs.append(tmp_R[:, ::-1])

                    '''
                    根据advs,构造policy_grads梯度用于更新actor网络,构造value_grads梯度用于更新critic网络
                    env_actions (agent_nums, env_nums), env_obs (env_nums, agent_nums, t_max, state_dim)
                    '''
                    env_actions = [[] for _ in range(max_agent_num)]
                    env_obs = []
                    for i in range(num_envs):
                        env_actions = np.append(env_actions, envs[i].actions, axis=1)
                        env_obs.append(envs[i].obs.reshape(max_agent_num, -1, state_dim))
                    policy_grads, value_grads = agents.calculate_grads(advs, env_actions, agent_num)
                    policy_grads.extend(value_grads)

                    '''
                    再次给agents网络喂数据env_obs, 让网络正向传播, 此时的env_obs是多个step_obs的集合(若t_max>1),
                    正向传播的目的是让网络记住一些用于更新的状态,此时网络参数is_train=True
                    '''
                    envs_outputs = agents.forward(env_obs, agent_mask, bucket_key=t_max, is_train=True)

                    '''
                     构造辅助矩阵来辅助更新网络(只有SAMP方法需要,也是本文的创新点)
                    '''
                    policy_grads = agents.gen_matrix(policy_grads, envs, envs_outputs)

                    '''
                    误差(梯度)方向传播backward, 并更新网络update
                    '''
                    agents.model.backward(out_grads=policy_grads)
                    agents.model.update()

                    training_steps += 1
                    if training_steps % 10 == 0:
                        sample_num += 1
                        agents.cal_episode_values(envs_outputs, episode_values)

                    '''
                    重置envs实例中的缓存,t=0
                    '''
                    for i in range(num_envs):
                        envs[i].clear()

                    t = 0
                '''
                若所有当前游戏实例已到达结束状态,跳出while循环
                '''
                all_done = np.all([envs[i].done for i in range(num_envs)])
                t += 1

            steps_left -= episode_step
            epoch_reward += episode_rewards
            time_episode_end = time.time()
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%.2f, Collision:%.2f, fps:%.1f, " \
                       % (epoch, episode, steps_left, episode_step, steps_per_epoch,
                          np.mean(episode_rewards), np.mean(episode_collisions),
                          episode_step / (time_episode_end - time_episode_start))

            info_str += 'Values: (' + ', '.join(('%.1f' % f) for f in tuple(episode_values)) + ')' \
                        + ', Collisions:' + str(tuple(episode_collisions)) \
                        + '\nRewards: ' + np.array2string(episode_rewards, precision=2)
            logging.info(info_str)
            print info_str
            # print episode_info

        end = time.time()
        fps = steps_per_epoch / (end - start)
        agents.save_params(epoch) if not testing else agents.vis_epoch()
        epoch_info = "Epoch:%d, FPS:%f, Avg Reward: %f/%d" % (epoch, fps, np.mean(epoch_reward) / float(episode),
                                                              episode)
        logging.info(epoch_info)
        print epoch_info


if __name__ == '__main__':
    main()
