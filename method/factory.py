import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA


class Factory(object):
    def __init__(self, state_dim, agent_num, signal_num, act_space, dir, folder, env, config):

        self.name = config.method

        self.comms_vis = [[] for _ in range(agent_num)]
        self.commNet_vis = [[] for _ in range(agent_num)]
        self.reward_target = [[] for _ in range(agent_num)]
        self.intera_matrix = np.zeros((agent_num, agent_num), dtype=int)
        self.connect_target = [[] for _ in range(agent_num)]
        self.brake_target = [[] for _ in range(agent_num)]
        self.alphas = np.zeros((agent_num, agent_num))

        if config.game_name == 'traffic':
            self.comm_norm = np.zeros(env.game.map.shape)
            self.brake_map = np.zeros(env.game.map.shape)

        self.env = env
        self.cnt = 0

        losses = None
        # keep all network structure with same depth of 4


    def collect_data(self, step_obs, step_policys, info, step_outputs, step_actions):
        assert self.num_envs == 1

        if self.config.game_name == 'hunterworld':
            if info is not None:
                self.intera_matrix += info[:]

            for idx in range(self.agent_num):
                self.comms_vis[self.agent_num - idx - 1].append(step_outputs[-(idx + 1)].asnumpy().flatten())
                reward_label = np.argwhere(info[self.agent_num - idx - 1] > 0)
                if len(reward_label) == 0:
                    self.reward_target[self.agent_num - idx - 1].append(self.agent_num - idx - 1)
                elif len(reward_label) > 0:
                    self.reward_target[self.agent_num - idx - 1].append(reward_label[0, 0])
                connect_label = np.argwhere(
                    step_obs[self.agent_num - idx - 1][:-2].reshape(24, -1)[:, 7:] > 0)

                # if len(connect_label) == 0:
                #     self.connect_target[self.agent_num - idx - 1].append(self.agent_num - idx - 1)
                # elif len(connect_label) > 0:
                #     assert not self.agent_num - idx - 1 == connect_label[0, 1]
                #     self.connect_target[self.agent_num - idx - 1].append(connect_label[0, 1])

            if self.name == 'commNet':
                self.commNet_vis.append(step_outputs[-1 - self.agent_num].asnumpy().flatten())

        else:

            if self.env.game.agents_dict.has_key(0):
                loc = self.env.game.agents_dict[0].loc
                out_of_maze = self.env.game.agents_dict[0].out_of_maze
                # make sure the car is in maze and is not initialized state
                if not out_of_maze:
                    self.cnt += 1
                    for idx in range(self.agent_num):
                        self.comms_vis[idx].append(step_outputs[-self.agent_num + idx].asnumpy().flatten())
                        self.brake_target[idx].append(step_policys[idx].asnumpy()[0, 0])

                    self.comm_norm[loc[1], loc[0]] += np.linalg.norm(step_outputs[-self.agent_num].asnumpy().flatten())
                    self.brake_map[loc[1], loc[0]] += step_policys[0].asnumpy()[0, 0]

                    if self.name == 'commNet':
                        self.commNet_vis[0].append(step_outputs[-self.agent_num - 1].asnumpy().flatten())

    def vis_step(self, step_obs, step_outputs):
        if self.name == 'VAIN' and self.env.game.vis == True:
            # step_obs[0][1:].reshape(3, 3, 5, 4)[2, 0]
            print step_outputs[-self.agent_num * 2 - 1].asnumpy()
            for idx in range(self.agent_num):
                alpha = step_outputs[-self.agent_num * 2 + idx].asnumpy().flatten()
                cnt = 0
                tmp = range(self.agent_num)
                tmp.remove(idx)
                for other_idx in tmp:
                    self.alphas[idx][other_idx] = alpha[cnt]
                    cnt += 1
            sc = plt.imshow(self.alphas, cmap='viridis')
            clb = plt.colorbar(sc)
            plt.show()
            # sys.exit(0)
        if self.name in ["AAI", 'ARMI']:
            for idx in range(self.agent_num):
                alpha = step_outputs[-self.agent_num * 2 + idx].asnumpy().flatten()
                cnt = 0
                tmp = range(self.agent_num)
                tmp.remove(idx)
                # print alpha
                # print np.mean(self.model.get_params()[0]['gate-%d_weight' %idx].asnumpy())
                for other_idx in tmp:
                    self.alphas[idx][other_idx] = alpha[cnt]
                    cnt += 1
            sc = plt.imshow(self.alphas, cmap='viridis', vmin=0)
            clb = plt.colorbar(sc)
            plt.show()
            print
        if self.name == 'AAI-supervised':
            for idx in range(self.agent_num):
                alpha = step_outputs[-self.agent_num * 3 + idx].asnumpy().flatten()
                cnt = 0
                tmp = range(self.agent_num)
                tmp.remove(idx)
                # print alpha
                # print np.mean(self.model.get_params()[0]['gate-%d_weight' %idx].asnumpy())
                for other_idx in tmp:
                    self.alphas[idx][other_idx] = alpha[cnt]
                    cnt += 1
            sc = plt.imshow(self.alphas, cmap='viridis', vmin=0, vmax=1)
            clb = plt.colorbar(sc)
            plt.show()
            print

    def vis_epoch(self):

        plt.figure(figsize=(30, 8))
        if self.config.game_name == 'hunterworld':

            print self.intera_matrix
            for i in range(self.agent_num):
                print np.asarray(self.comms_vis[i]).shape
                plt.subplot(2, 6, i + 1)
                pca1 = PCA(n_components=2)
                Y = pca1.fit_transform(np.array(self.comms_vis[i]))
                # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=10)
                # Y = tsne.fit_transform(np.array(comms_vis[i]))
                sc = plt.scatter(Y[:, 0], Y[:, 1], c=self.reward_target[i], s=3, alpha=0.8, cmap='Set2', vmin=0, vmax=4)
                clb = plt.colorbar(sc)
                plt.subplot(2, 6, i + 7)
                sc = plt.scatter(Y[:, 0], Y[:, 1], c=self.connect_target[i], s=3, alpha=0.8, cmap='Set2', vmin=0,
                                 vmax=4)
                clb = plt.colorbar(sc)
            if self.name == 'commNet':
                plt.subplot(2, 6, 6)
                pca = PCA(n_components=2)
                Y = pca.fit_transform(np.array(self.commNet_vis))
                sc = plt.scatter(Y[:, 0], Y[:, 1], c=self.connect_target[0], s=3, alpha=0.8, cmap='Set2', vmin=0,
                                 vmax=4)
                clb = plt.colorbar(sc)
            plt.subplot(2, 6, 12)
            sc = plt.imshow(self.intera_matrix, cmap='viridis')
            clb = plt.colorbar(sc)
            plt.show()

        else:

            for idx in range(self.agent_num):
                plt.subplot(2, 7, idx + 1)
                pca = PCA(n_components=2)
                print len(self.comms_vis[idx])
                Y = pca.fit_transform(np.array(self.comms_vis[idx]))
                sc = plt.scatter(Y[:, 0], Y[:, 1], c=self.brake_target[idx], s=3, alpha=0.8, cmap='cool', vmin=0.0,
                                 vmax=1.0)
            # brake map, best when 2 agent
            plt.subplot(2, 7, self.agent_num + 1)
            sc = plt.imshow(self.brake_map / self.cnt, cmap='viridis')
            clb = plt.colorbar(sc)
            plt.subplot(2, 7, self.agent_num + 2)
            sc = plt.imshow(self.comm_norm / self.cnt, cmap='viridis')
            clb = plt.colorbar(sc)

            if self.name == 'commNet':
                plt.subplot(2, 7, self.agent_num + 3)
                pca = PCA(n_components=2)
                print len(self.commNet_vis[0])
                Y = pca.fit_transform(np.array(self.commNet_vis[0]))
                sc = plt.scatter(Y[:, 0], Y[:, 1], c=self.brake_target[0], s=3, alpha=0.8, cmap='cool', vmin=0.0,
                                 vmax=1.0)
                clb = plt.colorbar(sc)

            plt.title(s=self.name)
            plt.show()
