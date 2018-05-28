import os
from common import *


class Method:
    def __init__(self, config):
        self.config = config
        if not os.path.exists(self.config.save_folder):
            os.makedirs(self.config.save_folder)
        symbol = self.symbol()
        self.bind_network(symbol)
        self.init_network()

    def symbol(self):
        raise NotImplementedError("Please override this method")

    def bind_network(self, syms):
        sym = mx.sym.Group(syms)
        self.model = mx.mod.Module(sym, data_names=self.data_names(), label_names=None, context=self.config.ctx)
        self.model.bind(data_shapes=self.data_shapes(self.config.num_envs), label_shapes=None,
                        inputs_need_grad=False, grad_req="write")
        print self.config.method

    def data_names(self):
        raise NotImplementedError("Please override this method")

    def data_shapes(self, batch_size, unroll_len):
        raise NotImplementedError("Please override this method")

    def init_network(self):
        self.model.init_params(self.config.init_func)
        optimizer_params = {'learning_rate': self.config.learning_rate,
                            'rescale_grad': 1.0}
        if self.config.grad_clip:
            optimizer_params['clip_gradient'] = self.config.clip_magnitude

        self.model.init_optimizer(
            kvstore='local', optimizer=self.config.update_rule,
            optimizer_params=optimizer_params)

    def init_variant(self):
        pass

    def parse_outputs(self, step_outputs):
        step_values = step_outputs[self.config.max_agent_num:self.config.max_agent_num * 2]
        step_policys = step_outputs[self.config.max_agent_num * 2:self.config.max_agent_num * 3]
        step_actions = np.zeros((self.config.max_agent_num, self.config.num_envs, 1), dtype=int)
        step_values_np = np.zeros((self.config.max_agent_num, self.config.num_envs, 1))
        for i in range(self.config.max_agent_num):
            step_policy = step_policys[i].asnumpy()
            us = np.random.uniform(size=step_policy.shape[0])[:, np.newaxis]
            step_action = (np.cumsum(step_policy, axis=1) > us).argmax(axis=1)
            step_actions[i] = step_action[:].reshape(-1, 1)
            step_values_np[i] = step_values[i].asnumpy()
        step_values_np = np.transpose(step_values_np, (1, 0, 2))
        step_actions = np.transpose(step_actions, (1, 0, 2))
        return step_values_np, step_actions

    def last_state_update(self, t, step_outputs):
        pass

    def forward(self, states, comms=None, last_hidden_states=None, agent_mask=None, bucket_key=1, is_train=False):
        states = self.reshape_states(states) if is_train else states
        data = [mx.nd.array(states, ctx=self.config.ctx)]
        self.model.reshape(self.data_shapes(batch_size=states.shape[0]))
        self.model.forward(data_batch=mx.io.DataBatch(data=data, label=None), is_train=is_train)
        return self.model.get_outputs()

    def reshape_states(self, states):
        return np.asarray(states).transpose((0, 2, 1, 3)).reshape(-1, self.config.max_agent_num, self.config.state_dim)

    def calculate_grads(self, advs, env_actions, agent_num):
        policy_grads = []
        value_grads = []
        # advs (env_num, agent_num, t_max) transform to (agent_num, env_num * t_max)
        neg_advs_v = np.transpose(-np.asarray(advs), (1, 0, 2)).reshape(self.config.max_agent_num, -1)
        env_actions = env_actions.astype(int)
        for i in range(self.config.max_agent_num):
            neg_advs_np = np.zeros((neg_advs_v[i].shape[0], self.config.act_space), dtype=np.float32)
            if i < agent_num:
                neg_advs_np[np.arange(neg_advs_np.shape[0]), env_actions[i]] = neg_advs_v[i]
                # neg_advs (env_nums, act_space)
                neg_advs = mx.nd.array(neg_advs_np, ctx=self.config.ctx)
                # value_grad (env_nums,)
                value_grad = mx.nd.array(self.config.vf_wt * neg_advs_v[i][:, np.newaxis], ctx=self.config.ctx)
                policy_grads.append(neg_advs)
                value_grads.append(value_grad)
            else:
                policy_grads.append(mx.nd.array(neg_advs_np, ctx=self.config.ctx))
                value_grads.append(mx.nd.zeros(neg_advs_v[i][:, np.newaxis].shape, ctx=self.config.ctx))
        return policy_grads, value_grads

    def gen_matrix(self, policy_grads, envs, envs_outputs):
        return policy_grads

    def cal_episode_values(self, envs_outputs, episode_values):
        values = envs_outputs[self.config.max_agent_num:2 * self.config.max_agent_num]
        for i in range(self.config.max_agent_num):
            episode_values[i] += np.mean(values[i].asnumpy())

    def load_params(self, epoch):
        self.model.load_params(self.config.save_folder + '/network-dqn_mx%04d.params' % epoch)

    def save_params(self, epoch):
        self.model.save_params(self.config.save_folder + '/network-dqn_mx%04d.params' % epoch)
