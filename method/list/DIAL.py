from method.base import Method
from method.common import *


class DIAL(Method):
    def data_shapes(self, batch_size, unroll_len):
        data_shapes = [
            ('state', (batch_size, self.config.max_agent_num * unroll_len, self.config.state_dim)), ]
        data_shapes.append(('init_comms', (batch_size, self.config.max_agent_num, self.config.signal_num)))
        return data_shapes

    def data_names(self):
        data_names = ['state', 'init_comms']
        # data_names.append('agent_mask')
        return data_names

    def symbol(self):
        pass

    def symbol_unroll(self, unroll_len):
        data = mx.sym.Variable('state')
        data = mx.sym.SliceChannel(data=data, num_outputs=unroll_len, name='sliced_data')

        # mask = mx.sym.Variable('agent_mask')
        # masks = mx.sym.SliceChannel(data=mask, num_outputs=config.max_agent_num, name='mask_sliced')

        last_coms = mx.sym.Variable('init_comms')
        last_coms = mx.sym.SliceChannel(data=last_coms, num_outputs=self.config.max_agent_num, squeeze_axis=1,
                                        name='sliced_init_com')

        fc1_weight = []
        fc2_weight = []
        policy_weights = []
        value_weights = []
        value_bias = []
        comm_weights = []
        comm_bias = []
        for idx in range(self.config.max_agent_num):
            fc1_weight.append(mx.sym.Variable('fc1_%d_weight' % idx))
            fc2_weight.append(mx.sym.Variable('fc2_%d_bias' % idx))
            policy_weights.append(mx.sym.Variable('policy_%d_weight' % idx))
            value_weights.append(mx.sym.Variable('value_%d_weight' % idx))
            value_bias.append(mx.sym.Variable('value_%d_bias' % idx))
            comm_weights.append(mx.sym.Variable('comm_%d_weight' % idx))
            comm_bias.append(mx.sym.Variable('comm_%d_bias' % idx))

        log_policys = []
        out_policys = []
        values = []
        neg_entropys = []

        for t in range(unroll_len):
            states = mx.sym.SliceChannel(data=data[t], num_outputs=self.config.max_agent_num, squeeze_axis=1,
                                         name='sliced_states')
            blockgrads_comms = []
            comms_noisy = []
            references = []
            for idx in range(self.config.max_agent_num):
                # data_input = [mx.sym.broadcast_mul(states[idx], masks[idx], axis=1)]
                data_input = [states[idx]]
                data_input.extend([last_coms[id] for id in range(self.config.max_agent_num) if id is not idx])
                concat_comms = mx.sym.Concat(*data_input, name='concat1_t%d_a%d' % (t, idx))

                fc1 = mx.sym.FullyConnected(data=concat_comms, name='fc1_t%d_a%d' % (t, idx), num_hidden=256,
                                            weight=fc1_weight[idx], no_bias=True)
                fc1_relu = mx.sym.Activation(data=fc1, name='fc1_relu_t%d_a%d' % (t, idx), act_type="tanh")

                fc2 = mx.sym.FullyConnected(data=fc1_relu, name='fc2_t%d_a%d' % (t, idx),
                                            num_hidden=256, weight=fc2_weight[idx], no_bias=True)
                fc2 = mx.sym.Activation(data=fc2, name='fc2_relu_t%d_a%d' % (t, idx), act_type="tanh")
                policy_fc = mx.sym.FullyConnected(data=fc2, name='policy_fc_t%d_a%d' % (t, idx),
                                                  num_hidden=self.config.act_space,
                                                  weight=policy_weights[idx], no_bias=True)
                policy = mx.sym.SoftmaxActivation(data=policy_fc, name='policy_t%d_a%d' % (t, idx))
                policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1 - 1e-5)
                log_policy = mx.sym.log(data=policy, name='log_policy_t%d_a%d' % (t, idx))
                out_policy = mx.sym.BlockGrad(data=policy, name='out_policy_t%d_a%d' % (t, idx))
                neg_entropy = policy * log_policy
                neg_entropy = mx.sym.MakeLoss(
                    data=neg_entropy, grad_scale=self.config.entropy_wt, name='neg_entropy_t%d_a%d' % (t, idx))
                value = mx.sym.FullyConnected(data=fc2, name='value_t%d_a%d' % (t, idx), bias=value_bias[idx],
                                              weight=value_weights[idx], num_hidden=1)
                comm_fc = mx.sym.FullyConnected(data=fc2, name='com_t%d_a%d' % (t, idx), weight=comm_weights[idx],
                                                bias=comm_bias[idx], num_hidden=self.config.signal_num)
                comm_tanh = mx.sym.Activation(data=comm_fc, name='tanh_comm_t%d_a%d' % (t, idx), act_type="tanh")
                # noise = mx.sym.normal(loc=0, scale=2, shape=(config.num_envs, config.signal_num))
                # comm_noisy = comm_tanh * 10 + noise
                # comm_noisy = mx.sym.Activation(data=comm_noisy, name='sigmoid_t%d_a%d' % (t, idx), act_type="sigmoid")
                # comm_noisy = mx.sym.broadcast_mul(comm_noisy, masks[idx], axis=1)
                comms_noisy.append(comm_tanh)
                log_policys.append(log_policy)
                out_policys.append(out_policy)
                neg_entropys.append(neg_entropy)
                values.append(value)
                blockgrads_comms.append(mx.sym.BlockGrad(comm_tanh))
                references.append(mx.sym.BlockGrad(comm_fc))

            last_coms = comms_noisy[:]

        layers = []
        layers.extend(log_policys)
        layers.extend(values)
        layers.extend(out_policys)
        layers.extend(neg_entropys)
        layers.extend(blockgrads_comms)
        layers.extend(references)

        return layers

    def gen_DIAL_sym(self, unroll_len):
        syms = self.symbol_unroll(unroll_len=unroll_len)
        return (mx.sym.Group(syms), self.data_names(), None)

    def bind_network(self, syms):
        self.model = mx.mod.BucketingModule(self.gen_DIAL_sym, default_bucket_key=self.config.t_max,
                                            context=self.config.ctx)
        self.model.bind(data_shapes=self.data_shapes(self.config.num_envs, self.config.t_max), label_shapes=None,
                        inputs_need_grad=False, grad_req="write")

    def init_variant(self):
        self.last_step_coms = np.zeros((self.config.num_envs, self.config.max_agent_num, self.config.signal_num))

    def forward(self, states, comms=None, last_hidden_states=None, agent_mask=None, bucket_key=1, is_train=False):
        states = self.reshape_states(states) if is_train else states
        data = [mx.nd.array(states, ctx=self.config.ctx)]
        data.append(mx.nd.array(self.last_step_coms, ctx=self.config.ctx))
        # data.append(mx.nd.array(agent_mask, ctx=self.config.ctx))
        self.model.switch_bucket(bucket_key, self.data_shapes(batch_size=states.shape[0], unroll_len=bucket_key))
        self.model._curr_module.forward(data_batch=mx.io.DataBatch(data=data, label=None), is_train=is_train)
        return self.model.get_outputs()

    def reshape_states(self, states):
        return np.asarray(states).transpose((0, 2, 1, 3)).reshape(self.config.num_envs, -1, self.config.state_dim)

    def last_state_update(self, t, step_outputs):
        if t == 1:
            self.last_comms_buffer = self.last_step_coms[:]
        self.step_coms_np = np.zeros((self.config.max_agent_num, self.config.num_envs, self.config.signal_num))
        self.step_coms = step_outputs[self.config.max_agent_num * 4:self.config.max_agent_num * 5]

        for i in range(self.config.max_agent_num):
            self.step_coms_np[i] = self.step_coms[i].asnumpy()

        # update comms with discretise/regularise unit
        # last_step_coms = 0.5 * (np.sign(step_coms_np.transpose((1, 0, 2))) + 1)
        self.last_step_coms = np.array(self.step_coms_np.transpose((1, 0, 2)))
        # make sure only active agent has comm value
        # last_step_coms[:, max_agent_num:, :] = 0

    def calculate_grads(self, advs, env_actions, max_agent_num):
        policy_grads = []
        value_grads = []
        neg_advs_v = -np.asarray(advs).transpose((2, 1, 0))
        env_actions = env_actions.astype(int).reshape((self.config.max_agent_num, self.config.num_envs, -1)).transpose(
            (2, 0, 1))
        for t in range(self.config.t_max):
            for i in range(self.config.max_agent_num):
                neg_advs_np = np.zeros((neg_advs_v[t][i].shape[0], self.config.act_space), dtype=np.float32)
                if i < max_agent_num:
                    neg_advs_np[np.arange(neg_advs_np.shape[0]), env_actions[t][i]] = neg_advs_v[t][i][:]
                    neg_advs_np = mx.nd.array(neg_advs_np, ctx=self.config.ctx)
                    value_grad = mx.nd.array(self.config.vf_wt * neg_advs_v[t][i][:, np.newaxis], ctx=self.config.ctx)
                    policy_grads.append(neg_advs_np)
                    value_grads.append(value_grad)
                else:
                    policy_grads.append(mx.nd.array(neg_advs_np, ctx=self.config.ctx))
                    value_grads.append(mx.nd.zeros(neg_advs_v[t][i][:, np.newaxis].shape, ctx=self.config.ctx))
        return policy_grads, value_grads

    def cal_episode_values(self, envs_outputs, episode_values):
        values = envs_outputs[
                 self.config.max_agent_num * self.config.t_max: self.config.t_max * 2 * self.config.max_agent_num]
        for t in range(self.config.t_max):
            for i in range(self.config.max_agent_num):
                episode_values[i] += np.mean(values[i + t * self.config.max_agent_num].asnumpy())
