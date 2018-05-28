from method.base import Method
from method.common import *


class DIAL_rnn(Method):


    def data_shapes(self, batch_size, unroll_len):
        data_shapes = [('state', (batch_size, self.config.max_agent_num * unroll_len, self.config.state_dim)), ]
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                data_shapes.append(
                    (("l%d_a%d_init_c" % (i, idx)), (batch_size, self.config.DIAL_rnn_hidden)))
                data_shapes.append(
                    (("l%d_a%d_init_h" % (i, idx)), (batch_size, self.config.DIAL_rnn_hidden)))
        data_shapes.append(('init_comms', (batch_size, self.config.max_agent_num, self.config.signal_num)))
        data_shapes.append(('agent_mask', (batch_size, self.config.max_agent_num)))
        return data_shapes

    def data_names(self):
        data_names = ['state']
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                data_names.append("l%d_a%d_init_c" % (i, idx))
                data_names.append("l%d_a%d_init_h" % (i, idx))
        data_names.append('init_comms')
        data_names.append('agent_mask')
        return data_names

    def symbol(self):
        pass

    def symbol_unroll(self, unroll_len):
        num_lstm_layer = self.config.num_layers
        max_agent_num = self.config.max_agent_num
        dropout = self.config.dropout
        signal_num = self.config.signal_num

        data = mx.sym.Variable('state')
        data = mx.sym.SliceChannel(data=data, num_outputs=unroll_len, name='sliced_data')

        mask = mx.sym.Variable('agent_mask')
        masks = mx.sym.SliceChannel(data=mask, num_outputs=self.config.max_agent_num, name='mask_sliced')

        last_coms = mx.sym.Variable('init_comms')
        last_coms = mx.sym.SliceChannel(data=last_coms, num_outputs=max_agent_num, squeeze_axis=1,
                                        name='sliced_init_com')

        log_policys = []
        out_policys = []
        values = []
        neg_entropys = []

        comm_fcs_weights = []
        comm_fcs_bias = []
        task_fcs_weights = []
        policy_weights = []
        policy_bias = []
        value_weights = []
        value_bias = []
        comm_weights = []
        comm_bias = []
        for idx in range(max_agent_num):
            comm_fcs_weights.append(mx.sym.Variable('comm_fc_%d_weight' % idx))
            comm_fcs_bias.append(mx.sym.Variable('comm_fc_%d_bias' % idx))
            task_fcs_weights.append(mx.sym.Variable('task_fc_%d_weight' % idx))
            policy_weights.append(mx.sym.Variable('policy_%d_weight' % idx))
            policy_bias.append(mx.sym.Variable('policy_%d_bias' % idx))
            value_weights.append(mx.sym.Variable('value_%d_weight' % idx))
            value_bias.append(mx.sym.Variable('value_%d_bias' % idx))
            comm_weights.append(mx.sym.Variable('comm_%d_weight' % idx))
            comm_bias.append(mx.sym.Variable('comm_%d_bias' % idx))

        param_cells = [[] for _ in range(max_agent_num)]
        last_states = [[] for _ in range(max_agent_num)]
        for idx in range(max_agent_num):
            for i in range(num_lstm_layer):
                param_cells[idx].append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_a%d_i2h_weight" % (i, idx)),
                                                  i2h_bias=mx.sym.Variable("l%d_a%d_i2h_bias" % (i, idx)),
                                                  h2h_weight=mx.sym.Variable("l%d_a%d_h2h_weight" % (i, idx)),
                                                  h2h_bias=mx.sym.Variable("l%d_a%d_h2h_bias" % (i, idx))))
                state = LSTMState(c=mx.sym.Variable("l%d_a%d_init_c" % (i, idx)),
                                  h=mx.sym.Variable("l%d_a%d_init_h" % (i, idx)))
                last_states[idx].append(state)
            assert (len(last_states[idx]) == num_lstm_layer)

        # comm_bn_means = []
        # comm_bn_vars = []
        # bn_data_means = []
        # bn_data_vars = []
        # for idx in range(max_agent_num):
        #     comm_bn_means.append(mx.sym.Variable('comm_moving_mean%d' % idx))
        #     comm_bn_vars.append(mx.sym.Variable('comm_moving_var%d' % idx))
        #     bn_data_means.append(mx.sym.Variable('data_moving_mean%d' % idx))
        #     bn_data_vars.append(mx.sym.Variable('data_moving_var%d' % idx))

        for t in range(unroll_len):
            states = mx.sym.SliceChannel(data=data[t], num_outputs=max_agent_num, squeeze_axis=1, name='sliced_states')
            blockgrads_comms = []
            comms_noisy = []
            references = []

            for idx in range(max_agent_num):

                data_input = [mx.sym.broadcast_mul(states[idx], masks[idx])]
                # data_input = [states[idx]]
                data_input.extend([last_coms[id] for id in range(self.config.max_agent_num) if id is not idx])
                concat_inputs = mx.sym.Concat(*data_input, name='concat1_t%d_a%d' % (t, idx))

                # bn_comm = mx.symbol.BatchNorm(data=mx.sym.Concat(*comm_input, name='concat_comm_t%d_a%d' % (t, idx)),
                #                               name='comm_bn_t%d_a%d' % (t, idx), fix_gamma=True,
                #                               moving_mean=comm_bn_means[idx], moving_var=comm_bn_vars[idx])
                # bn_data = mx.symbol.BatchNorm(data=states[idx], name='data_bn_t%d_a%d' % (t, idx), fix_gamma=True,
                #                               moving_mean=bn_data_means[idx], moving_var=bn_data_vars[idx])
                # bn_data = mx.sym.FullyConnected(data=bn_data, name='task_fc_t%d_a%d' % (t, idx), num_hidden=256,
                #                                 weight=task_fcs_weights[idx], no_bias=True)
                # bn_data = mx.sym.Activation(data=bn_data, name='relu_task_t%d_a%d' % (t, idx), act_type="tanh")

                fc1 = mx.sym.FullyConnected(data=concat_inputs, name='task_fc_t%d_a%d' % (t, idx), num_hidden=256,
                                            weight=task_fcs_weights[idx], no_bias=True)
                fc1_relu = mx.sym.Activation(data=fc1, name='relu_task_t%d_a%d' % (t, idx), act_type="relu")

                # stack LSTM
                for i in range(num_lstm_layer):
                    if i == 0:
                        dp_ratio = 0.
                    else:
                        dp_ratio = dropout
                    next_state = lstm_agent(self.config.DIAL_rnn_hidden,
                                            indata=fc1_relu,
                                            prev_state=last_states[idx][i],
                                            param=param_cells[idx][i],
                                            seqidx=t, layeridx=i, dropout=dp_ratio, agent_id=idx)

                    # new_h = mx.sym.broadcast_mul(1.0 - masks[idx], last_states[idx][i].h) + mx.sym.broadcast_mul(masks[idx], next_state.h)
                    # next_state = LSTMState(c=next_state.c, h=new_h)

                    hidden = next_state.h
                    last_states[idx][i] = next_state

                if dropout > 0.:
                    hidden = mx.sym.Dropout(data=hidden, p=dropout)

                policy_fc = mx.sym.FullyConnected(data=hidden, name='policy_fc_t%d_a%d' % (t, idx),
                                                  weight=policy_weights[idx], num_hidden=self.config.act_space,
                                                  no_bias=True)
                policy = mx.sym.SoftmaxActivation(data=policy_fc, name='policy_t%d_a%d' % (t, idx))
                policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1 - 1e-5)
                log_policy = mx.sym.log(data=policy, name='log_policy_t%d_a%d' % (t, idx))
                out_policy = mx.sym.BlockGrad(data=policy, name='out_policy_t%d_a%d' % (t, idx))
                neg_entropy = policy * log_policy
                neg_entropy = mx.sym.MakeLoss(
                    data=neg_entropy, grad_scale=self.config.entropy_wt, name='neg_entropy_t%d_a%d' % (t, idx))
                value = mx.sym.FullyConnected(data=hidden, name='value_t%d_a%d' % (t, idx),
                                              weight=value_weights[idx],
                                              bias=value_bias[idx], num_hidden=1)
                comm_fc = mx.sym.FullyConnected(data=hidden, name='com_t%d_a%d' % (t, idx), weight=comm_weights[idx],
                                                no_bias=True, num_hidden=signal_num)
                comm_tanh = mx.sym.Activation(data=comm_fc, name='tanh_comm_t%d_a%d' % (t, idx), act_type="tanh")
                # noise = mx.sym.normal(loc=0, scale=2, shape=(self.config.num_envs, self.config.signal_num))
                # comm_noisy = comm_tanh * 10 + noise
                # comm_noisy = mx.sym.Activation(data=comm_noisy, name='sigmoid_t%d_a%d' % (t, idx), act_type="sigmoid")
                # comm_noisy = mx.sym.broadcast_mul(comm_noisy, masks[idx], axis=1)
                comms_noisy.append(comm_tanh)
                log_policys.append(log_policy)
                out_policys.append(out_policy)
                neg_entropys.append(neg_entropy)
                values.append(value)
                blockgrads_comms.append(mx.sym.BlockGrad(comm_tanh))
                references.append(mx.sym.BlockGrad(comm_tanh))

            last_coms = comms_noisy[:]

        layers = []
        layers.extend(log_policys)
        layers.extend(values)
        layers.extend(out_policys)
        layers.extend(neg_entropys)

        for idx in range(max_agent_num):
            for i in range(num_lstm_layer):
                layers.append(mx.sym.BlockGrad(last_states[idx][i].c))
                layers.append(mx.sym.BlockGrad(last_states[idx][i].h))

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
        self.last_hidden_states = []
        for i in range(self.config.num_layers):
            for idx in range(self.config.max_agent_num):
                self.last_hidden_states.append(
                    mx.nd.zeros((self.config.num_envs, self.config.lstm_hidden), self.config.ctx))
                self.last_hidden_states.append(
                    mx.nd.zeros((self.config.num_envs, self.config.lstm_hidden), self.config.ctx))

    def forward(self, states, agent_mask=None, bucket_key=1, is_train=False):
        states = self.reshape_states(states) if is_train else states
        data = [mx.nd.array(states, ctx=self.config.ctx)]
        data.extend(self.last_hidden_states)
        data.append(mx.nd.array(self.last_step_coms, ctx=self.config.ctx))
        data.append(mx.nd.array(agent_mask, ctx=self.config.ctx))
        self.model.switch_bucket(bucket_key, self.data_shapes(batch_size=states.shape[0], unroll_len=bucket_key))
        self.model._curr_module.forward(data_batch=mx.io.DataBatch(data=data, label=None), is_train=is_train)
        return self.model.get_outputs()

    def reshape_states(self, states):
        return np.asarray(states).transpose((0, 2, 1, 3)).reshape(self.config.num_envs, -1, self.config.state_dim)

    def last_state_update(self, t, step_outputs):
        if t == 1:
            self.last_comms_buffer = self.last_step_coms[:]
            self.last_hidden_states_buffer = [self.last_hidden_states[i].copy() for i in
                                              range(len(self.last_hidden_states))]
        self.step_coms_np = np.zeros((self.config.max_agent_num, self.config.num_envs, self.config.signal_num))
        # make sure only active agent has init_c and init_h values
        tmp = step_outputs[self.config.max_agent_num * 4: self.config.max_agent_num * 6]
        last_hidden_states = []
        for i in range(len(tmp)):
            if i < self.config.max_agent_num * 2:
                last_hidden_states.append(tmp[i].copy())
            else:
                last_hidden_states.append(
                    mx.nd.zeros((self.config.num_envs, self.config.lstm_hidden), self.config.q_ctx))
        self.step_coms = step_outputs[self.config.max_agent_num * 6: self.config.max_agent_num * 7]

        for i in range(self.config.max_agent_num):
            self.step_coms_np[i] = self.step_coms[i].asnumpy()
        self.last_step_coms = np.array(self.step_coms_np.transpose((1, 0, 2)))


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
                    policy_grads.append(mx.nd.array(neg_advs_np, ctx=self.config.q_ctx))
                    value_grads.append(mx.nd.zeros(neg_advs_v[t][i][:, np.newaxis].shape, ctx=self.config.q_ctx))
        return policy_grads, value_grads

