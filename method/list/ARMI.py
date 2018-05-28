from method.base import Method
from method.common import *


class ARMI(Method):
    def data_shapes(self, batch_size):
        data_shapes = [('state', (batch_size, self.config.max_agent_num, self.config.state_dim)), ]
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                data_shapes.append((("l%d_a%d_init_c" % (i, idx)), (batch_size, self.config.message_hidden)))
                data_shapes.append((("l%d_a%d_init_h" % (i, idx)), (batch_size, self.config.message_hidden)))
                data_shapes.append((("obs_l%d_a%d_init_c" % (i, idx)), (batch_size, self.config.lstm_hidden)))
                data_shapes.append((("obs_l%d_a%d_init_h" % (i, idx)), (batch_size, self.config.lstm_hidden)))
        return data_shapes

    def data_names(self):
        data_names = ['state']
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                data_names.append("l%d_a%d_init_c" % (i, idx))
                data_names.append("l%d_a%d_init_h" % (i, idx))
                data_names.append("obs_l%d_a%d_init_c" % (i, idx))
                data_names.append("obs_l%d_a%d_init_h" % (i, idx))
        return data_names

    def symbol(self):
        data = mx.sym.Variable('state')
        data = mx.sym.SliceChannel(data=data, num_outputs=self.config.max_agent_num, squeeze_axis=1, name='sliced_data')

        e_weight_W = mx.sym.Variable('energy_W_weight', shape=(self.config.feature_hidden, self.config.attend_size))
        e_weight_U = mx.sym.Variable('energy_U_weight', shape=(self.config.message_hidden, self.config.attend_size))
        e_weight_v = mx.sym.Variable('energy_v_bias', shape=(self.config.attend_size, 1))

        # message extractor
        messages = []
        for idx in range(self.config.max_agent_num):
            i = str(idx)
            fc = mx.sym.FullyConnected(data=data[idx], name='fc1-' + i, num_hidden=self.config.message_hidden,
                                       no_bias=True)
            relu = mx.sym.Activation(data=fc, name='relu1-' + i, act_type="relu")
            messages.append(relu)

        # parameters for recurrent messages
        param_cells = [[] for _ in range(self.config.max_agent_num)]
        last_states = [[] for _ in range(self.config.max_agent_num)]
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                param_cells[idx].append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_a%d_i2h_weight" % (i, idx)),
                                                  i2h_bias=mx.sym.Variable("l%d_a%d_i2h_bias" % (i, idx)),
                                                  h2h_weight=mx.sym.Variable("l%d_a%d_h2h_weight" % (i, idx)),
                                                  h2h_bias=mx.sym.Variable("l%d_a%d_h2h_bias" % (i, idx))))
                state = LSTMState(c=mx.sym.Variable("l%d_a%d_init_c" % (i, idx)),
                                  h=mx.sym.Variable("l%d_a%d_init_h" % (i, idx)))
                last_states[idx].append(state)
            assert (len(last_states[idx]) == self.config.num_layers)

        # parameters for observation functions
        observation_param_cells = [[] for _ in range(self.config.max_agent_num)]
        observation_last_states = [[] for _ in range(self.config.max_agent_num)]
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                observation_param_cells[idx].append(
                    LSTMParam(i2h_weight=mx.sym.Variable("obs_l%d_a%d_i2h_weight" % (i, idx)),
                              i2h_bias=mx.sym.Variable("obs_l%d_a%d_i2h_bias" % (i, idx)),
                              h2h_weight=mx.sym.Variable("obs_l%d_a%d_h2h_weight" % (i, idx)),
                              h2h_bias=mx.sym.Variable("obs_l%d_a%d_h2h_bias" % (i, idx))))
                state = LSTMState(c=mx.sym.Variable("obs_l%d_a%d_init_c" % (i, idx)),
                                  h=mx.sym.Variable("obs_l%d_a%d_init_h" % (i, idx)))
                observation_last_states[idx].append(state)
            assert (len(last_states[idx]) == self.config.num_layers)

        log_policys = []
        out_policys = []
        neg_entropys = []
        values = []
        references = []
        attend_reference = []

        for idx in range(self.config.max_agent_num):
            hidden_all = []
            com_order = [i for i in reversed(range(self.config.max_agent_num)) if not i == idx]
            for com_idx in com_order:
                # stack LSTM
                for i in range(self.config.num_layers):
                    if i == 0:
                        dp_ratio = 0.
                    else:
                        dp_ratio = self.config.dropout
                    next_state = lstm_agent(num_hidden=self.config.message_hidden, indata=messages[com_idx],
                                            prev_state=last_states[idx][i],
                                            param=param_cells[idx][i], agent_id=idx,
                                            seqidx=com_idx, layeridx=i, dropout=dp_ratio)

                    hidden = next_state.h
                    last_states[idx][i] = next_state

                if self.config.dropout > 0.:
                    hidden = mx.sym.Dropout(data=hidden, p=self.config.dropout)

                hidden_all.append(hidden)

            obs_feature = mx.sym.FullyConnected(data=data[idx], name='obs-fc-%d' % idx,
                                                num_hidden=self.config.feature_hidden,
                                                no_bias=True)
            obs_feature = mx.sym.Activation(data=obs_feature, name='obs-relu-%d' % idx, act_type="relu")

            hidden = mx.sym.Concat(hidden, obs_feature)
            next_state = lstm_agent(num_hidden=self.config.ART_lstm_hidden, indata=hidden,
                                    prev_state=observation_last_states[idx][i],
                                    param=observation_param_cells[idx][i], agent_id=idx,
                                    seqidx=idx, layeridx=self.config.num_layers, dropout=self.config.dropout)
            hidden = next_state.h

            if self.config.use_attend > 0:
                concat_hiddens = mx.sym.Concat(*hidden_all, dim=1,
                                               name='concat_hiddens')  # (batch, hidden_size * seq_len)
                concat_hiddens = mx.sym.Reshape(data=concat_hiddens, shape=(0, len(hidden_all), -1),
                                                name='_reshape_concat_attended')
                energy_all = []
                pre_compute = mx.sym.dot(obs_feature, e_weight_W)
                for com_idx in com_order:
                    h = messages[com_idx]  # (batch, attend_dim)
                    energy = pre_compute + mx.sym.dot(h, e_weight_U)  # (batch, state_dim)
                    energy = mx.sym.Activation(energy, act_type="tanh")  # (batch, state_dim)
                    energy = mx.sym.dot(energy, e_weight_v)  # (batch, 1)
                    energy_all.append(energy)

                all_energy = mx.sym.Concat(*energy_all, dim=1)  # (batch, seq_len)
                alpha = mx.sym.SoftmaxActivation(all_energy)  # (batch, seq_len)
                alpha = mx.sym.Reshape(data=alpha, shape=(0, len(energy_all), 1))  # (batch, seq_len, 1)
                attened_hiddens = mx.sym.broadcast_mul(alpha, concat_hiddens)  # (batch, seq_len, attend_dim)
                attened_hiddens = mx.sym.sum(data=attened_hiddens, axis=1)  # (batch,  attend_dim)
                hidden = mx.sym.Concat(hidden, attened_hiddens)
                attend_reference.append(mx.sym.BlockGrad(alpha))

            references.append(mx.sym.BlockGrad(hidden))

            policy_fc = mx.sym.FullyConnected(data=hidden, name='policy_fc_%d' % idx, num_hidden=self.config.act_space,
                                              no_bias=True)
            policy = mx.sym.SoftmaxActivation(data=policy_fc, name='policy_a%d' % idx)
            policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1 - 1e-5)
            log_policy = mx.sym.log(data=policy, name='log_policy_a%d' % idx)
            out_policy = mx.sym.BlockGrad(data=policy, name='out_policy_a%d' % idx)
            neg_entropy = policy * log_policy
            neg_entropy = mx.sym.MakeLoss(
                data=neg_entropy, grad_scale=self.config.entropy_wt, name='neg_entropy_a%d' % idx)
            value = mx.sym.FullyConnected(data=hidden, name='value_a%d' % idx, num_hidden=1)

            log_policys.append(log_policy)
            out_policys.append(out_policy)
            neg_entropys.append(neg_entropy)
            values.append(value)

        layers = []
        layers.extend(log_policys)
        layers.extend(values)
        layers.extend(out_policys)
        layers.extend(neg_entropys)
        layers.extend(attend_reference)
        layers.extend(references)

        return layers

    def bind_network(self, syms):
        sym = mx.sym.Group(syms)
        self.model = mx.mod.Module(sym, data_names=self.data_names(), label_names=None, context=self.config.ctx)
        self.model.bind(data_shapes=self.data_shapes(self.config.num_envs), label_shapes=None,
                        inputs_need_grad=False, grad_req="write")

    def forward(self, states, comms=None, last_hidden_states=None, agent_mask=None, bucket_key=1, is_train=False):
        states = self.reshape_states(states) if is_train else states
        data = [mx.nd.array(states, ctx=self.config.ctx)]
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                data.append(mx.nd.zeros((states.shape[0], self.config.message_hidden), ctx=self.config.ctx))
                data.append(mx.nd.zeros((states.shape[0], self.config.message_hidden), ctx=self.config.ctx))
                data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
                data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
        self.model.reshape(self.data_shapes(batch_size=states.shape[0]))
        self.model.forward(data_batch=mx.io.DataBatch(data=data, label=None), is_train=is_train)
        return self.model.get_outputs()
