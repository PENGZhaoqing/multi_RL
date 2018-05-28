from method.base import Method
from method.common import *


class AMP(Method):

    def data_shapes(self, batch_size):
        data_shapes = [('state', (batch_size, self.config.max_agent_num, self.config.state_dim)), ]
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                data_shapes.append((("l%d_a%d_init_c" % (i, idx)), (batch_size, self.config.lstm_hidden)))
                data_shapes.append((("l%d_a%d_init_h" % (i, idx)), (batch_size, self.config.lstm_hidden)))
        return data_shapes

    def data_names(self):
        data_names = ['state']
        for idx in range(self.config.max_agent_num):
            for i in range(self.config.num_layers):
                data_names.append("l%d_a%d_init_c" % (i, idx))
                data_names.append("l%d_a%d_init_h" % (i, idx))
        return data_names

    def symbol(self):
        data = mx.sym.Variable('state')
        data = mx.sym.SliceChannel(data=data, num_outputs=self.config.max_agent_num, squeeze_axis=1, name='sliced_data')

        # init
        e_weight_Ws = []
        e_weight_Us = []
        e_weight_vs = []
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

            e_weight_Ws.append(
                mx.sym.Variable('a%d_energy_W_weight' % idx, shape=(self.config.signal_num, self.config.signal_num)))
            e_weight_Us.append(
                mx.sym.Variable('a%d_energy_U_weight' % idx, shape=(self.config.signal_num, self.config.signal_num)))
            e_weight_vs.append(mx.sym.Variable('a%d_energy_v_weight' % idx, shape=(self.config.signal_num, 1)))

        references = []
        attend_reference = []

        gates_layer = []
        messages = []
        for idx in range(self.config.max_agent_num):
            i = str(idx)
            fc = mx.sym.FullyConnected(data=data[idx], name='fc1-' + i, num_hidden=self.config.signal_num, no_bias=True)
            relu = mx.sym.Activation(data=fc, name='relu1-' + i, act_type="relu")
            gate = mx.sym.FullyConnected(data=data[idx], name='gate-' + i, num_hidden=1, no_bias=True)
            gate_tanh1 = mx.sym.Activation(data=gate, name='tanh1-' + i, act_type="sigmoid")
            gates_layer.append(gate_tanh1)
            messages.append(relu)

        hidden_all = []
        for idx in range(self.config.max_agent_num):
            energy_all = []
            comms_all = []
            pre_compute = mx.sym.dot(messages[idx], e_weight_Ws[idx])
            for comm_id in range(self.config.max_agent_num):
                if comm_id == idx: continue
                if self.config.gated:
                    comms_all.append(mx.sym.broadcast_mul(messages[comm_id], gates_layer[idx]))
                else:
                    comms_all.append(messages[comm_id])
                energy = pre_compute + mx.sym.dot(messages[comm_id],
                                                  e_weight_Us[idx], name='a%d_energy_%d' % (idx, comm_id))
                energy = mx.sym.Activation(energy, act_type="tanh",
                                           name='a%d_energy_%d' % (idx, comm_id))  # (batch, state_dim)
                energy = mx.sym.dot(energy, e_weight_vs[idx], name='a%d_energy_%d' % (idx, comm_id))  # (batch, 1)
                energy_all.append(energy)

            concat_comms = mx.sym.Concat(*comms_all, dim=1,
                                         name='a%d_concat_hiddens' % idx)  # (batch, hidden_size * seq_len)
            concat_comms = mx.sym.Reshape(data=concat_comms, shape=(0, len(comms_all), -1),
                                          name='a%d_reshape_concat_attended' % idx)  # [(batch, 9L, hidden)],

            all_energy = mx.sym.Concat(*energy_all, dim=1, name='a%d_all_energy' % idx)  # (batch, seq_len)
            alpha = mx.sym.SoftmaxActivation(all_energy, name='a%d_alpha_1' % idx)  # (batch, seq_len)
            # alpha = mx.sym.Activation(all_energy, name='a%d_alpha_1' % idx, act_type="tanh")  # (batch, seq_len)
            reshape_alpha = mx.sym.Reshape(data=alpha, shape=(0, len(energy_all), 1),
                                           name='a%d_alpha_2' % idx)  # (batch, seq_len, 1)
            attened_comms = mx.sym.broadcast_mul(reshape_alpha, concat_comms,
                                                 name='a%d_weighted_attended' % idx)  # (batch, seq_len, attend_dim)
            attened_comms = mx.sym.sum(data=attened_comms, axis=1,
                                       name='a%d_weighted_attended_2' % idx)  # (batch,  attend_dim)

            for com_idx in range(1):

                if com_idx == 1:
                    input = attened_comms
                else:
                    input = mx.sym.Concat(*[messages[idx], attened_comms])

                # stack LSTM
                for i in range(self.config.num_layers):
                    if i == 0:
                        dp_ratio = 0.
                    else:
                        dp_ratio = self.config.dropout
                    next_state = lstm_agent(num_hidden=self.config.ML_lstm_hidden, indata=input,
                                            prev_state=last_states[idx][i],
                                            param=param_cells[idx][i], agent_id=idx,
                                            seqidx=com_idx, layeridx=i, dropout=dp_ratio)
                    hidden = next_state.h
                    last_states[idx][i] = next_state

                if self.config.dropout > 0.:
                    hidden = mx.sym.Dropout(data=hidden, p=self.config.dropout)

            hidden_all.append(hidden)
            attend_reference.append(mx.sym.BlockGrad(alpha))
            references.append(mx.sym.BlockGrad(hidden))

        # hidden_all = []
        # for idx in range(config.max_agent_num):
        #     hidden = mx.sym.Concat(*[data[idx], comms_layer[idx]])
        #     fc = mx.sym.FullyConnected(data=hidden, name='fc%d-%d' % (2, idx), num_hidden=config.independent_hidden1,
        #                                no_bias=True)
        #     relu = mx.sym.Activation(data=fc, name='relu%d-%d' % (2, idx), act_type="tanh")
        #     # relu = mx.sym.Dropout(data=relu, p=config.dropout)
        #     fc = mx.sym.FullyConnected(data=relu, name='fc%d-%d' % (3, idx), num_hidden=config.independent_hidden2,
        #                                no_bias=True)
        #     relu = mx.sym.Activation(data=fc, name='relu%d-%d' % (3, idx), act_type="tanh")
        #     hideen_all.append(relu)

        losses = actor_critic(data=hidden_all, config=self.config)
        losses.extend(attend_reference)
        losses.extend(references)

        return losses

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
                data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
                data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
        self.model.reshape(self.data_shapes(batch_size=states.shape[0]))
        self.model.forward(data_batch=mx.io.DataBatch(data=data, label=None), is_train=is_train)
        return self.model.get_outputs()
