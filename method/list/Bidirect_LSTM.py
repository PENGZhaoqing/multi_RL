from method.base import Method
from method.common import *


class Bidirect_LSTM(Method):
    def data_shapes(self, batch_size):
        data_shapes = [('state', (batch_size, self.config.max_agent_num, self.config.state_dim)), ]
        for i in range(self.config.num_layers):
            data_shapes.append((("forward_source_l%d_init_c" % i), (batch_size, self.config.lstm_hidden)))
            data_shapes.append((("forward_source_l%d_init_h" % i), (batch_size, self.config.lstm_hidden)))
            data_shapes.append((("backward_source_l%d_init_c" % i), (batch_size, self.config.lstm_hidden)))
            data_shapes.append((("backward_source_l%d_init_h" % i), (batch_size, self.config.lstm_hidden)))

        return data_shapes

    def data_names(self):
        data_names = ['state']
        for i in range(self.config.num_layers):
            data_names.append("forward_source_l%d_init_c" % i)
            data_names.append("forward_source_l%d_init_h" % i)
            data_names.append("backward_source_l%d_init_c" % i)
            data_names.append("backward_source_l%d_init_h" % i)
        return data_names

    def symbol(self):
        num_lstm_layer = self.config.num_layers
        max_agent_num = self.config.max_agent_num
        dropout = self.config.dropout

        value_weight = mx.sym.Variable("value_weight")
        policy_weight = mx.sym.Variable("policy_weight")
        value_bias = mx.sym.Variable("value_bias")
        concat_fc_weight = mx.sym.Variable("concat_fc_weight")

        forward_param_cells = []
        forward_last_states = []
        for i in range(num_lstm_layer):
            forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("forward_source_l%d_i2h_weight" % i),
                                                 i2h_bias=mx.sym.Variable("forward_source_l%d_i2h_bias" % i),
                                                 h2h_weight=mx.sym.Variable("forward_source_l%d_h2h_weight" % i),
                                                 h2h_bias=mx.sym.Variable("forward_source_l%d_h2h_bias" % i)))
            forward_state = LSTMState(c=mx.sym.Variable("forward_source_l%d_init_c" % i),
                                      h=mx.sym.Variable("forward_source_l%d_init_h" % i))
            forward_last_states.append(forward_state)
        assert (len(forward_last_states) == num_lstm_layer)
        backward_param_cells = []
        backward_last_states = []
        for i in range(num_lstm_layer):
            backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("backward_source_l%d_i2h_weight" % i),
                                                  i2h_bias=mx.sym.Variable("backward_source_l%d_i2h_bias" % i),
                                                  h2h_weight=mx.sym.Variable("backward_source_l%d_h2h_weight" % i),
                                                  h2h_bias=mx.sym.Variable("backward_source_l%d_h2h_bias" % i)))
            backward_state = LSTMState(c=mx.sym.Variable("backward_source_l%d_init_c" % i),
                                       h=mx.sym.Variable("backward_source_l%d_init_h" % i))
            backward_last_states.append(backward_state)
        assert (len(backward_last_states) == num_lstm_layer)

        data = mx.sym.Variable('state')
        data = mx.sym.SliceChannel(data=data, num_outputs=max_agent_num, squeeze_axis=1, name='sliced_data')

        forward_hidden_all = []
        backward_hidden_all = []
        for idx in range(max_agent_num):
            forward_hidden = data[idx]
            backward_hidden = data[max_agent_num - 1 - idx]

            # stack LSTM
            for i in range(num_lstm_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = dropout

                forward_next_state = lstm(self.config.lstm_hidden, indata=forward_hidden,
                                          prev_state=forward_last_states[i],
                                          param=forward_param_cells[i],
                                          seqidx=idx, layeridx=i, dropout=dp_ratio)
                backward_next_state = lstm(self.config.lstm_hidden, indata=backward_hidden,
                                           prev_state=backward_last_states[i],
                                           param=backward_param_cells[i],
                                           seqidx=idx, layeridx=i, dropout=dp_ratio)

                forward_hidden = forward_next_state.h
                forward_last_states[i] = forward_next_state
                backward_hidden = backward_next_state.h
                backward_last_states[i] = backward_next_state
            # decoder
            if dropout > 0.:
                forward_hidden = mx.sym.Dropout(data=forward_hidden, p=dropout)
                backward_hidden = mx.sym.Dropout(data=backward_hidden, p=dropout)

            forward_hidden_all.append(forward_hidden)
            backward_hidden_all.insert(0, backward_hidden)

        bi_hidden_all = []
        for f, b in zip(forward_hidden_all, backward_hidden_all):
            bi = mx.sym.Concat(f, b, dim=1)
            bi_hidden_all.append(bi)

        log_policys = []
        out_policys = []
        values = []
        neg_entropys = []
        for idx in range(max_agent_num):
            i = str(idx)
            concat_fc = mx.sym.FullyConnected(data=bi_hidden_all[idx], name='concat_fc_a' + i,
                                              num_hidden=self.config.lstm_fc_hidden,
                                              no_bias=True, weight=concat_fc_weight)
            concat_fc = mx.sym.Activation(data=concat_fc, name='concat_fc_a%d' % idx, act_type="relu")
            policy_fc = mx.sym.FullyConnected(data=concat_fc, name='policy_fc' + i, num_hidden=self.config.act_space,
                                              no_bias=True, weight=policy_weight)
            policy = mx.sym.SoftmaxActivation(data=policy_fc, name='policy' + i)
            policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1 - 1e-5)
            log_policy = mx.sym.log(data=policy, name='log_policy' + i)
            out_policy = mx.sym.BlockGrad(data=policy, name='out_policy' + i)
            neg_entropy = policy * log_policy
            neg_entropy = mx.sym.MakeLoss(
                data=neg_entropy, grad_scale=self.config.entropy_wt, name='neg_entropy' + i)
            value = mx.sym.FullyConnected(data=concat_fc, name='value' + i, num_hidden=1, weight=value_weight,
                                          bias=value_bias)
            log_policys.append(log_policy)
            out_policys.append(out_policy)
            neg_entropys.append(neg_entropy)
            values.append(value)
        layers = []
        layers.extend(log_policys)
        layers.extend(values)
        layers.extend(out_policys)
        layers.extend(neg_entropys)

        for i in range(max_agent_num):
            layers.extend(mx.sym.BlockGrad(bi_hidden_all[i]))
        return layers

    def forward(self, states, comms=None, last_hidden_states=None, agent_mask=None, bucket_key=1, is_train=False):
        states = self.reshape_states(states) if is_train else states
        data = [mx.nd.array(states, ctx=self.config.ctx)]
        data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
        data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
        data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
        data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
        self.model.reshape(self.data_shapes(batch_size=states.shape[0]))
        self.model.forward(data_batch=mx.io.DataBatch(data=data, label=None), is_train=is_train)
        return self.model.get_outputs()
