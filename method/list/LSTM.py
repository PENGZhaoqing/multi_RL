from method.base import Method
from method.common import *

class LSTM(Method):

    def data_shapes(self,batch_size):
        data_shapes = [('state', (batch_size, self.config.max_agent_num, self.config.state_dim)), ]
        for i in range(self.config.num_layers):
            data_shapes.append((("l%d_init_c" % i), (batch_size, self.config.lstm_hidden)))
            data_shapes.append((("l%d_init_h" % i), (batch_size, self.config.lstm_hidden)))

        return data_shapes

    def data_names(self):
        data_names = ['state']
        for i in range(self.config.num_layers):
            data_names.append("l%d_init_c" % i)
            data_names.append("l%d_init_h" % i)
        return data_names

    def symbol(self):
        num_lstm_layer = self.config.num_layers
        dropout = self.config.dropout
        max_agent_num = self.config.max_agent_num

        value_weight = mx.sym.Variable("value_weight")
        policy_weight = mx.sym.Variable("policy_weight")
        value_bias = mx.sym.Variable("value_bias")
        concat_fc_weight = mx.sym.Variable("concat_fc_weight")

        param_cells = []
        last_states = []

        for i in range(num_lstm_layer):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
            state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                              h=mx.sym.Variable("l%d_init_h" % i))
            last_states.append(state)
        assert (len(last_states) == num_lstm_layer)

        data = mx.sym.Variable('state')
        data = mx.sym.SliceChannel(data=data, num_outputs=max_agent_num, squeeze_axis=1, name='sliced_data')

        hidden_all = []
        for seqidx in range(max_agent_num):
            hidden = data[seqidx]

            # stack LSTM
            for i in range(num_lstm_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = dropout
                next_state = lstm(self.config.lstm_hidden, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seqidx, layeridx=i, dropout=dp_ratio)

                hidden = next_state.h
                last_states[i] = next_state
            # decoder
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            hidden_all.append(hidden)

        log_policys = []
        out_policys = []
        values = []
        neg_entropys = []
        for idx in range(max_agent_num):
            i = str(idx)
            concat_fc = mx.sym.FullyConnected(data=hidden_all[idx], name='concat_fc_a' + i,
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
                data=neg_entropy, grad_scale=0.01, name='neg_entropy' + i)
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
        return layers


    def forward(self, states, comms=None, last_hidden_states=None, agent_mask=None, bucket_key=1, is_train=False):
        states = self.reshape_states(states) if is_train else states
        data = [mx.nd.array(states, ctx=self.config.ctx)]
        data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
        data.append(mx.nd.zeros((states.shape[0], self.config.lstm_hidden), ctx=self.config.ctx))
        self.model.reshape(self.data_shapes(batch_size=states.shape[0]))
        self.model.forward(data_batch=mx.io.DataBatch(data=data, label=None), is_train=is_train)
        return self.model.get_outputs()