from method.base import Method
from method.common import *


class VAIN(Method):
    def data_shapes(self, batch_size):
        data_shapes = [('state', (batch_size, self.config.max_agent_num, self.config.state_dim)), ]
        return data_shapes

    def data_names(self):
        data_names = ['state']
        return data_names

    def symbol(self):

        data = mx.sym.Variable('state')
        data = mx.sym.SliceChannel(data=data, num_outputs=self.config.max_agent_num, squeeze_axis=1, name='sliced_data')

        energy_all = []
        fc1s = []
        comms = []
        for idx in range(self.config.max_agent_num):
            # fc1s
            fc = mx.sym.FullyConnected(data=data[idx], name='fc1_%d' % idx, num_hidden=self.config.VAIN_fc1_hidden,
                                       no_bias=True)
            relu = mx.sym.Activation(data=fc, name='relu1_%d' % idx, act_type="relu")
            fc1s.append(relu)
            # comms
            com_fc = mx.sym.FullyConnected(data=data[idx], name='fc1_com_%d' % idx,
                                           num_hidden=self.config.VAIN_com_hidden,
                                           no_bias=True)
            com_relu = mx.sym.Activation(data=com_fc, name='relu1_com_%d' % idx, act_type="relu")
            comms.append(com_relu)
            # energies
            energy = mx.sym.FullyConnected(data=data[idx], name='energy_%d' % idx, num_hidden=1, no_bias=True)
            energy_all.append(energy)

        concats_outputs = []
        references = []
        references_alpha = []
        references_alpha.append(
            mx.sym.BlockGrad(mx.sym.Concat(*[energy_all[i] for i in range(self.config.max_agent_num)], dim=1)))
        for idx in range(self.config.max_agent_num):
            concat_comms = mx.sym.Concat(*[comms[i] for i in range(self.config.max_agent_num) if i is not idx], dim=1,
                                         name='concat_comms_%d' % idx)  # (batch, hidden_size * max_agent_num)
            concat_comms = mx.sym.Reshape(data=concat_comms, shape=(0, len(comms) - 1, -1),
                                          name='_reshape_concat_attended_%d' % idx)
            all_energy = mx.sym.Concat(*[energy_all[i] for i in range(self.config.max_agent_num) if i is not idx],
                                       dim=1,
                                       name='_all_energies_%d' % idx)

            minus = mx.sym.broadcast_sub(all_energy, energy_all[idx])
            alpha = mx.sym.SoftmaxActivation(mx.sym.square(minus), name='_alpha_%d' % idx)
            alpha = mx.sym.Reshape(data=alpha, shape=(0, len(energy_all) - 1, 1),
                                   name='_alpha_reshape_%d' % idx)  # (batch, max_agent_num, 1)
            attened_comm = mx.sym.broadcast_mul(alpha, concat_comms,
                                                name='_weighted_attended_%d' % idx)  # (batch, max_agent_num, attend_dim)
            attened_comm = mx.sym.sum(data=attened_comm, axis=1, name='_weighted_attended_sum_%d' % idx)
            concat = mx.sym.Concat(fc1s[idx], attened_comm, name='concat_output_%d' % idx)
            # fc = mx.sym.FullyConnected(data=fc1s[idx] + attened_comm, name='fc3_%d' % idx,
            #                            num_hidden=config.VAIN_fc2_hidden, no_bias=True)
            fc = mx.sym.FullyConnected(data=concat, name='fc3_%d' % idx,
                                       num_hidden=self.config.VAIN_fc2_hidden, no_bias=True)
            relu = mx.sym.Activation(data=fc, name='relu3_%d' % idx, act_type="relu")
            concats_outputs.append(relu)
            references.append(mx.sym.BlockGrad(fc1s[idx]))
            references_alpha.append(mx.sym.BlockGrad(alpha))

        losses = actor_critic(data=concats_outputs, config=self.config)
        losses.extend(references_alpha)
        losses.extend(references)

        return losses

    def bind_network(self, syms):
        sym = mx.sym.Group(syms)
        self.model = mx.mod.Module(sym, data_names=self.data_names(), label_names=None, context=self.config.ctx)
        self.model.bind(data_shapes=self.data_shapes(self.config.num_envs), label_shapes=None,
                        inputs_need_grad=False, grad_req="write")
