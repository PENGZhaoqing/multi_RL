from method.base import Method
from method.common import *


class CommNet(Method):
    def data_shapes(self, batch_size):
        data_shapes = [('state', (batch_size, self.config.max_agent_num, self.config.state_dim)), ]
        return data_shapes

    def data_names(self):
        data_names = ['state']
        return data_names

    def symbol(self):

        fc1s = features_layer(config=self.config)

        average_hidden1 = fc1s[0]
        for i in range(1, self.config.max_agent_num):
            average_hidden1 += fc1s[i]
        average_hidden1 /= self.config.max_agent_num
        references = []
        fc2s = []

        for idx in range(self.config.max_agent_num):
            comm1 = mx.sym.FullyConnected(data=average_hidden1, name='comm1_%d' % idx,
                                          num_hidden=self.config.commNet_hidden,
                                          no_bias=True)
            hidden1 = mx.sym.FullyConnected(data=fc1s[idx], name='hidden1_%d' % idx,
                                            num_hidden=self.config.commNet_hidden,
                                            no_bias=True)
            tmp = mx.sym.Activation(data=comm1 + hidden1, name='tanh1' + str(idx),
                                    act_type="tanh")
            fc2s.append(tmp)
            references.append(mx.sym.BlockGrad(tmp))

        losses = actor_critic(data=fc2s, config=self.config)
        losses.extend(mx.sym.BlockGrad(average_hidden1))
        losses.extend(references)
        return losses

    def bind_network(self, syms):
        sym = mx.sym.Group(syms)
        self.model = mx.mod.Module(sym, data_names=self.data_names(), label_names=None, context=self.config.ctx)
        self.model.bind(data_shapes=self.data_shapes(self.config.num_envs), label_shapes=None,
                        inputs_need_grad=False, grad_req="write")
