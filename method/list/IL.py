from method.base import Method
from method.common import *


class IL(Method):
    def data_shapes(self, batch_size):
        data_shapes = [('state', (batch_size, self.config.max_agent_num, self.config.state_dim)), ]
        return data_shapes

    def data_names(self):
        data_names = ['state']
        return data_names

    def symbol(self):
        fc1s = features_layer(config=self.config)
        fc2s = independent_layers(data=fc1s, config=self.config, depth=2)
        # fc3s = independent_layers(data=fc2s, config=config, depth=3)
        losses = actor_critic(data=fc2s, config=self.config)
        for idx in range(self.config.max_agent_num):
            losses.append(mx.sym.BlockGrad(fc2s[idx]))
        return losses
