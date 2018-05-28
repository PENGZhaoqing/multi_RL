from method.list.AMP import AMP
from method.list.ARMI import ARMI
from method.list.Bidirect_LSTM import Bidirect_LSTM
from method.list.CommNet import CommNet
from method.list.DIAL import DIAL
from method.list.DIAL_rnn import DIAL_rnn
from method.list.IL import IL
from method.list.LSTM import LSTM
from method.list.SAMP import SAMP
from method.list.VAIN import VAIN


class ModelFactory(object):
    @classmethod
    def create(self, config):
        if config.method == 'AMP':
            return AMP(config)
        elif config.method == 'SAMP':
            return SAMP(config)
        elif config.method == 'ARMI':
            return ARMI(config)
        elif config.method == 'Bidirect_LSTM':
            return Bidirect_LSTM(config)
        elif config.method == 'LSTM':
            return LSTM(config)
        elif config.method == 'IL':
            return IL(config)
        elif config.method == 'DIAL':
            return DIAL(config)
        elif config.method == 'DIAL_rnn':
            return DIAL_rnn(config)
        elif config.method == 'CommNet':
            return CommNet(config)
        elif config.method == 'VAIN':
            return VAIN(config)
        else:
            raise Exception('no such method!')
