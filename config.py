import mxnet as mx


class Config(object):
    def __init__(self, args):

        self.dir = None
        self.folder = None

        # Default training settings
        self.ctx = mx.gpu(0)
        self.init_func = mx.init.Xavier(rnd_type='uniform', factor_type="in",
                                        magnitude=1)
        self.learning_rate = 1e-3
        self.update_rule = "adam"
        self.grad_clip = True
        self.clip_magnitude = 1

        # Default model settings
        self.hidden_size = 200
        self.gamma = 0.99
        self.lambda_ = 1.0
        self.vf_wt = 0.5  # Weight of value function term in the loss
        self.entropy_wt = 0.01  # Weight of entropy term in the loss

        self.num_envs = 32
        self.t_max = 1
        self.method = None
        self.game_name = None
        self.mode = None
        self.dynamic = False
        self.folder = None
        self.state_dim = None
        self.dir = None

        self.max_agent_num = 5
        # Init game traffic
        self.prob_start = 0.05
        self.prob_end = 0.2

        # Init game hunterworld
        self.toxin_num = 10
        self.prey_num = 20

        self.feature_hidden = 256
        self.independent_hidden1 = 256
        self.independent_hidden2 = 256
        self.act_space = None

        self.gated = False
        self.commNet_hidden = 256

        self.VAIN_fc1_hidden = 256
        self.VAIN_com_hidden = 256
        self.VAIN_fc2_hidden = 256

        self.VAINs_fc1_hidden = 256
        self.VAINs_com_hidden = 256
        self.VAINs_fc2_hidden = 256

        self.DIAL_rnn_hidden = 256

        self.DIAL_hidden = 256

        self.lstm_fc_hidden = 256
        self.lstm_hidden = 256

        self.ML_lstm_hidden = 256

        self.ART_lstm_hidden = 256
        self.message_hidden = 50
        self.use_attend = 1
        self.attend_size = self.message_hidden
        self.num_layers = 1

        self.signal_num = 256
        # self.dropout = 0.6
        self.dropout = 0.0

        # Override defaults with values from `args`.
        for arg in self.__dict__:
            if arg in args.__dict__:
                self.__setattr__(arg, args.__dict__[arg])

        self.save_folder = 'params/' + self.game_name + '/' + self.method
        self.save_log_path = 'log/' + self.game_name
        self.save_log = self.method + '_agents_' + str(self.max_agent_num) + '_dynamic_' + str(self.dynamic)
