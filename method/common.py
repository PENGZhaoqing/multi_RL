import mxnet as mx
from collections import namedtuple
import numpy as np
import random

mx.random.seed(0)
random.seed(0)
np.random.seed(0)

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def features_layer(config):
    data = mx.sym.Variable('state')
    layers = []
    data = mx.sym.SliceChannel(data=data, num_outputs=config.max_agent_num, squeeze_axis=1, name='sliced_data')

    for idx in range(config.max_agent_num):
        fc = mx.sym.FullyConnected(data=data[idx], name='fc1-%d' % idx, num_hidden=config.feature_hidden, no_bias=True)
        relu = mx.sym.Activation(data=fc, name='relu1-%d' % idx, act_type="relu")
        layers.append(relu)
    return layers


def actor_critic(data, config):
    log_policys = []
    out_policys = []
    values = []
    neg_entropys = []
    for idx in range(config.max_agent_num):
        i = str(idx)
        policy_fc = mx.sym.FullyConnected(data=data[idx], name='policy_fc' + i, num_hidden=config.act_space,
                                          no_bias=True)
        policy = mx.sym.SoftmaxActivation(data=policy_fc, name='policy' + i)
        policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1 - 1e-5)
        log_policy = mx.sym.log(data=policy, name='log_policy' + i)
        out_policy = mx.sym.BlockGrad(data=policy, name='out_policy' + i)
        neg_entropy = policy * log_policy
        neg_entropy = mx.sym.MakeLoss(
            data=neg_entropy, grad_scale=config.entropy_wt, name='neg_entropy' + i)
        value = mx.sym.FullyConnected(data=data[idx], name='value' + i, num_hidden=1)
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


def independent_layers(data, depth, config):
    fcs = []
    for idx in range(config.max_agent_num):
        fc = mx.sym.FullyConnected(data=data[idx], name='fc%d-%d' % (depth, idx), num_hidden=config.independent_hidden1,
                                   no_bias=True)
        relu = mx.sym.Activation(data=fc, name='relu%d-%d' % (depth, idx), act_type="relu")
        fcs.append(relu)
    return fcs


def lstm_agent(num_hidden, agent_id, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_a%d_i2h" % (seqidx, layeridx, agent_id))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_a%d_h2h" % (seqidx, layeridx, agent_id))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_a%d_slice" % (seqidx, layeridx, agent_id))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)
