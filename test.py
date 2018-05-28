import mxnet as mx
import numpy as np

data = mx.sym.Variable('state')
data = mx.sym.SliceChannel(data=data, num_outputs=5, squeeze_axis=1, name='sliced_data')

e_weight_Ws = []
e_weight_Us = []
e_weight_vs = []
for idx in range(5):
    e_weight_Ws.append(mx.sym.Variable('a%d_energy_W_weight' % idx, shape=(10, 10)))
    e_weight_Us.append(mx.sym.Variable('a%d_energy_U_weight' % idx, shape=(10, 10)))
    e_weight_vs.append(mx.sym.Variable('a%d_energy_v_weight' % idx, shape=(10, 1)))

alphas = []
for idx in range(5):
    energy_all = []
    comms_all = []
    pre_compute = mx.sym.dot(data[idx], e_weight_Ws[idx])
    for comm_id in range(5):
        if comm_id == idx: continue
        comms_all.append(data[comm_id])
        energy = pre_compute + mx.sym.dot(data[comm_id],
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
    alpha = mx.sym.Activation(all_energy, name='a%d_alpha_1' % idx, act_type="sigmoid")  # (batch, seq_len)
    alphas.append(alpha)

loss = mx.sym.Group(alphas)
print loss.infer_shape(state=(1, 5, 10))

model = mx.mod.Module(loss, data_names=['state'], label_names=None, context=mx.cpu())
model.bind(data_shapes=[('state', (1, 5, 10)), ], label_shapes=None,
           inputs_need_grad=False, grad_req="write")
model.init_params()
model.init_optimizer()

data = np.random.random((1, 5, 10))
data = mx.nd.array(data, ctx=mx.cpu())

# target = []
# target.append(mx.nd.array([[0, 1, 0, 0]], ctx=mx.cpu()))
# target.append(mx.nd.array([[0, 1, 0, 0]], ctx=mx.cpu()))
# target.append(mx.nd.array([[0, 1, 0, 0]], ctx=mx.cpu()))
# target.append(mx.nd.array([[0, 1, 0, 0]], ctx=mx.cpu()))
# target.append(mx.nd.array([[0, 1, 0, 0]], ctx=mx.cpu()))

for i in range(10000):
    model.forward(mx.io.DataBatch(data=[data], label=None), is_train=True)
    outputs = model.get_outputs()

    print outputs[0].asnumpy()

    gradient = []
    for output in outputs:
        tmp = output.asnumpy()
        tmp[0, 1] = tmp[0, 1]-1
        gradient.append(mx.nd.array(tmp, ctx=mx.cpu(0)))

    model.backward(out_grads=gradient)
    model.update()
