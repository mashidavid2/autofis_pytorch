import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os, json
from config.tf_config import tf_config

dtype = torch.float32 if tf_config['dtype'] == 'float32' else torch.float64
minval = tf_config['minval']
maxval = tf_config['maxval']
mean = tf_config['mean']
stddev = tf_config['stddev']

def get_variable(init_type='xavier', shape=None, name=None, minval=minval, maxval=maxval, mean=mean,
                 stddev=stddev, dtype=dtype):
    if type(init_type) is str:
        init_type = init_type.lower()
    if init_type == 'tnormal':
        return torch.nn.Parameter(torch.Tensor(*shape).normal_(mean=mean, std=stddev).type(dtype), requires_grad=True)
    elif init_type == 'uniform':
        return torch.nn.Parameter(torch.Tensor(*shape).uniform_(minval, maxval).type(dtype), requires_grad=True)
    elif init_type == 'normal':
        return torch.nn.Parameter(torch.Tensor(*shape).normal_(mean=mean, std=stddev).type(dtype), requires_grad=True)
    elif init_type == 'xavier':
        std = np.sqrt(6. / np.sum(shape))
        a = np.sqrt(3.) * std
        print(name, 'initialized from:', -a, a)
        return torch.nn.Parameter(torch.Tensor(*shape).uniform_(-a, a).type(dtype), requires_grad=True)
    elif init_type == 'xavier_out':
        std = np.sqrt(3. / shape[1])
        a = np.sqrt(3.) * std
        print(name, 'initialized from:', -a, a)
        return torch.nn.Parameter(torch.Tensor(*shape).uniform_(-a, a).type(dtype), requires_grad=True)
    elif init_type == 'xavier_in':
        std = np.sqrt(3. / shape[0])
        a = np.sqrt(3.) * std
        print(name, 'initialized from:', -a, a)
        return torch.nn.Parameter(torch.Tensor(*shape).uniform_(-a, a).type(dtype), requires_grad=True)
    elif init_type == 'zero':
        return torch.nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=True)
    elif init_type == 'one':
        return torch.nn.Parameter(torch.ones(*shape, dtype=dtype), requires_grad=True)
    elif init_type == 'identity' and len(shape) == 2 and shape[0] == shape[1]:
        return torch.nn.Parameter(torch.eye(shape[0], dtype=dtype))
    elif 'int' in init_type.__class__.__name__ or 'float' in init_type.__class__.__name__:
        return torch.nn.Parameter(torch.ones(*shape, dtype=dtype) * init_type, requires_grad=True)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * torch.where(x >= 0.0, x, alpha * torch.nn.functional.elu(x))

def activate(act_type):
    if type(act_type) is str:
        act_type = act_type.lower()
    if act_type == 'sigmoid':
        return nn.Sigmoid()
    elif act_type == 'softmax':
        return nn.Softmax(dim=1)
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'tanh':
        return nn.Tanh()
    elif act_type == 'elu':
        return nn.ELU()
    elif act_type == 'selu':
        return nn.SELU()
    elif act_type == 'none':
        pass 
    else:
        pass 

def get_optimizer(opt_algo):
    opt_algo = opt_algo.lower()
    if opt_algo == 'adaldeta':
        return optim.Adadelta
    elif opt_algo == 'adagrad':
        return optim.Adagrad
    elif opt_algo == 'adam':
        return optim.Adam
    elif opt_algo == 'moment':
        return optim.SGD  # for MomentumOptimizer, use SGD with momentum parameter
    elif opt_algo == 'ftrl':
        return optim.Ftrl
    elif opt_algo == 'gd' or opt_algo == 'sgd':
        return optim.SGD
    elif opt_algo == 'padagrad':
        return optim.Adagrad  # for ProximalAdagradOptimizer, use Adagrad with proximity parameter
    elif opt_algo == 'pgd':
        return optim.SGD  # for ProximalGradientDescentOptimizer, use SGD with proximity parameter
    elif opt_algo == 'rmsprop':
        return optim.RMSprop
    else:
        return optim.SGD  # default to SGD optimizer

import torch.nn as nn

def get_loss(loss_func, pos_weight = None):
    loss_func = loss_func.lower()
    if loss_func == 'weight' or loss_func == 'weighted':
        return nn.BCEWithLogitsLoss()
        # return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_func == 'sigmoid':
        return nn.BCEWithLogitsLoss()
    elif loss_func == 'softmax':
        return nn.CrossEntropyLoss()

def check(x):
    try:
        return x is not None and x is not False and float(x) > 0
    except TypeError:
        return True

def get_l2_loss(params, variables):
    loss = None
    for p, v in zip(params, variables):
        if not type(p) is list:
            if check(p):
                if type(v) is list:
                    for _v in v:
                        if loss is None:
                            loss = p * F.mse_loss(_v, torch.zeros_like(_v))
                        else:
                            loss += p * F.mse_loss(_v, torch.zeros_like(_v))
                else:
                    if loss is None:
                        loss = p * F.mse_loss(v, torch.zeros_like(v))
                    else:
                        loss += p * F.mse_loss(v, torch.zeros_like(v))
        else:
            for _lp, _lv in zip(p, v):
                if loss is None:
                    loss = _lp * F.mse_loss(_lv, torch.zeros_like(_lv))
                else:
                    loss += _lp * F.mse_loss(_lv, torch.zeros_like(_lv))
    return loss

def normalize(norm, x, scale):
    if norm:
        return x * scale
    else:
        return x

def mul_noise(noisy, x, training=None):
    if noisy and training is not None:
        with torch.no_grad():
            noise = torch.randn_like(x) * noisy + 1
            return x * noise if training else x
    else:
        return x

def add_noise(noisy, x, training):
    if noisy:
        with torch.no_grad():
            noise = torch.randn_like(x) * noisy
            return x + noise if training else x
    else:
        return x

def create_placeholder(num_inputs, dtype=dtype, training=False):
    inputs = torch.zeros((None, num_inputs), dtype=torch.int32, name='input')
    labels = torch.zeros((None), dtype=torch.float32, name='label')
    if check(training):
        training = torch.zeros(1, dtype=dtype, name='training')
    return inputs, labels, training    

def split_data_mask(inputs, num_inputs, norm=False, real_inputs=None, num_cat=None):
    if not check(real_inputs):
        if check(norm):
            mask = np.sqrt(1. / num_inputs)
        else:
            mask = 1
        flag = norm
    else:
        inputs, mask = inputs[:, :real_inputs], inputs[:, real_inputs:]
        mask = mask.float()
        if check(norm):
            mask /= np.sqrt(num_cat + 1)
            mask_cat, mask_mul = mask[:, :num_cat], mask[:, num_cat:]
            sum_mul = torch.sum(mask_mul, dim=1, keepdim=True)
            sum_mul = torch.max(sum_mul, torch.ones_like(sum_mul))
            mask_mul /= torch.sqrt(sum_mul)
            mask = torch.cat([mask_cat, mask_mul], dim=1)
        flag = True
        num_inputs = real_inputs
    return inputs, mask, flag, num_inputs

def drop_out(training, keep_probs):
    keep_probs = torch.Tensor(keep_probs)
    if training:
        mask = torch.Tensor(keep_probs.shape).uniform_() < keep_probs
        keep_probs = keep_probs * mask
    else:
        keep_probs = keep_probs
    return keep_probs

def embedding_lookup(init, input_dim, factor, inputs, apply_mask=False, mask=None,
                     use_w=True, use_v=True, use_b=True, fm_path=None, fm_step=None,  third_order=False,order=None,
                     embedsize=None):
    xw, xv, b, xps = None, None, None, None
    if fm_path is not None and fm_step is not None:
        fm_dict = load_fm(fm_path, fm_step)
        if use_w:
            w = torch.tensor(fm_dict['w'], dtype=torch.float32, requires_grad=True)
            xw = w[inputs]
            if apply_mask:
                xw = xw * mask
        if use_v:
            v = torch.tensor(fm_dict['v'], dtype=torch.float32, requires_grad=True)
            xv = v[inputs]
            if apply_mask:
                xv = xv * mask.unsqueeze(1)
        if use_b:
            b = torch.tensor(fm_dict['b'], dtype=torch.float32, requires_grad=True)
        #TODO: deal with xps
    else:
        if use_w:
            w = get_variable(init, name='w', shape=[input_dim,])
            xw = w[inputs.long()]
            if apply_mask:
                xw = xw * mask
        if use_v:
            v = get_variable(init_type=init, name='v', shape=[input_dim, factor])
            xv = v[inputs.long()]
            if apply_mask:
                xv = xv * mask.unsqueeze(1)
        if third_order:
            third_v = get_variable(init_type=init, name='third_v', shape=[input_dim, factor])
            xps = third_v[inputs.long()]
            if apply_mask:
                xps = xps * mask.unsqueeze(1)
        if use_b:
            b = get_variable('zero', name='b', shape=[1])
    return xw, xv, b, xps

def layer_normalization(x, reduce_dim=1, out_dim=None, scale=None, bias=None):
    if isinstance(reduce_dim, int):
        reduce_dim = [reduce_dim]
    if isinstance(out_dim, int):
        out_dim = [out_dim]
    mean = x.mean(dim=reduce_dim, keepdim=True)
    var = x.var(dim=reduce_dim, keepdim=True)
    x = (x - mean) / (var + 1e-6).sqrt()
    if scale is not False:
        scale = scale if scale is not None else torch.ones(out_dim, dtype=torch.float)
    if bias is not False:
        bias = bias if bias is not None else torch.zeros(out_dim, dtype=torch.float)
    if scale is not False and bias is not False:
        return x * scale + bias
    elif scale is not False:
        return x * scale
    elif bias is not False:
        return x + bias
    else:
        return x

### look weird
def linear(xw):
    # l = torch.sum(xw, dim=1).squeeze()
    l = torch.sum(xw)
    return l

def output(x):
    if type(x) is list:
        logits = sum(x)
    else:
        logits = x
    outputs = torch.sigmoid(logits)
    return logits, outputs

def bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, node_in, batch_norm=False, layer_norm=False, training=True,
            res_conn=False):
    layer_kernels = []
    layer_biases = []
    x_prev = None
    for i in range(len(layer_sizes)):
        wi = get_variable(init, name='w_%d' % i, shape=[node_in, layer_sizes[i]])
        bi = get_variable(0, name='b_%d' % i, shape=[layer_sizes[i]])
        print(wi.shape, bi.shape)
        print(layer_acts[i], layer_keeps[i])

        h = torch.matmul(h, wi)
        if i < len(layer_sizes) - 1:
            if batch_norm:
                h = nn.BatchNorm1d(layer_sizes[i], affine=False)(h)
            elif layer_norm:
                h = layer_normalization(h, out_dim=layer_sizes[i], bias=False)
        h = h + bi
        if res_conn:
            if x_prev is None:
                x_prev = h
            elif layer_sizes[i-1] == layer_sizes[i]:
                h += x_prev
                x_prev = h
        h = F.dropout(
            activate(h, layer_acts[i]),
            layer_keeps[i],
            training=training)
        node_in = layer_sizes[i]
        layer_kernels.append(wi)
        layer_biases.append(bi)
    return h, layer_kernels, layer_biases

def load_fm(fm_path, fm_step, fm_data):
    fm_abs_path = os.path.join(
        os.path.join(
            os.path.join(
                os.path.join(
                    os.path.join(
                        os.path.join(
                            os.path.dirname(
                                os.path.dirname(
                                    os.path.abspath(__file__))),
                            'log'),
                        fm_data),
                    'FM'),
                fm_path),
            'checkpoints'),
        'model.ckpt-%d' % fm_step)
    print('load fm from', fm_abs_path)
    state_dict = torch.load(fm_abs_path, map_location='cpu')
    fm_dict = {'w': state_dict['embedding/w'],
               'v': state_dict['embedding/v'],
               'b': state_dict['embedding/b']}
    return fm_dict


# make string to title (make logs for debugging)
def string_to_title(text, n):
    text_length = len(text)
    even = len(text) / 2 == 0

    if text_length > n:
        return text

    if even:
        deco = "=" * int((n - text_length) / 2)
        return deco + str(text) + deco
    else:
        deco = "=" * int((n - text_length - 1) / 2)
        return deco + str(text) + deco + "="

# encoder for save processing step dictionary
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)

def get_dict_with_str_key(sample_dict):
    new_dict = {}
    if type(sample_dict) is not dict:
        return sample_dict

    for key in sample_dict:
        new_dict[str(key)] = get_dict_with_str_key(sample_dict[key])

    return new_dict
