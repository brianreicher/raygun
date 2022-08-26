
# ORIGINALLY WRITTEN BY TRI NGUYEN (HARVARD, 2021)
# WRITTEN IN JAX BY BRIAN REICHER (NORTHEASTERN, 2022)
import jax
import haiku as hk


def get_norm_layers(net):  # TODO JAX
    # return [n for n in net.modules() if 'norm' in type(n).__name__.lower()]
    pass

def get_running_norm_stats(net):  # TODO Implement with JAX
    # means = []
    # vars = []
    # for norm in get_norm_layers(net):
    #     means.append(norm.running_mean)
    #     vars.append(norm.running_var)
    # means = torch.cat(means)
    # vars = torch.cat(vars)
    # return means, vars
    pass

def set_norm_mode(net, mode='train'):  # TODO JAX
    # if mode == 'fix_norms':
    #     net.train()
    #     for m in net.modules():
    #         if 'norm' in type(m).__name__.lower():
    #             m.eval()
    
    # if mode == 'train':
    #     net.train()

    # if mode == 'eval':
    #         net.eval()
    pass

def init_weights(net, init_type='normal', init_gain=0.02, nonlinearity='relu'):  # TODO JAX
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    # def init_func(m):  # define the initialization function
    #     classname = m.__class__.__name__
    #     if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
    #         if init_type == 'normal':
    #             hk.Transformed.init .normal_(m.weight.data, 0.0, init_gain)
    #         elif init_type == 'xavier':
    #             init.xavier_normal_(m.weight.data, gain=init_gain)
    #         elif init_type == 'kaiming':
    #             init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #         elif init_type == 'orthogonal':
    #             init.orthogonal_(m.weight.data, gain=init_gain)
    #         else:
    #             raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    #         if hasattr(m, 'bias') and m.bias is not None:
    #             init.constant_(m.bias.data, 0.0)
    #     elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    #         init.normal_(m.weight.data, 1.0, init_gain)
    #         init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    # net.apply(init_func)  # apply the initialization function <init_func>
    pass


class NoiseBlock(hk.Module):
    """Definies a block for producing and appending a feature map of gaussian noise with mean=0 and stdev=1"""

    def __init__(self):
        super().__init__()

    def __call__(self, x):  # TODO JAX tensors?
        shape = list(x.shape)
        shape[1] = 1 # only make one noise feature
        noise = jax.numpy.empty(shape).to(x.device).normal_()
        # noise = torch.empty(shape, device=x.device).normal_()
        # return torch.cat([x, noise.requires_grad_()], 1)
        return jax.numpy.concatenate(([x, noise]),1)


class ParameterizedNoiseBlock(hk.Module):
    """Definies a block for producing and appending a feature map of gaussian noise with mean and stdev defined by the first two feature maps of the incoming tensor"""

    def __init__(self):
        super().__init__()

    def __call__(self, x):  # TODO JAX tensors?
        noise = jax.random.normal(x[:,0,...], jax.nn.relu(x[:,1,...])).unsqueeze(1)
        return jax.numpy.concatenate([x, noise], 1)
