import jax
import haiku as hk
import jax.numpy as jnp
import functools
from raygun.jax.networks.utils import NoiseBlock


class ResnetGenerator2D(hk.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations, and (optionally) the injection of a feature map of random noise into the first upsampling layer.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, output_nc=1, ngf=64, norm_layer= hk.BatchNorm, use_dropout=False, n_blocks=6, padding_type='VALID', activation=jax.nn.relu, add_noise=False, n_downsampling=2):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output imagesf
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zeros | valid
            activation          -- non-linearity layer to apply (default is ReLU)
            add_noise           -- whether to append a noise feature to the data prior to upsampling layers: True | False | 'param'
            n_downsampling      -- number of times to downsample data before ResBlocks
        """
        assert(n_blocks >= 0)
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == hk.InstanceNorm
        else:
            use_bias = norm_layer == hk.InstanceNorm

        p = 0
        updown_p = 1
        padder = []
        # if padding_type.lower() == 'reflect':  # TODO parallel in JAX?
        #     padder = [hk.pad.same(3)]
        if padding_type.lower() == 'replicate':
            # padder = [hk.pad.same(3)]
            pass
        elif padding_type.lower() == 'zeros':
            # p = 3
            pass
        elif padding_type.lower() == 'valid':
            p = 'VALID'
            updown_p = 'VALID'

        model = []
        # model += padder.copy()
        model += [hk.Conv2D(ngf, kernel_shape=7, padding=p, with_bias=use_bias),
                 norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001),
                 activation]
        print('pass 1')
        
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [hk.Conv2D(output_channels=ngf * mult * 2, kernel_shape=3, stride=2, padding=updown_p, with_bias=use_bias),
                      norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001),
                      activation]
        
        print('pass 2')
        
        mult = 2 ** n_downsampling  # TODO INHERITANCE ISSUE WITH ADDING RESNET 3D Block POSITIONAL ARGUMENTS
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock2D(dim=(ngf * mult), padding_type=p, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, activation=activation)]

        print('pass 3')
        
        if add_noise == 'param':                   # add noise feature if necessary
            # model += [ParameterizedNoiseBlock()]
            pass
        elif add_noise:                   
            model += [NoiseBlock()]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [hk.Conv2DTranspose(
                                         int(ngf * mult / 2),
                                         kernel_shape=3, stride=2,
                                         padding=updown_p,
                                         with_bias=use_bias),
                       norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001),
                      activation]
        # model += padder.copy()
        model += [hk.Conv2D(output_nc, kernel_shape=7, padding=p)]
        model += [jax.nn.tanh]

        self.model = hk.Sequential(*model)

    def __call__(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock2D(hk.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, activation=jax.nn.relu):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, activation)
        self.padding_type = padding_type

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, activation=jax.nn.relu):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zeros | valid
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
            activation          -- non-linearity layer to apply (default is ReLU)
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        p = 0
        padder = []
        # if padding_type == 'reflect':  # TODO parallel in JAX?
        #     padder = [torch.nn.ReflectionPad2d(1)]
        if padding_type.upper() == 'REPLICATE':
            # padder = [hk.pad.same(1)]
            pass
        elif padding_type.upper() == 'ZEROS':
            # p = 1
            pass
        elif padding_type.upper() == 'VALID':
            p = 'VALID'
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block = []
        # conv_block += padder.copy()

        conv_block += [hk.Conv2D(dim, kernel_shape=3, padding=p, with_bias=use_bias),  norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001), activation]
        if use_dropout:
            key = jax.random.PRNGKey(22)
            conv_block += [hk.dropout(key, 0.2)]  # TODO
        
        # conv_block += padder.copy()
        conv_block += [hk.Conv2D(dim, kernel_shape=3, padding=p, with_bias=use_bias), norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001)]

        return hk.Sequential(*conv_block)

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-2] + shape

        offset = tuple(
            jax.lax.div((a - b), 2)
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def __call__(self, x):
        """Forward function (with skip connections)"""
        if self.padding_type.upper() == 'VALID': # crop for valid networks
            res = self.conv_block(x)
            out = self.crop(x, res.size()[-2:]) + res
        else:
            out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator3D(hk.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations, and (optionally) the injection of a feature map of random noise into the first upsampling layer.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, output_nc=1, ngf=64, norm_layer=hk.BatchNorm, use_dropout=False, n_blocks=6, padding_type='VALID', activation=jax.nn.relu, add_noise=False, n_downsampling=2):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zeros | valid
            activation          -- non-linearity layer to apply (default is ReLU)
            add_noise           -- whether to append a noise feature to the data prior to upsampling layers: True | False | 'param'
            n_downsampling      -- number of times to downsample data before ResBlocks
        """
        assert(n_blocks >= 0)
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == hk.InstanceNorm
        else:
            use_bias = norm_layer == hk.InstanceNorm
        
        p = 0
        updown_p = 1
        padder = []
        # if padding_type.lower() == 'reflect':  # TODO JAX parallel?
        #     padder = [torch.nn.ReflectionPad3d(3)]
        if padding_type.upper() == 'REPLICATE':
            # padder = [hk.pad.same(3)]
            pass
        elif padding_type.upper() == 'ZEROS':
            # p = 3
            pass
        elif padding_type.upper() == 'VALID':
            p = 'VALID'
            updown_p = 'VALID' # TODO 

        model = []
        model += [hk.Conv3D(ngf, kernel_shape=7, padding=p, with_bias=use_bias),
                 norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001),
                 activation]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [hk.Conv3D(output_channels=ngf * mult * 2, kernel_shape=3, stride=2, padding=updown_p, with_bias=use_bias), #TODO: Make actually use padding_type for every convolution (currently does zeros if not valid)
                      norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001),
                      activation]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock3D(dim=ngf * mult, padding_type=p, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, activation=activation)]

        if add_noise:                   
            model += [NoiseBlock()]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [hk.Conv3DTranspose(
                                         output_channels=int(ngf * mult / 2),
                                         kernel_shape=3, stride=2,
                                         padding=updown_p,
                                         with_bias=use_bias),
                      norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001),
                      activation]
        # model += padder.copy()
        model += [hk.Conv3D(output_channels=output_nc, kernel_shape=7, padding=p)]
        model += [jax.nn.tanh]

        self.model = hk.Sequential(*model)

    def __call__(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock3D(hk.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, activation=jax.nn.relu):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, activation)
        self.padding_type = padding_type

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, activation=jax.nn.relu):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zeros | valid
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
            activation          -- non-linearity layer to apply (default is ReLU)
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        p = 0
        padder = []
        # if padding_type == 'reflect':
        #     padder = [torch.nn.ReflectionPad3d(1)]
        if padding_type.upper() == 'REPLICATE':
            # padder = [hk.pad.same(1)]
            pass
        elif padding_type.upper() == 'ZEROS':
            # p = 1
            pass
        elif padding_type.upper == 'VALID':
            p = 'VALID'
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block = []
        conv_block += padder.copy()

        conv_block += [hk.Conv3D(dim, kernel_shape=3, padding=p, with_bias=use_bias), norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001), activation]
        
        print('dropout check')
        if use_dropout:  # TODO
            key = jax.random.PRNGKey(22)
            conv_block += [hk.dropout(key, 0.2)]  # TODO

        # conv_block += padder.copy()
        conv_block += [hk.Conv3D(dim, kernel_shape=3, padding=p, with_bias=use_bias), norm_layer(create_offset=True, create_scale=True, decay_rate=0.0001)]

        return hk.Sequential(*conv_block)

    def crop(self, x, shape):

        x_target_size = x.size()[:-3] + shape

        offset = tuple(
            jax.lax.div((a - b), 2)
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def __call__(self, x):
        """Forward function (with skip connections)"""
        if self.padding_type.upper() == 'VALID': # crop for valid networks
            res = self.conv_block(x)
            out = self.crop(x, res.size()[-3:]) + res
        else:
            out = x + self.conv_block(x)  # add skip connections
        return out


class ResNet(ResnetGenerator2D, ResnetGenerator3D):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations, and (optionally) the injection of a feature map of random noise into the first upsampling layer.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, ndims, **kwargs):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zeros | valid
            activation          -- non-linearity layer to apply (default is ReLU)
            add_noise           -- whether to append a noise feature to the data prior to upsampling layers: True | False | 'param'
            n_downsampling      -- number of times to downsample data before ResBlocks
        """
        if ndims == 2:
            ResnetGenerator2D.__init__(self, **kwargs)
        elif ndims == 3:            
            ResnetGenerator3D.__init__(self, **kwargs)
        else:
            raise ValueError(ndims, 'Only 2D or 3D currently implemented. Feel free to contribute more!')
