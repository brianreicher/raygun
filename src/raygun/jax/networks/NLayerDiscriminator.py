import jax
import haiku as hk
import functools


class NLayerDiscriminator2D(hk.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, ngf=64, n_layers=3, norm_layer=hk.BatchNorm,
                 kw=4, downsampling_kw=None):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ngf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == hk.InstanceNorm
        # else:
        #     use_bias = norm_layer == hk.InstanceNorm

        if downsampling_kw is None:
            downsampling_kw = kw

        padw = "VALID"
        ds_kw = downsampling_kw
        sequence = [hk.Conv2D(output_channels=ngf, kernel_shape=ds_kw, stride=2, padding=padw), jax.nn.leaky_relu(0.2, True)]
        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                hk.Conv2D(output_channels=ngf * nf_mult, kernel_shape=ds_kw, stride=2, padding=padw, with_bias=True),
                norm_layer(create_scale=False, create_offset=False, decay_rate=0.999),  # TODO FIX OFFSET AND DECAY RATE
                jax.nn.leaky_relu(0.2, True)
            ]

        # nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            hk.Conv2D(output_channels=ngf * nf_mult, kernel_shape=kw, stride=1, padding=padw, with_bias=True),
            # norm_layer(ngf * nf_mult),
            norm_layer(create_scale=True, create_offset=False, decay_rate=0.999),  # TODO FIX OFFSET AND DECAY RATE
            jax.nn.leaky_relu(0.2, True)
        ]

        sequence += [hk.Conv2D(output_channels=1, kernel_shape=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = hk.Sequential(*sequence)

    @property
    def FOV(self):
        # Returns the receptive field of one output neuron for a network (written for patch discriminators)
        # See https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-region for formula derivation
        
        L = 0 # num of layers
        k = [] # [kernel width at layer l]
        s = [] # [stride at layer i]
        for layer in self.model:
            if hasattr(layer, 'kernel_shape'):
                L += 1
                k += [layer.kernel_shape[-1]]
                s += [layer.stride[-1]]
        
        r = 1
        for l in range(L-1, 0, -1):
            r = s[l]*r + (k[l] - s[l])

        return r

    def __cal__(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerDiscriminator3D(hk.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, ngf=64, n_layers=3, norm_layer=hk.BatchNorm,
                 kw=4, downsampling_kw=None,
                 ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ngf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == hk.InstanceNorm
        else:
            use_bias = norm_layer == hk.InstanceNorm

        if downsampling_kw is None:
            downsampling_kw = kw

        padw = "VALID"
        ds_kw = downsampling_kw
        sequence = [hk.Conv3D(output_channels=ngf, kernel_shape=ds_kw, stride=2, padding=padw), jax.nn.leaky_relu(0.2, True)]
        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                hk.Conv3D(output_channels=ngf * nf_mult, kernel_shape=ds_kw, stride=2, padding=padw, with_bias=True),
                norm_layer(create_scale=False, create_offset=False, decay_rate=0.999),
                jax.nn.leaky_relu(0.2, True)
            ]

        # nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            hk.Conv3D(ngf * nf_mult, kernel_shape=kw, stride=1, padding=padw, with_bias=True),
            norm_layer(create_scale=False, create_offset=False, decay_rate=0.999),
            jax.nn.leaky_relu(0.2, True)
        ]

        sequence += [hk.Conv3D(1, kernel_shape=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = hk.Sequential(*sequence)

    def __call__(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerDiscriminator(NLayerDiscriminator2D, NLayerDiscriminator3D):
    """Defines a PatchGAN discriminator"""

    def __init__(self, ndims, **kwargs):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ngf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """        
        if ndims == 2:
            NLayerDiscriminator2D.__init__(self, **kwargs)
        elif ndims == 3:            
            NLayerDiscriminator3D.__init__(self, **kwargs)
        else:
            raise ValueError(ndims, 'Only 2D or 3D currently implemented. Feel free to contribute more!')
