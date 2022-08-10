import math
import numpy as  np
import jax
import haiku as hk


class ConvPass(hk.Module):
    
    def __init__(
            self,
            input_nc,
            output_nc,
            kernel_sizes,
            activation,
            padding='VALID',
            residual=False,
            norm_layer=None,
            data_format='NCDHW'):

        super().__init__()

        # if activation is not None:
        #     if isinstance(activation, str):
        #         self.activation = getattr(jax.nn, activation)
        #     else:
        #         self.activation = activation  # assume activation is a defined function
        # else:
        #     self.activation = jax.numpy.identity
        if activation is not None:
            activation = getattr(jax.nn, activation)
            
        self.residual = residual
        
        layers = []

        for i, kernel_size in enumerate(kernel_sizes):

            self.dims = len(kernel_size)

            conv = {
                2: hk.Conv2D,
                3: hk.Conv3D,
                # 4: Conv4d  # TODO
            }[self.dims]

            if data_format is None:
                in_data_format = {
                    2: 'NCHW',
                    3: 'NCDHW'
                }[self.dims]
            else:
                in_data_format = data_format

            try:
                layers.append(
                    conv(
                        output_channels=output_nc,
                        kernel_shape=kernel_size,
                        padding=padding,
                        # padding_mode=padding_mode,
                        data_format=in_data_format))
                if residual and i == 0:
                    if input_nc < output_nc: 
                        groups = input_nc
                    else: 
                        groups = output_nc
                    self.x_init_map = conv(
                                input_nc,
                                output_nc,
                                np.ones(self.dims, dtype=int),
                                padding=padding, 
                                # padding_mode=padding_mode, TODO
                                bias=False,
                                feature_group_count=groups
                                )
            except KeyError:
                raise RuntimeError("%dD convolution not implemented" % self.dims)

            if norm_layer is not None:
                layers.append(norm_layer(output_nc))
                
            if not (residual and i == (len(kernel_sizes)-1)):
                layers.append(activation)
            
            input_nc = output_nc

        self.conv_pass = hk.Sequential(layers)
        
    def crop(self, x, shape):
        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]
    
    def __call__(self, x):
        if not self.residual:
            return self.conv_pass(x)
        else:
            res = self.conv_pass(x)
            if self.padding.lower() == 'valid':
                init_x = self.crop(self.x_init_map(x), res.size()[-self.dims:])
            else:
                init_x = self.x_init_map(x)
            return self.activation(init_x + res)  
# class ConvPass(hk.Module):
    
#     def __init__(
#             self,
#             out_channels,
#             kernel_sizes,
#             activation,
#             padding='VALID',
#             data_format='NCDHW'):

#         super().__init__()

#         if activation is not None:
#             activation = getattr(jax.nn, activation)

#         layers = []

#         for kernel_size in kernel_sizes:

#             self.dims = len(kernel_size)

#             conv = {
#                 2: hk.Conv2D,
#                 3: hk.Conv3D,
#                 # 4: Conv4d  # TODO
#             }[self.dims]

#             if data_format is None:
#                 in_data_format = {
#                     2: 'NCHW',
#                     3: 'NCDHW'
#                 }[self.dims]
#             else:
#                 in_data_format = data_format

#             try:
#                 layers.append(
#                     conv(
#                         output_channels=out_channels,
#                         kernel_shape=kernel_size,
#                         padding=padding,
#                         data_format=in_data_format))
#             except KeyError:
#                 raise RuntimeError(
#                     "%dD convolution not implemented" % self.dims)

#             if activation is not None:
#                 layers.append(activation)

#         self.conv_pass = hk.Sequential(layers)

#     def __call__(self, x):

#         return self.conv_pass(x)

class ConvDownsample(hk.Module):
    
    def __init__(
                self,
                # input_nc,
                output_nc,
                kernel_sizes,
                downsample_factor,
                activation,
                padding='valid',
                # padding_mode='reflect',
                norm_layer=None,
                data_format='NCDHW'):
               
        super().__init__()

        if activation is not None:
            if isinstance(activation, str):
                self.activation = getattr(jax.nn, activation)
            else:
                self.activation = activation()  # assume activation is a defined function
        else:
            self.activation = jax.numpy.identity()
        
        self.padding = padding
        
        layers = []

        self.dims = len(kernel_sizes)
        

        conv = {
            2: hk.Conv2D,
            3: hk.Conv3D,
            # 4: Conv4d  # TODO
        }[self.dims]

        if data_format is None:
            in_data_format = {
                2: 'NCHW',
                3: 'NCDHW'
            }[self.dims]
        else:
            in_data_format = data_format

        try:
            layers.append(
                conv(
                    output_channels=output_nc,
                    kernel_shape=kernel_sizes,
                    stride=downsample_factor,
                    padding=padding,
                    # padding_mode=padding_mode,
                    data_format=in_data_format))
 
        except KeyError:
            raise RuntimeError("%dD convolution not implemented" % self.dims)

        if norm_layer is not None:
            layers.append(norm_layer(output_nc))
            
        layers.append(self.activation)
        self.conv_pass = hk.Sequential(layers)
    
    def __call__(self, x):
        return self.conv_pass(x)


class MaxDownsample(hk.Module):  # TODO: check data format type
    
    def __init__(
                self,
                downsample_factor,
                flexible=True):
        
        super().__init__()
    
        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor
        self.flexible = flexible
        
        self.down = hk.MaxPool(window_shape=downsample_factor,
                                  strides=downsample_factor,
                                  padding='VALID')
    
    def __call__(self, x):
        if self.flexible:
            try:
                return self.down(x)
            except:
                self.check_mismatch(x.size())
        else:
            self.check_mismatch(x.size())
            return self.down(x)
    
    def check_mismatch(self, size):
        for d in range(1, self.dims+1):
            if size[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d" % (
                        size,
                        self.downsample_factor,
                        self.dims - d))
       

class Upsample(hk.Module):
    
    def __init__(
            self,
            scale_factor,
            mode='transposed_conv',
            output_nc=None,
            crop_factor=None,
            next_conv_kernel_sizes=None,
            data_format='NCDHW'):

        super(Upsample, self).__init__()
        
        if crop_factor is not None:
            assert next_conv_kernel_sizes is not None, "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes
        self.dims = len(scale_factor)
        
        if mode == 'transposed_conv':
            up = {
                2: hk.Conv2DTranspose,
                3: hk.Conv3DTranspose
            }[self.dims]

            if data_format is None:
                    in_data_format = {
                    2: 'NCHW',
                    3: 'NCDHW'
                }[self.dims]
            else:
                in_data_format = data_format

            self.up = up(
                output_channels=output_nc,
                kernel_shape=scale_factor,
                stride=scale_factor,
                data_format=in_data_format)

        else:
            raise RuntimeError("Unimplemented")  # Not implemented in Haiku
        
    def crop_to_factor(self, x, factor, kernel_sizes):

        shape = x.shape
        spatial_shape = shape[-self.dims:]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes)
            for d in range(self.dims)
        )

        ns = (
            int(math.floor(float(s - c)/f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n*f + c
            for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:

            assert all((
                    (t > c) for t, c in zip(
                        target_spatial_shape,
                        convolution_crop))
                ), \
                "Feature map with shape %s is too small to ensure " \
                "translation equivariance with factor %s and following " \
                "convolutions %s" % (
                    shape,
                    factor,
                    kernel_sizes)

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.shape[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.shape, x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def __call__(self, f_left, g_out):

        g_up = self.up(g_out)

        if self.crop_factor is not None:
            g_cropped = self.crop_to_factor(
                g_up,
                self.crop_factor,
                self.next_conv_kernel_sizes)
        else:
            g_cropped = g_up

        f_cropped = self.crop(f_left, g_cropped.size()[-self.dims:])

        return jax.lax.concatenate((f_cropped, g_cropped), dimension=1)


class UNet(hk.Module):
    
    def __init__(self,
            ngf,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down=None,
            kernel_size_up=None,
            activation='relu',
            input_nc=None,
            output_nc=None,
            num_heads=1,
            constant_upsample=False,
            downsample_method='max',
            padding_type='VALID',
            residual=False,
            norm_layer=None,
            name=None
            ):
        
        super().__init__(name=name)
        self.ndims = len(downsample_factors[0])
        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.input_nc = input_nc
        self.output_nc = output_nc if output_nc else ngf
        self.residual = residual
        # if add_noise == 'param':                   # add noise feature if necessary
        #     self.noise_layer = ParameterizedNoiseBlock()
        # elif add_noise:
        #     self.noise_layer = NoiseBlock()  # TODO add utils methods
        # else:
        #     self.noise_layer = None
        
        if kernel_size_down is None:
            kernel_size_down = [[(3,)*self.ndims, (3,)*self.ndims]]*self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3,)*self.ndims, (3,)*self.ndims]]*(self.num_levels - 1)
        
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if padding_type.lower() == 'valid':
                if factor_product is None:
                    factor_product = list(factor)
                else:
                    factor_product = list(
                        f*ff
                        for f, ff in zip(factor, factor_product))
            elif padding_type.lower() == 'same':
                factor_product = None
            else:
                raise f'Invalid padding_type option: {padding_type}'
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]
        
        # Left pass
        self.l_conv = [ConvPass(input_nc
                                     if level == 0
                                     else ngf*fmap_inc_factor**(level - (downsample_method.lower() == 'max')),
                                     ngf*fmap_inc_factor**level,
                                     kernel_size_down[level],
                                     activation=activation,
                                     padding=padding_type,
                                     residual=self.residual,
                                     norm_layer=norm_layer)
                                     for level in range(self.num_levels)
                                     ]
        
        # Left downsample
        if downsample_method.lower() == 'max':
            self.l_down = [MaxDownsample(downsample_factors[level])
                                                for level in range(self.num_levels-1)]
        elif downsample_method.lower() == 'convolve':
            self.l_down = [ConvDownsample(
                                                                ngf*fmap_inc_factor**(level + 1),
                                                                kernel_size_down[level][0],
                                                                downsample_factors[level],
                                                                activation=activation,
                                                                padding=padding_type,
                                                                norm_layer=norm_layer)
                                                                for level in range(self.num_levels - 1)]
        else:
            raise RuntimeError(f'Unknown downsampling method: {downsample_method}. Please use "max" or "convolve" instead.')

        # Righthand up/crop/concatenate
        self.r_up = [[Upsample(downsample_factors[level],
                                                    mode='nearest' if constant_upsample else 'transposed_conv',
                                                    # input_nc=ngf*fmap_inc_factor**(level + 1) + (level==1 and (add_noise is not False)),
                                                    output_nc=ngf*fmap_inc_factor**(level + 1),
                                                    crop_factor=crop_factors[level],
                                                    next_conv_kernel_sizes=kernel_size_up[level])
                                                    for level in range(self.num_levels - 1)
                                            ]for _ in range(num_heads)]
        
        self.r_conv = [[ConvPass(
                                    ngf*fmap_inc_factor**level +
                                    ngf*fmap_inc_factor**(level + 1),
                                    ngf*fmap_inc_factor**level,
                                    # if output_nc is None or level != 0
                                    # else output_nc,
                                    kernel_size_up[level],
                                    activation=activation,
                                    padding=padding_type,
                                    residual=self.residual,
                                    norm_layer=norm_layer)
                                    for level in range(self.num_levels - 1)]
                                    for _ in range(self.num_heads)]
        
    def rec_forward(self, level, f_in, total_level):
    
        prefix = "    "*(total_level-1-level)
        print(prefix + "Creating U-Net layer %i" % (total_level-1-level))
        print(prefix + "f_in: " + str(f_in.shape))

        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)
        print(prefix + "f_left: " + str(f_left.shape))

        # end of recursion
        if level == 0:

            print(prefix + "bottom layer")
            fs_out = [f_left]*self.num_heads

        else:

            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in, total_level=total_level)
            print(prefix + "g_out: " + str(gs_out[0].shape))

            # up, concat, and crop
            fs_right = [
                self.r_up[h][i](f_left, gs_out[h])
                for h in range(self.num_heads)
            ]
            print(prefix + "f_right: " + str(fs_right[0].shape))

            # convolve
            fs_out = [
                self.r_conv[h][i](fs_right[h])
                for h in range(self.num_heads)
            ]

        print(prefix + "f_out: " + str(fs_out[0].shape))

        return fs_out
    
    
    def __call__(self, x):

        y = self.rec_forward(self.num_levels - 1, x, total_level=self.num_levels)

        if self.num_heads == 1:
            return y[0]

        return y