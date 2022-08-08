#%%
import math
import jax
import haiku as hk
#%%
class ConvPass(hk.Module):
    
    def __init__(
            self,
            input_nc,
            ouput_nc,
            kernel_sizes,
            activation,
            padding='VALID',
            residual=False,
            padding_mode='reflect',
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
        
        self.residual = residual
        self.padding = padding
        
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
                        output_channels=ouput_nc,
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

            if norm_layers is not None:
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
    
    def forward(self, x):
        if not self.residual:
            return self.conv_pass(x)
        else:
            res = self.conv_pass(x)
            if self.padding.lower() == 'valid':
                init_x = self.crop(self.x_init_map(x), res.size()[-self.dims:])
            else:
                init_x = self.x_init_map(x)
            return self.activation(init_x + res)  
#%%
class ConvDownsample(hk.Module):
    
    def __init__(
            self,
            input_nc,
            output_nc,
            kernel_sizes,
            downsample_factor,
            activation,
            padding='valid',
            padding_mode='reflect',
            norm_layer=None):
               
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
                    output_channels=ouput_nc,
                    kernel_shape=kernel_size,
                    stride=downsample_factor,
                    padding=padding,
                    # padding_mode=padding_mode,
                    data_format=in_data_format))
 
        except KeyError:
            raise RuntimeError("%dD convolution not implemented" % self.dims)

        if norm_layers is not None:
            layers.append(norm_layer(output_nc))
            
        layers.append(self.activation)
        self.conv_pass = hk.Sequential(layers)
    
    def forward(self, x):
        return self.conv_pass(x)
#%%
class MaxDownsample(hk.Module):  # TODO: check data format type
    
    def __init__(
                self,
                downsample_factor,
                flexible=True):
        
        super().__init__()
    
        self.dims = len(dowmsample_factor)
        self.downsample_factor = downsample_factor
        self.flexible = flexible
        
        pool = hk.MaxPool
        self.down = pool(downsample_factor,
                         stride = downsample_factor)
    
    def forward(self, x):
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
        return
#%%
class Upsample(hk.module):
    
    def __init__(
            self,
            scale_factor,
            mode='transposed_conv',
            output_nc=None,
            crop_factor=None,
            next_conv_kernel_sizes=None,
            data_format=None):

        super().__init__()
        
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

    def forward(self, f_left, g_out):

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
            output_nc=None,
            num_heads=1,
            constant_upsample=False,
            downsample_method='max',
            padding_type='valid',
            residual=False,
            norm_layer=None,
            add_noise=False
            # fov=(1, 1, 1),
            # voxel_size=(1, 1, 1),
            # num_fmaps_out=None
            ):          