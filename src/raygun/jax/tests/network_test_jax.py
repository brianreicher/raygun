#%%
from raygun.jax.networks import *
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk
import optax
import jmp
import time
from typing import Tuple, Any, NamedTuple, Dict
from skimage import data


class Params(NamedTuple):
    weight: jnp.ndarray
    opt_state: jnp.ndarray
    loss_scale: jmp.LossScale


# replicated here to reduce dependency
class GenericJaxModel():

    def __init__(self):
        pass

    def initialize(self, rng_key, inputs, is_training):
        raise RuntimeError("Unimplemented")

    def forward(self, inputs):
        raise RuntimeError("Unimplemented")

    def train_step(self, inputs, pmapped):
        raise RuntimeError("Unimplemented")

# RIP Queen Elizabeth 9/8/22
class JAXModel(GenericJaxModel):

    def __init__(self, learning_rate = 0.5e-4):  # TODO set up for **kwargs
        super().__init__()
        self.learning_rate = learning_rate

        # to make assigning precision policy easier
        class MyModel(hk.Module):

            def __init__(self, name=None):
                super().__init__(name=name)
                # self.net = UNet(
                #     ngf=3,
                #     fmap_inc_factor=2,
                #     downsample_factors=[[2,2,2],[2,2,2],[2,2,2]]
                #     )
                # self.net = NLayerDiscriminator3D(ngf=3)
                # net = getattr(raygun.jax.networks, network_type)
                # self.net = net(net_kwargs)
                self.net = ResnetGenerator2D(ngf=3)
               
            def __call__(self, x):
                return self.net(x)

        def _forward(x):  # Temporary set of _forward()
            net = MyModel()
            return net(x)


        policy = jmp.get_policy('p=f32,c=f32,o=f32')
        hk.mixed_precision.set_policy(MyModel, policy)

        self.model = hk.without_apply_rng(hk.transform(_forward))
        self.opt = optax.adam(learning_rate)

        @jit
        def _forward(params, inputs):
            return {'affs': self.model.apply(params.weight, inputs['raw'])}

        self.forward = _forward

        @jit
        def _loss_fn(weight, raw, gt, loss_scale):
            pred_affs = self.model.apply(weight, x=raw)
            loss = optax.l2_loss(predictions=pred_affs, targets=gt)
            # loss = loss*2*mask  # optax divides loss by 2 so we mult it back
            loss_mean = loss.mean()
            return loss_scale.scale(loss_mean), (pred_affs, loss, loss_mean)

        @jit
        def _apply_optimizer(params, grads):
            updates, new_opt_state = self.opt.update(grads, params.opt_state)
            new_weight = optax.apply_updates(params.weight, updates)
            return new_weight, new_opt_state

        def _train_step(params, inputs) -> Tuple[Params, Dict[str, jnp.ndarray], Any]:

            raw, gt = inputs['raw'], inputs['gt']

            grads, (pred_affs, loss, loss_mean) = jax.grad(
                _loss_fn, has_aux=True)(params.weight, raw, gt,
                                        params.loss_scale)

            # dynamic mixed precision loss scaling
            grads = params.loss_scale.unscale(grads)
            new_weight, new_opt_state = _apply_optimizer(params, grads)

            # if any update is non-finite, skip updates
            grads_finite = jmp.all_finite(grads)
            new_loss_scale = params.loss_scale.adjust(grads_finite)
            new_weight, new_opt_state = jmp.select_tree(
                                            grads_finite,
                                            (new_weight, new_opt_state),
                                            (params.weight, params.opt_state))

            new_params = Params(new_weight, new_opt_state, new_loss_scale)
            outputs = {'affs': pred_affs, 'grad': loss}
            return new_params, outputs, loss_mean

        self.train_step = _train_step


    def initialize(self, rng_key, inputs):
        weight = self.model.init(rng_key, inputs['raw'])
        opt_state = self.opt.init(weight)
        loss_scale = jmp.NoOpLossScale()
        return Params(weight, opt_state, loss_scale)


class NetworkTestJAX():  # TODO setup for **kwargs
    
    def __init__(self, net_type = 'UNet', 
                 task=None, im='astronaut', 
                 batch_size=None, 
                 noise_factor=3, 
                 model=JAXModel(), 
                 num_epochs=15, 
                 **net_kwargs) -> None:

        self.task = task
        self.im = im
        n_devices = jax.local_device_count() 
        if batch_size is None:
            self.batch_size = 4*n_devices
        else: 
            self.batch_size = batch_size
        self.noise_factor = noise_factor
    
        self.model = model
        self.num_epochs = num_epochs
        
        # TODO
        # net_type, net_kwargs
        
        self.inputs = None
        self.model_params = None
    
    
    def im2batch(self, im):
        im = jnp.expand_dims(im, 0)
        batch = []
        for i in range(self.batch_size):
            batch.append(jnp.expand_dims(im, 0))
        return jnp.concatenate(batch)


    def data_engine(self):
        if self.task is None:
            self.inputs = {
                                    'raw': jnp.ones([self.batch_size, 1, 132, 132, 132]),
                                    'gt': jnp.zeros([self.batch_size, 3, 40, 40, 40]),
                                    # 'raw': jnp.ones([16, 1, 512, 512, 512]),
                                    # 'gt': jnp.zeros([16, 1, 512, 512, 512])
                                 }
        else:  # TODO raw/gt shape are off and creating fmaps which are too small for valid convolutions
            gt_import = getattr(data, self.im)()
            if len(gt_import.shape) > 2:  # Strips to use only one image
                gt_import = gt_import[...,0]
            gt = self.im2batch(im=jnp.asarray(gt_import))

            noise_key = jax.random.PRNGKey(22)
            noise = self.im2batch(im=jax.random.uniform(key=noise_key, shape=gt_import.shape))
            
            raw = (gt*noise) / self.noise_factor + (gt/self.noise_factor)
            
            self.inputs = {
                                            'raw': raw,
                                            'gt': gt, 
                                          }    
    
    # init model
    def init_model(self):
        if self.inputs is None:  # Create data engine if it does not exist
            self.data_engine()
            
        self.rng = jax.random.PRNGKey(42)
        self.model_params = self.model.initialize(self.rng, self.inputs)

    # test train loop
    def train(self) -> None:
        if self.model_params is None:  # Init model if not created
            self.init_model()
        for _ in range(self.num_epochs):
            t0 = time.time()
            
            self.model_params, outputs, loss = jax.jit(self.model.train_step)(self.model_params, self.inputs)

                                                                    
            print(f'Loss: {loss}, took {time.time()-t0}s')

# %%
