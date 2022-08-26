#%%
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk
from raygun.jax.networks import UNet


class TestUNet():
        
    def __init__(self):
        self.ngf=3,
        self.fmap_inc_factor=2,
        self.downsample_factors=[[2, 2, 2], [2, 2, 2]]
        self.x = jnp.zeros((1, 1, 100, 80, 48))

    def _forward(self):
        transformed = hk.without_apply_rng(hk.transform(UNet))
        return transformed(self.x)
    
    def transform(self):
        # model = hk.without_apply_rng(hk.transform(self._forward()))
        model = self._forward()
        rng_key = jax.random.PRNGKey(42)
        weight = model.init(rng_key, self.x)

        y = jit(model.apply)(weight, self.x)

        assert y.shape == (1, 3, 60, 40, 8)
        
        return y
#%%
def test():
    test = TestUNet()
    return test._forward()