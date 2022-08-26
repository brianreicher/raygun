#%%
import os
import tempfile
from raygun.torch.systems import CycleGAN

config_path = '/n/groups/htem/users/jlr54/raygun/experiments/ieee-isbi-2022/01_cycle_gans/test_conf.json'
system = CycleGAN(config_path)
#%%
cur_dir = os.getcwd()
temp_dir = tempfile.TemporaryDirectory()
os.chdir(temp_dir.name)

print(f'Executing test in {os.getcwd()}')

#%%
batch = system.test()

# %%
os.chdir(cur_dir)
temp_dir.cleanup()