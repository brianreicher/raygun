#%%
from functools import partial
import numpy as np
import torch
from raygun.torch.networks import *
from skimage import data
import matplotlib.pyplot as plt
from tqdm import trange
torch.cuda.set_device(1)

# %%
class TorchTest():
    def __init__(self, 
                net=None, 
                activation=None,
                norm=None, 
                size=24, 
                seed=42, 
                noise_factor=3, 
                img='astronaut', 
                ind=31, 
                name='',
                **network_kwargs):
        
        torch.manual_seed(seed)
        if net is None:  
            if norm is None:
                norm = partial(torch.nn.InstanceNorm2d, track_running_stats=True, momentum=0.01)
            if activation is None:
                activation = torch.nn.ReLU
            self.net = torch.nn.Sequential(
                            ResNet(1, 1, 32, norm, n_blocks=4, activation=activation), 
                            torch.nn.Tanh()
                        ).to('cuda')
        else:
            self.net = net(network_kwargs)
        self.size = size
        self.mode = 'train'
        self.ind = ind
        self.name = name
        self.noise_factor = noise_factor
        # self.loss_fn = torch.nn.MSELoss() JAX
        # self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-5) JAX
        if img is not None:
            self.data = getattr(data, img)()
            if len(self.data.shape) > 2:
                self.data = self.data[...,0]
            self.data = (torch.cuda.FloatTensor(self.data).unsqueeze(0) / 255) * 2 - 1
            self.size = self.data.shape[-1]
    
    def im2batch(self, img):  # TODO JAX
        mid = self.size // 2        
        patches = []
        patches.append(torch.cuda.FloatTensor(img[:, :mid, :mid]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, mid:, :mid]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, :mid, mid:]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, mid:, mid:]).unsqueeze(0))
        return torch.cat(patches).requires_grad_()
    
    def batch2im(self, batch):  # TODO JAX
        batch = batch.detach().cpu().squeeze()
        return torch.cat((torch.cat((batch[0], batch[1])), torch.cat((batch[2], batch[3]))), axis=1)

    def get_data(self):  # TODO JAX
        if self.data is None:
            ind = torch.randint(low=0, high=200, size=(1,))[0]
            is_face = ind >= 100
            gt = torch.cuda.FloatTensor(data.lfw_subset()[ind][:self.size, :self.size]).unsqueeze(0) * 2 - 1
            
        else:
            is_face = None
            gt = self.data
                
        noise = ((torch.randperm(self.size**2, device='cuda').reshape((self.size, self.size)).unsqueeze(0) / self.size**2) * 2 - 1).requires_grad_() # should always be mean=0 var=1
        # noise = torch.rand_like(gt, device='cuda', requires_grad=True)

        img = (gt*noise) / self.noise_factor + (gt / self.noise_factor)

        return self.im2batch(img.detach()), self.im2batch(gt), is_face

    def eval(self, show=True, patches=None, gt=None):
        self.net.eval()
        patches, gt, out, is_face = self.forward(patches=patches, gt=gt)
        if show:
            self.show()
        self.set_mode()
        return self.out

    def show(self):
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(self.img, cmap='gray', vmin=-1, vmax=1)
        axs[0].set_ylabel(self.name)
        axs[0].set_title('Input')
        axs[1].imshow(self.out, cmap='gray', vmin=-1, vmax=1)
        axs[1].set_title('Output')
        axs[2].imshow(self.gt, cmap='gray', vmin=-1, vmax=1)
        axs[2].set_title('Actual')

    def forward(self, patches=None, gt=None, is_face=None):
        if patches is None or gt is None:
            patches, gt, is_face = self.get_data()
        self.img = self.batch2im(patches)
        self.gt = self.batch2im(gt)
        out = self.net(patches)
        self.out = self.batch2im(out)
        self.is_face = is_face

        return patches, gt, out, is_face

    def step(self, show=False, patches=None, gt=None):
        self.optim.zero_grad(True)
        patches, gt, out, is_face = self.forward(patches=patches, gt=gt)
        loss = self.loss_fn(out, gt)
        loss.backward()
        self.optim.step()
        if show:
            self.show()
        return loss.item()

def eval_models(data_src, models):
    outs = {}
    patches, gt, is_face = data_src.get_data()
    for name, model in models.items():
        outs[name] = model.eval(show=False, patches=patches, gt=gt)
    num = len(models.keys()) + 2
    fig, axs = plt.subplots(1, num, figsize=(5*num, 5))
    axs[0].imshow(data_src.batch2im(patches), cmap='gray', vmin=-1, vmax=1)
    axs[0].set_title('Input')
    gt = data_src.batch2im(gt)
    axs[-1].imshow(gt, cmap='gray', vmin=-1, vmax=1)
    axs[-1].set_title('Real')
    for ax, name in zip(axs[1:-1], models.keys()):
        ax.imshow(outs[name], cmap='gray', vmin=-1, vmax=1)
        mse = torch.mean((gt - outs[name])**2)
        ax.set_title(f'{name}: MSE={mse}')

#%%
model = TorchTest()
patches, gt, out, is_face = model.forward()
model.show()
model.step(True)

#%%
def training_loop(model=UNet, steps=100, show_every=200):
    losses = np.zeros((steps,))

    ticker = trange(steps)
    model = TorchTest(net=model)
    data_src = TorchTest()
    for step in ticker:
        ticker_postfix = {}
        patches, gt, is_face = data_src.get_data()            
        losses[step] = model.step((step % show_every)==0, patches=patches, gt=gt)
        ticker_postfix = losses[step]
        ticker.set_postfix(ticker_postfix)
    eval_models((data_src, model))
    plt.figure(figsize=(15,10))
    for name, loss in losses.items():
        plt.plot(loss, label=name)
    plt.title('Losses')
    plt.ylim([0,.1])
    plt.legend()