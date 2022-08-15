#%%
from functools import partial
from operator import mod
import numpy as np
import torch
from raygun.torch.networks import *
from raygun.torch.networks.ResNet import *
from skimage import data
import matplotlib.pyplot as plt
from tqdm import trange
torch.cuda.set_device(1)


class TorchBuild():
    def __init__(self, 
                net=None, 
                activation=None,
                norm=None, 
                size=24, 
                seed=42, 
                noise_factor=3, 
                img='astronaut', 
                ind=31, 
                name=''):
        
        torch.manual_seed(seed)
        if net is None:  # FIX PARAM MODE
            self.net = ResNet(2)

        else:
            self.net = net.to('cuda')
        
        self.size = size
        self.mode = 'train'
        self.ind = ind
        self.name = name
        self.noise_factor = noise_factor
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-5)
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

#%%
class TorchTrainTest():
    
    def __init__(self, net=ResNet(2), name='ResNet', **net_kwargs) -> None:
        self.model = TorchBuild(net=net)  # TODO fix **net_kwargs
        self.data_src = TorchBuild()
        self.name = name
    
    def train_network(self, steps=1000, show_every=200):
        self.losses = {}
        name = self.name + '-loss'
        self.losses[name] = np.zeros((steps,))
        ticker = trange(steps)
        for step in ticker:
            ticker_postfix = {}
            patches, gt, is_face = self.data_src.get_data()           
            self.losses[name][step] = self.model.step((step % show_every)==0, patches=patches, gt=gt)
            ticker_postfix[name] = self.losses[name][step]
            ticker.set_postfix(ticker_postfix)
            
    # TODO fix eval_models()
    def eval_models(model, name):
        outs = {}
        test = TorchBuild()
        patches, gt, is_face = test.get_data()
        outs[name] = model.eval(show=False, patches=patches, gt=gt)
        # num = len(models.keys()) + 2
        fig, axs = plt.subplots(1, figsize=(5, 5))
        axs[0].imshow(test.batch2im(patches), cmap='gray', vmin=-1, vmax=1)
        axs[0].set_title('Input')
        gt = test.batch2im(gt)
        axs[-1].imshow(gt, cmap='gray', vmin=-1, vmax=1)
        axs[-1].set_title('Real')
        for ax, name in zip(axs[1:-1]):
            ax.imshow(outs[name], cmap='gray', vmin=-1, vmax=1)
            mse = torch.mean((gt - outs[name])**2)
            ax.set_title(f'{name}: MSE={mse}')
            

    def eval_plot(self):
        plt.figure(figsize=(15,10))
        for name, loss in self.losses.items():
            plt.plot(loss, label=name)
        plt.title('Losses')
        plt.ylim([0,.1])
        plt.legend()
    # %%
