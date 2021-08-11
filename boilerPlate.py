import numpy as np
import torch
import random
import logging
from skimage import filters
from skimage.util import *

import gunpowder as gp
from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.ext import torch, tensorboardX, NoSuchModule
from gunpowder.nodes.generic_train import GenericTrain

from typing import Dict, Union, Optional

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

from skimage import filters

class BoilerPlate(gp.BatchFilter):
    def __init__(self, raw_array, mask_array, hot_array, plate_size=None, perc_hotPixels = 0.198):
        self.raw_array = raw_array
        self.mask_array = mask_array
        self.hot_array = hot_array
        self.plate_size = plate_size
        self.perc_hotPixels = perc_hotPixels

    def setup(self):
        # tell downstream nodes about the new arrays
        mask_spec = self.spec[self.raw_array].copy()
        mask_spec.dtype = bool
        self.provides(
            self.mask_array,
            mask_spec)
        self.provides(
            self.hot_array,
            self.spec[self.raw_array].copy())

    def prepare(self, request):
        self.ndims = request[self.raw_array].roi.dims()
        deps = gp.BatchRequest()
        deps[self.raw_array] = request[self.hot_array].copy() # make sure we're getting the needed data (should already be requested, but just in case)
        return deps

    def process(self, batch, request):
        # make heat mask
        mask_data, hot_data = self.boil(batch)

        # create the array spec for the new arrays
        mask_spec = batch[self.raw_array].spec.copy() #TODO: DETERMINE IF THIS IS NECESSARY
        mask_spec.roi = request[self.mask_array].roi.copy() #TODO: DETERMINE IF THIS IS NECESSARY
        mask_spec.dtype = bool
        hot_spec = batch[self.raw_array].spec.copy() #TODO: DETERMINE IF THIS IS NECESSARY
        hot_spec.roi = request[self.hot_array].roi.copy() #TODO: DETERMINE IF THIS IS NECESSARY

        # create a new batch to hold the new array
        new_batch = gp.Batch()

        # create a new array
        mask = gp.Array(mask_data, mask_spec)
        hot = gp.Array(hot_data, hot_spec)

        # store it in the batch
        new_batch[self.mask_array] = mask
        new_batch[self.hot_array] = hot

        # return the new batch
        # print(f"Batch coming from upstream: {batch}")
        # print(f"New batch going downstream: {new_batch}")
        return new_batch

    def boil(self, batch):
        raw_data = batch[self.raw_array].data

        if not isinstance(self.plate_size, type(None)):
            if len(np.array(self.plate_size).flatten()) > 1:
                self.plate_shape = self.plate_size
            else:
                self.plate_shape = [self.plate_size,]*self.ndims           
            self.pad_width = (np.array(raw_data.shape[-self.ndims:]) - np.array(self.plate_shape)) // 2
        else:
            self.plate_shape = raw_data.shape[-self.ndims:]
            self.pad_width = (0,)*self.ndims
        
        self.numPix = int(np.prod(self.plate_shape) * self.perc_hotPixels)
        mask = np.zeros_like(raw_data) > 0
        hot = raw_data.copy()
        for i, data in enumerate(raw_data):
            mask, hot = self.get_heat(data, mask, hot, i)

        return mask, hot
    
    def get_heat(self, raw_data, mask, hot, ind):
        ind = tuple([ind]) # yes, i know this is cudegy (jRho, August 2021)
        coords = self.getStratifiedCoords(self.numPix, self.plate_shape)        
        hot_pixels = np.random.choice(raw_data.flatten(), size=self.numPix)
        for i, coord in enumerate(coords):
            this_coord = tuple(np.add(coord, self.pad_width).astype(int))
            mask[ind + this_coord] = True
            hot[ind + this_coord] = hot_pixels[i]

        return mask, hot

    def getStratifiedCoords(self, numPix, shape):
        '''
        Produce a list of approx. 'numPix' random coordinate, sampled from 'shape' using startified sampling.
        '''
        ndims = len(shape)
        box_size = np.round((np.prod(shape) / numPix)**(1/ndims)).astype(np.int)
        coords = []
        box_counts = np.ceil(shape / box_size).astype(int)
        
        rands = torch.tensor(np.random.randint(0, box_size, (np.prod(box_counts), ndims)))
        
        tensors = []
        for count in box_counts:
            tensors.append(torch.tensor(range(count)))        
        
        offset_tensors = torch.meshgrid(tensors)
        
        offsets = torch.zeros((len(offset_tensors[0].flatten()), ndims))
        for i, tensor in enumerate(offset_tensors):
            offsets[:, i] = tensor.flatten()
        
        temp_coords = offsets + rands
        coords = []
        for i in range(temp_coords.shape[0]):
            include = True
            for l, lim in enumerate(shape):
                include = include and (lim > temp_coords[i, l])
            if include:
                coords.append(tuple(temp_coords[i, :]))
        return coords

class GaussBlur(gp.BatchFilter):

  def __init__(self, array, sigma):
    self.array = array
    self.sigma = sigma
    self.truncate = 4

  def prepare(self, request):

    # the requested ROI for array
    roi = request[self.array].roi

    # 1. compute the context
    context = gp.Coordinate((self.truncate,)*roi.dims()) * self.sigma

    # 2. enlarge the requested ROI by the context
    context_roi = roi.grow(context, context)
    context_roi = request[self.array].roi.intersect(context_roi) # make sure it doesn't go out of bounds

    # create a new request with our dependencies
    deps = gp.BatchRequest()
    deps[self.array] = context_roi

    # return the request
    return deps

  def process(self, batch, request):

    # 3. smooth the whole array (including the context)
    data = batch[self.array].data
    data = filters.gaussian(
      data,
      sigma=self.sigma,
      truncate=self.truncate)

    # 4. make sure to match original datatype
    # if data.dtype != batch[self.array].spec.dtype:
    #     if batch[self.array].spec.dtype == 'uint8':
    #         data = img_as_ubyte(data)
    #     elif batch[self.array].spec.dtype == 'uint16':
    #         data = img_as_uint(data)
    #     elif batch[self.array].spec.dtype == 'int16':
    #         data = img_as_int(data)
    #     elif batch[self.array].spec.dtype == 'float64':
    #         data = img_as_float(data)
    batch[self.array].spec.dtype = data.dtype
    batch[self.array].data = data

    # 5. crop the array back to the request
    batch[self.array] = batch[self.array].crop(request[self.array].roi)