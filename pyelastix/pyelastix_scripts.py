import daisy
import pyelastix
import matplotlib.pyplot as plt


moving_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0_unaligned90nm.n5'
moving_name = 'volumes/raw_90nm'
moving_ds = daisy.open_ds(moving_file, moving_name)
moving_im = moving_ds.to_ndarray()
print(moving_im.shape)

fixed_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
fixed_name = 'volumes/raw_30nm'
fixed_ds = daisy.open_ds(fixed_file, fixed_name)
fixed_im = fixed_ds.to_ndarray()
print(fixed_im.shape)

image_moving = moving_im.astype('float32')
image_fixed = fixed_im.astype('float32')

# Get default params and adjust
params = pyelastix.get_default_params(type='AFFINE')
params.FixedInternalImagePixelType = "float"
params.MovingInternalImagePixelType = "float"
params.ResultImagePixelType = "float"
params.NumberOfResolutions = 3
params.MaximumNumberOfIterations = 1000
print(dir(pyelastix))

# Register
aligned_image, field = pyelastix.register(image_moving, image_fixed, params, exact_params=False, verbose=1)


fig = plt.figure(1)
plt.clf()
plt.subplot(231); plt.imshow(im1)
plt.subplot(232); plt.imshow(im2)
plt.subplot(234); plt.imshow(im3)
plt.subplot(235); plt.imshow(field[0])
plt.subplot(236); plt.imshow(field[1])

if hasattr(plt, 'use'):
    plt.use().Run()  # visvis
else:
    plt.show() 