import daisy
import pyelastix


moving_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0_unaligned90nm.n5'
moving_name = 'volumes/raw_90nm'
moving_ds = daisy.open_ds(moving_file, moving_name)
moving_im = moving_ds.to_ndarray()

fixed_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
fixed_name = 'volumes/raw_30nm'
fixed_ds = daisy.open_ds(fixed_file, fixed_name)
fixed_im = fixed_ds.to_ndarray()
