import os
from nilearn import plotting

atlas_base = './inverted/atlas/'
roi_name = 'aalFlirted_warp_flirt_flirt.nii.gz'

subfolders = os.listdir(atlas_base)

for sub in subfolders:
    filename = os.path.join(atlas_base, sub, roi_name)
    plotting.plot_roi(filename)