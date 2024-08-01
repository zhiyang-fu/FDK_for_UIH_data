# %%
import os
import numpy as np
import torch
import yaml

from Tools.torch_fdk import FDK
from Tools.imview3d import ImageSliceViewer3D
from Tools.write_dicom import write_array_to_dicom
from Tools.interp import torch_1d_interp

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['font.sans-serif'] = ['SimHei']

# %% User Parameters
scan_type = 'axial'
scan_anatomy = 'head'
preprocess_functor = 'zrebin'
# series_uid = '1.2.156.112605.66988329187441.221003011301.9.8720.379695'
series_uid = '1.2.156.112605.66988332599473.240410112317.9.1628.86557'

data_dir = os.path.join('/data/Recon/zhiyang.fu/dat_files', scan_type, scan_anatomy)
data_file = '{}_{}.dat'.format(series_uid, preprocess_functor)
geo_dir = os.path.join('/data/Recon/zhiyang.fu/geometry_files', scan_type, scan_anatomy)
geo_file = series_uid + '.yml'

# %% load geo
with open(os.path.join(geo_dir, geo_file), 'r') as f:
    geo = yaml.load(f, yaml.SafeLoader)
# commonly used variables
num_angles = geo['iViewNum']
nu, nv = geo['iDetColNum'], geo['iDetRowNum']
du  = geo['fDetGammaEqualSpace']
# %%
proj = np.reshape(np.fromfile(os.path.join(data_dir, data_file), dtype=np.float32),
               (num_angles, nv, nu))
ImageSliceViewer3D(proj)
# %% view truncation for overscans
geo['pfViewAngle'] = geo['pfViewAngle'][:2400]
proj = proj[:2400]
# %%
us = du * (torch.arange(nu) - nu/2 + 0.5)
if geo['iFocalSpotNum'] == 2:
    f = torch.tensor(np.empty_like(proj))
    for i in range(2):
        us_p = torch.tensor(geo['pfDetGamma'][i::2])
        f[i::2] = torch_1d_interp(us, us_p, torch.tensor(proj[i::2]))
else:
    us_p = torch.tensor(geo['pfDetGamma'])
    f = torch_1d_interp(us, us_p, torch.tensor(proj))

#%% save rebinned to binary data
with open(os.path.join(data_dir, data_file.replace('zrebin', 'gammarebin')), 'wb') as fid:
    fid.write(f.numpy().astype(np.float32))
# %% verify data are saved correctly
# g = np.reshape(np.fromfile(os.path.join(data_dir, data_file.replace('zrebin', 'gammarebin')), dtype=np.float32),(2400,40,936))
# %%
device= 'cuda:3'
fov = 256
A = FDK(geo=geo, nx=512, fov=fov).to(device)
# %%
img = A.inv(f.to(device))
# %%
ImageSliceViewer3D(img[0], [860,1260])
# %%
with open(os.path.join(data_dir.replace('dat_files', 'img_files'), data_file.replace('zrebin', 'torchfdk_fov256')), 'wb') as fid:
    fid.write(img[0].cpu().numpy().astype(np.float32))
# %%
dicom_dir = os.path.join('/home/dpa/mount/D03/Staff/Zhiyang.Fu/DICOM_files',
                        scan_type, scan_anatomy)
dicom_file = series_uid + '_torchfdk'
write_array_to_dicom( os.path.join(dicom_dir, dicom_file),
                     np.flip(img[0].cpu().numpy(),axis=(1,2)),
                    tilt=geo['fTiltAngle'],
                    slice_thickness=geo['fSliceThickness'],
                    mA=int(geo['fmA']),
                    kvp=geo['fkVp'], fov=fov)
