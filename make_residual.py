import Image, ImageChops
import numpy as np

hr_flist = 'flist/hr.flist'
lr_flist = 'flist/lrX2.flist'
res_flist = 'flist/lrX2res.flist'

with open(hr_flist) as f:
    hr_filename_list = f.read().splitlines()
with open(lr_flist) as f:
    lr_filename_list = f.read().splitlines()
with open(res_flist) as f:
    res_filename_list = f.read().splitlines()

for hr_filename, lr_filename, res_filename in zip(hr_filename_list, lr_filename_list, res_filename_list):
    hr_image = Image.open(hr_filename)
    lr_image = Image.open(lr_filename)
    lr_image = lr_image.resize(hr_image.size, Image.ANTIALIAS)
    lr_image = ImageChops.subtract(hr_image, lr_image, 1, 127)
    lr_image.save(res_filename)
