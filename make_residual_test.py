import Image, ImageChops
import numpy as np

hr_flist = 'flist/set5_predict.flist'
lr_flist = 'flist/set5_lrX2.flist'

with open(hr_flist) as f:
    hr_filename_list = f.read().splitlines()
with open(lr_flist) as f:
    lr_filename_list = f.read().splitlines()

for hr_filename, lr_filename in zip(hr_filename_list, lr_filename_list):
    hr_image = Image.open(hr_filename)
    lr_image = Image.open(lr_filename)
    lr_image = lr_image.resize(hr_image.size, Image.ANTIALIAS)
    hr_image = ImageChops.add(hr_image, lr_image, 1, -127)
    hr_image.save(hr_flist)
