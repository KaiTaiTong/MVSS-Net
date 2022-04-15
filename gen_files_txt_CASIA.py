import os
from pathlib import Path

# Constants
# ROOT = '/project/6003167/EECE571_2022/FakeImageDetection/datasets/CASIAv2/'
ROOT = 'CASIAv2'

num_samples = 500  # just for testing purposes

# Paths
save_txt_path = ROOT + '/files.txt'

real_image_path = ROOT + '/images/Au/'
fake_image_path = ROOT + '/images/Tp/'
mask_path = ROOT + '/masks/'
edge_path = ROOT + '/edges/'

# output file
fout = open(save_txt_path, 'w')

# processing
real_images = os.listdir(real_image_path)
fake_images = os.listdir(fake_image_path)
masks = os.listdir(mask_path)
edges = os.listdir(edge_path)

counter = 0
for i in real_images:
    if counter < num_samples:
        fout.write(os.path.join(real_image_path, i) + ' None None 0\n')
        counter += 1

counter = 0
for i in fake_images:
    gt_filename = Path(i).stem + '_gt.png'

    if counter < num_samples:
        if (gt_filename in masks and gt_filename in edges):
            fout.write(os.path.join(fake_image_path, i) + ' ' + os.path.join(mask_path, gt_filename) + ' ' + os.path.join(edge_path, gt_filename) + ' 1\n')
            counter += 1
            continue

    print("Error in handling " + i)

fout.close()