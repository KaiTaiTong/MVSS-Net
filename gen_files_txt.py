import os
import random
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='../FaceForensics/', help="root dir for data files")
    parser.add_argument("--num_samples", type=int, default=1000, help="number of samples for each real and fake data")
    args = parser.parse_args()
    print(args)

    assert(0 < args.num_samples <= 5000), "Input num_samples must be in (0, 5000]"

    # Paths
    save_txt_path = args.root + '/files.txt'

    image_path = args.root + '/images/'
    mask_path = args.root + '/masks/'
    edge_path = args.root + '/edges/'

    # output file
    fout = open(save_txt_path, 'w')

    # processing
    images = os.listdir(image_path)
    masks = os.listdir(mask_path)
    edges = os.listdir(edge_path)

    # For now, we just randomly shuffle the orders in images dir under a uniform distribution
    random.shuffle(images)
    num_fake = 0
    num_real = 0

    for i in images:
        if (i not in masks and i not in edges):
            if num_real < args.num_samples:
                fout.write(os.path.join(image_path, i) + ' None None 0\n')
                num_real += 1
                continue

        if (i in masks and i in edges):
            if num_fake < args.num_samples:
                fout.write(os.path.join(image_path, i) + ' ' + os.path.join(mask_path, i) + ' ' + os.path.join(edge_path, i) + ' 1\n')
                num_fake += 1
                continue

        if (num_fake == args.num_samples) and (num_real == args.num_samples):
            print('{} generated ...'.format(save_txt_path))
            break

    fout.close()


if __name__ == '__main__':
    main()