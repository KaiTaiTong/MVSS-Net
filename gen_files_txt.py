import os
import random
import argparse


def check_count_table_filled(count_table, num_train, num_valid, num_test):
    """ Helper function to check if count table has been filled """
    indicator_train = (count_table['train_fake']
                       == count_table['train_real'] == num_train)
    indicator_valid = (count_table['valid_fake']
                       == count_table['valid_real'] == num_valid)
    indicator_test = (count_table['test_fake'] ==
                      count_table['test_real'] == num_test)

    return indicator_train and indicator_valid and indicator_test


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default='../FaceForensics/', help="root dir for data files")
    parser.add_argument('--num_train', type=int, default=3000,
                        help='number of fake training samples')
    parser.add_argument('--num_valid', type=int, default=1000,
                        help='number of fake validation samples')
    parser.add_argument('--num_test', type=int, default=1000,
                        help='number of fake test samples')

    args = parser.parse_args()
    print(args)

    # Paths
    train_txt_path = args.root + '/train_files.txt'
    valid_txt_path = args.root + '/valid_files.txt'
    test_txt_path = args.root + '/test_files.txt'

    image_path = args.root + '/images/'
    mask_path = args.root + '/masks/'
    edge_path = args.root + '/edges/'

    # processing
    images = os.listdir(image_path)
    masks = os.listdir(mask_path)
    edges = os.listdir(edge_path)

    assert((args.num_train + args.num_valid + args.num_test) <= len(masks)), \
        "Sum of train, valid, test cannot exeed total fake samples: {}".format(
            len(masks))

    # output file
    train_fout = open(train_txt_path, 'w')
    valid_fout = open(valid_txt_path, 'w')
    test_fout = open(test_txt_path, 'w')

    # For now, we just randomly shuffle the orders in images dir under a uniform distribution
    count_table = {'train_fake': 0, 'train_real': 0,
                   'valid_fake': 0, 'valid_real': 0,
                   'test_fake': 0, 'test_real': 0}
    random.shuffle(images)

    for i in images:
        if (i not in masks and i not in edges):
            # real sample
            if count_table['train_real'] < args.num_train:
                train_fout.write(os.path.join(
                    image_path, i) + ' None None 0\n')
                count_table['train_real'] += 1
                continue
            if count_table['valid_real'] < args.num_valid:
                valid_fout.write(os.path.join(
                    image_path, i) + ' None None 0\n')
                count_table['valid_real'] += 1
                continue
            if count_table['test_real'] < args.num_test:
                test_fout.write(os.path.join(image_path, i) + ' None None 0\n')
                count_table['test_real'] += 1
                continue

        if (i in masks and i in edges):
            # fake sample
            if count_table['train_fake'] < args.num_train:
                train_fout.write(os.path.join(image_path, i) + ' ' + os.path.join(
                    mask_path, i) + ' ' + os.path.join(edge_path, i) + ' 1\n')
                count_table['train_fake'] += 1
                continue
            if count_table['valid_fake'] < args.num_valid:
                valid_fout.write(os.path.join(image_path, i) + ' ' + os.path.join(
                    mask_path, i) + ' ' + os.path.join(edge_path, i) + ' 1\n')
                count_table['valid_fake'] += 1
                continue
            if count_table['test_fake'] < args.num_test:
                test_fout.write(os.path.join(image_path, i) + ' ' + os.path.join(
                    mask_path, i) + ' ' + os.path.join(edge_path, i) + ' 1\n')
                count_table['test_fake'] += 1
                continue

        if check_count_table_filled(count_table, args.num_train, args.num_valid, args.num_test):
            print('{} generated ...'.format(train_txt_path))
            print('{} generated ...'.format(valid_txt_path))
            print('{} generated ...'.format(test_txt_path))
            break

    train_fout.close()
    valid_fout.close()
    test_fout.close()


if __name__ == '__main__':
    main()
