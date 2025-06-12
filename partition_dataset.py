import os
import re
from shutil import copyfile
import argparse
import math
import random


def iterate_dir(source, dest, ratio, copy_xml):
    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = [f for f in os.listdir(source)
              if re.search(r'\\.(jpg|jpeg|png|bmp)$', f, re.IGNORECASE)]

    random.shuffle(images)
    num_images = len(images)
    num_test_images = math.ceil(ratio * num_images)

    test_images = images[:num_test_images]
    train_images = images[num_test_images:]

    for filename in test_images:
        copyfile(os.path.join(source, filename), os.path.join(test_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(source, xml_filename)
            if os.path.exists(xml_path):
                copyfile(xml_path, os.path.join(test_dir, xml_filename))

    for filename in train_images:
        copyfile(os.path.join(source, filename), os.path.join(train_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(source, xml_filename)
            if os.path.exists(xml_path):
                copyfile(xml_path, os.path.join(train_dir, xml_filename))


def main():
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets")
    parser.add_argument('-i', '--imageDir', type=str, default=os.getcwd(),
                        help='Path to the folder where the image dataset is stored.')
    parser.add_argument('-o', '--outputDir', type=str, default=None,
                        help='Output folder path for train/test folders.')
    parser.add_argument('-r', '--ratio', type=float, default=0.1,
                        help='Test split ratio (default: 0.1)')
    parser.add_argument('-x', '--xml', action='store_true',
                        help='Copy associated XML files')
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    iterate_dir(args.imageDir, args.outputDir, args.ratio, args.xml)


if __name__ == '__main__':
    main()