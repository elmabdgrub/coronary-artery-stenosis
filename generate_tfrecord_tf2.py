import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
from PIL import Image
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Supprimer les avertissements TensorFlow inutiles
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (
                filename,
                width,
                height,
                member.find('name').text,
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=column_name)

def class_text_to_int(row_label, label_map_dict):
    return label_map_dict[row_label]

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, group.filename), 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_image_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = group.filename.split('.')[-1].encode('utf8')  # auto-detect format

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map_dict))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    parser = argparse.ArgumentParser(description="TensorFlow XML-to-TFRecord converter (TF 2.x compatible)")
    parser.add_argument('-x', '--xml_dir', type=str, required=True, help='Directory with XML files')
    parser.add_argument('-l', '--labels_path', type=str, required=True, help='Path to .pbtxt label map')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path for output TFRecord file')
    parser.add_argument('-i', '--image_dir', type=str, default=None, help='Directory with image files')
    parser.add_argument('-c', '--csv_path', type=str, default=None, help='Optional CSV output path')
    args = parser.parse_args()

    if args.image_dir is None:
        args.image_dir = args.xml_dir

    label_map = label_map_util.load_labelmap(args.labels_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map)

    examples = xml_to_csv(args.xml_dir)
    grouped = split(examples, 'filename')

    with tf.io.TFRecordWriter(args.output_path) as writer:
        for group in grouped:
            tf_example = create_tf_example(group, args.image_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())

    print(f'Successfully created the TFRecord file: {args.output_path}')
    if args.csv_path:
        examples.to_csv(args.csv_path, index=False)
        print(f'Successfully created the CSV file: {args.csv_path}')

if __name__ == '__main__':
    main()
