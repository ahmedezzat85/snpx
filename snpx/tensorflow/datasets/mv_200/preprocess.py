# """ Load datasets in memory

# """
from __future__ import absolute_import

import os
import argparse
import threading
from math import ceil
from datetime import datetime

import numpy as np 
import tensorflow as tf
import pandas as pd

from snpx import utils
from snpx.tensorflow.vgg_preprocessing import preprocess_image

##=======##=======##=======##
# CONSTANTS
##=======##=======##=======##
_DATASET_SIZE       = 80000
_NUM_CLASSES        = 200
_MAX_IMG_PER_TF_REC = 16000
_TRAIN_SET_SIZE     = 76000
_EVAL_SET_SIZE      = _DATASET_SIZE - _TRAIN_SET_SIZE

DATASET_DIR = os.path.dirname(__file__)

_IMAGE_TFREC_STRUCTURE = {
        'image' : tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([], tf.int64)
    }

##=======##=======##=======##=======##=======##=======##=======##
# Dataset Writer Functions (Save dataset to TFRECORD files)
##=======##=======##=======##=======##=======##=======##=======##
def _int64_feature(value):
    val = value if isinstance(value, list) else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class JpegDecoder(object):
    """ Decode JPG image from file.
    """
    def __init__(self):
        self.tf_sess = tf.Session()
        self._jpeg_img = tf.placeholder(dtype=tf.string)
        self._jpeg_dec = tf.image.decode_jpeg(self._jpeg_img, channels=3)

    def __call__(self, jpeg_image_file):
        try:
            with tf.gfile.FastGFile(jpeg_image_file, mode='rb') as fp:
                jpeg_image = fp.read()
            
            image = self.tf_sess.run(self._jpeg_dec, feed_dict={self._jpeg_img: jpeg_image})
            h, w, c = image.shape
            return jpeg_image, h, w, c
        except:
            print ('EXCEPTION --> ', jpeg_image_file)
    
    def close(self):
        self.tf_sess.close()

class TFRecFile(object):
    """ """
    def __init__(self, tf_rec_out_file):
        self.writer   = tf.python_io.TFRecordWriter(tf_rec_out_file)

    def add_image(self, image_file, label):
        with tf.gfile.FastGFile(image_file, mode='rb') as fp:
            image = fp.read()
        features = tf.train.Features(feature={
            'image' : _bytes_feature(image),
            'label' : _int64_feature(label)
        })
        tf_rec_proto = tf.train.Example(features=features)
        self.writer.write(tf_rec_proto.SerializeToString())

    def close(self):
        self.writer.close()
         

class TFDatasetWriter(object):
    """
    """
    def __init__(self, splits=(('train', _TRAIN_SET_SIZE), ('eval', _EVAL_SET_SIZE))):
        self.data_splits = []
        for name, size in splits:
            split_dict = {}
            split_dict['name']       = name
            split_dict['size']       = size / _NUM_CLASSES 
            split_dict['label_cnt']  = np.zeros([_NUM_CLASSES, 1])
            split_dict['images']     = []
            split_dict['num_tf_rec'] = ceil(size / _MAX_IMG_PER_TF_REC)
            self.data_splits.append(utils.DictToAttrs(split_dict))

    def _split_traininig_set(self):
        """ """
        # Read the training set from csv
        index_file = os.path.join(DATASET_DIR, 'training_ground_truth.csv')
        data = pd.read_csv(index_file, sep=',')
        image_files = data['IMAGE_NAME']
        labels      = data['CLASS_INDEX']

        # Random Shuffle with repeatable pattern
        index = list(range(_DATASET_SIZE))
        np.random.seed(54321)
        np.random.shuffle(index)

        # Split training set into train/val
        for i in index:
            label = labels[i]
            file  = image_files[i]
            label_idx = label - 1
            for split in self.data_splits:
                if split.label_cnt[label_idx] < split.size:
                    split.images.append((file, label))
                    split.label_cnt[label_idx] += 1
                    break
        for data_split in self.data_splits:
            # Save to CSV file
            header = ['IMAGE_NAME', 'CLASS_INDEX']
            df = pd.DataFrame(data_split.images, columns=header)
            df.to_csv(os.path.join(DATASET_DIR, data_split.name + '_set.csv'), index=False)

    def write(self):
        def create_tf_record(rec_file, image_list):
            rec_file = TFRecFile(os.path.join(DATASET_DIR, rec_file))
            for (image, label) in image_list:
                rec_file.add_image(os.path.join(DATASET_DIR, 'training', image), (label - 1))
            rec_file.close()

        start_time = datetime.now()
        t_start    = start_time
        self._split_traininig_set()

        coord = tf.train.Coordinator()
        threads = []
        for split in self.data_splits:
            if split.num_tf_rec > 1:
                data_set = utils.list_split(split.images, split.num_tf_rec)
                for rec_id in range(split.num_tf_rec):
                    args = (split.name+'_0'+str(rec_id+1)+'.tfrecords', data_set[rec_id])
                    th = threading.Thread(target=create_tf_record, args=args)
                    th.start()
                    threads.append(th)
            else:
                args = (split.name+'_01.tfrecords', split.images)
                th = threading.Thread(target=create_tf_record, args=args)
                th.start()
                threads.append(th)
        coord.join(threads)
        print ('ELAPSED TIME:  ', datetime.now() - t_start)
            

##=======##=======##=======##=======##=======##=======##=======##
# Dataset Reader Functions (Load dataset from TFRECORD files)
##=======##=======##=======##=======##=======##=======##=======##
class TFDatasetReader(object):
    """ 
    """
    def __init__(self, image_size=224, shuffle_buff_sz=1500):

        self.name        = 'IntelMovidius-200'
        self.shape       = (image_size, image_size, 3)
        self.dataset_sz  = _TRAIN_SET_SIZE
        self.num_classes = _NUM_CLASSES
        self.scale_min   = image_size + 32
        self.scale_max   = self.scale_min
        train_file_name  = os.path.join(DATASET_DIR, 'train_{:02d}.tfrecords')
        self.train_files = [train_file_name.format(i+1) for i in range(5)]
        self.eval_file   = os.path.join(DATASET_DIR, 'eval_01.tfrecords')
        self.shuffle_sz  = shuffle_buff_sz

    def _parse_eval_rec(self, tf_record, dtype):
        """ """
        feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
        image = tf.image.decode_jpeg(feature['image'], channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = preprocess_image(image, self.shape[0], self.shape[1], resize_side_min=self.scale_min)
        image = tf.cast(image, dtype)
        label = tf.cast(feature['label'], tf.int64)
        return image, label

    def _parse_train_rec(self, tf_record, dtype):
        """ """
        feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
        image = tf.image.decode_jpeg(feature['image'], channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = preprocess_image(image, self.shape[0], self.shape[1], is_training=True)
        label = tf.cast(feature['label'], tf.int64)
        return image, label

    def read(self, batch_size, for_training=True, data_format='NCHW', data_aug=False, dtype=tf.float32):
        """ """
        self.dtype = dtype

        eval_dataset  = tf.data.TFRecordDataset(self.eval_file)
        eval_dataset  = eval_dataset.map(lambda tf_rec: self._parse_eval_rec(tf_rec, dtype))
        eval_dataset  = eval_dataset.prefetch(batch_size)
        eval_dataset  = eval_dataset.batch(batch_size)
        out_types = eval_dataset.output_types
        out_shapes = eval_dataset.output_shapes
        data_iter  = tf.data.Iterator.from_structure(out_types, out_shapes)
        self.eval_init_op = data_iter.make_initializer(eval_dataset)

        if for_training is True:
            train_dataset = tf.data.Dataset.from_tensor_slices(self.train_files)
            train_dataset = train_dataset.flat_map(tf.data.TFRecordDataset)
            train_dataset = train_dataset.shuffle(len(self.train_files))
            # train_dataset = tf.data.TFRecordDataset(self.train_files)
            train_dataset = train_dataset.prefetch(batch_size)
            train_dataset = train_dataset.shuffle(self.shuffle_sz)
            train_dataset = train_dataset.map(lambda tf_rec: self._parse_train_rec(tf_rec, dtype), 8)
            train_dataset = train_dataset.batch(batch_size)
            train_dataset = train_dataset.prefetch(1)
            self.train_init_op = data_iter.make_initializer(train_dataset)
            
        self.images, labels = data_iter.get_next()
        self.labels  = tf.one_hot(labels, self.num_classes)
        if data_format.startswith('NC'):
            self.images = tf.transpose(self.images, [0, 3, 1, 2])


def main():
    """ """
    parser = argparse.ArgumentParser('IntelMovidius Dataset Processing Module')
    parser.add_argument('-w', '--writer', action='store_true', help='Dataset Writer Mode. Save the raw image\
                                                                    dataset into tfrecord files')
    parser.add_argument('-s', '--split', action='store_true')
    parser.add_argument('-r', '--reader', action='store_true', help='Dataset Reader Mode. Parse tfreord files')
    parser.add_argument('-n', '--num-files', type=int, default=5, help='Number of TFRECORD files for training dataset')
    parser.add_argument('-j', '--jpg', action='store_true', help='JPEG Test.')

    args = parser.parse_args()

    # Disable Tensorflow logs except for errors
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.writer is True:
        print ('Dataset Writer ...')
        writer = TFDatasetWriter()
        writer.write()
    elif args.split is True:
        print ('Split Only')
        writer = TFDatasetWriter()
        writer._split_traininig_set()
    elif args.reader is True:
        print ('Reader Test ....')
        reader = TFDatasetReader(image_size=192, shuffle_buff_sz=2000)
        reader.read(128, True, 'NCHW', True)
        with tf.Session() as sess:
            t_start = datetime.now()
            sess.run(reader.eval_init_op)
            i = 0
            while True:
                try:
                    images, labels = sess.run([reader.images, reader.labels])
                    print ('batch ', i)
                    i += 1
                except tf.errors.OutOfRangeError:
                    break
            print (images[0])
            print ('Time for eval set: ', datetime.now() - t_start)
            t_start = datetime.now()            
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
