import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import glob
import json
import tqdm
import os

def crop_to_aspect_ratio(img):
    ''''''
    shape = tf.shape(img)
    print(shape)
    X = tf.minimum(shape[0], shape[1])
    return tf.image.resize_image_with_crop_or_pad(img, X, X)

def resize_lanczos(img, target_shape):
    def rs(x):
        if x.shape[-1] == 1:
            x = np.tile(x, [1, 1, 3])
        return imresize(x, target_shape, interp='lanczos')
    return tf.py_func(rs, [img], tf.uint8)

def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value]))

def read_imgs(img_files, size=224, center_crop=True):
    t_files = tf.train.string_input_producer(img_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(t_files)
    img = tf.image.decode_jpeg(img_bytes)
    if center_crop:
        img = crop_to_aspect_ratio(img)
    rs = resize_lanczos(img, (size, size))
    img_bytes = tf.image.encode_jpeg(rs)
    return img_bytes
    
def read_captions(captions_json):
    j = json.loads(open(captions_json).read())
    _, captions = zip(*sorted(j.items(), key=lambda x: int(x[0])))
    captions = np.array(captions)
    return tf.train.input_producer(captions, shuffle=False).dequeue(), captions.max()+1, captions.shape[-1]

def read_classes(classes_json):
    j = json.loads(open(classes_json).read())
    d = {d['image_id']: d['category_id'] for d in j['annotations']}
    _, classes = zip(*sorted(d.items(), key=lambda x: int(x[0])))
    return tf.train.input_producer(classes, shuffle=False).dequeue()

def main(imgs_path, captions_json, classes_json, base_tfrecord, 
         novel_tfrecord=None, lowshot_value=20, 
         img_size=224, center_crop=True, truncate=False):
    ''''''
    img_files = sorted(glob.glob(os.path.join(imgs_path, '*.jpg')))
    if truncate:
        img_files = img_files[:truncate]
    n = len(img_files)
    print('found {} images'.format(n))
    coord = tf.train.Coordinator()
    print('made coord')
    img_op = read_imgs(img_files, size=img_size, center_crop=center_crop)
    print('got img tensor')
    caption_op, vocab_size, length = read_captions(captions_json)
    print('got captions')
    class_op = read_classes(classes_json)
    print('got classes')
    base = tf.python_io.TFRecordWriter(base_tfrecord)
    if novel_tfrecord:
        novel = tf.python_io.TFRecordWriter(novel_tfrecord)
        novel_classes = set([ 1, 23, 15, 46, 42, 56, 72, 17, 
                             60, 14, 61, 32, 30, 35, 58, 67,  
                              8, 71, 40,  2, 64, 52, 24, 27, 
                             63,  5, 79, 21, 37, 74, 49, 70, 
                             53, 45, 65, 16, 75,  9, 41, 26])
        novel_counts = dict(zip(list(novel_classes), [0]*40))
    print('starting')
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for _ in tqdm.trange(n):
                img, caption, class_ = sess.run([img_op, caption_op, class_op])
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_size': _int64_feature(img_size),
                    'vocab_size': _int64_feature(vocab_size),
                    'length': _int64_feature(length),
                    'image': _bytes_feature(img),
                    'caption': _bytes_feature(caption.tostring()),
                    'class': _int64_feature(class_)
                }))
                if novel_tfrecord: 
                    if class_ in novel_classes:
                        if novel_counts[class_] < lowshot_value:
                            novel.write(example.SerializeToString())
                            novel_counts[class_] += 1
                    else:
                        base.write(example.SerializeToString())
                else:
                    base.write(example.SerializeToString())
            base.close()
            if novel_tfrecord:
                novel.close()
        except:
            raise
    
if __name__== '__main__':
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('imgs_path', type=str)
    p.add_argument('captions_json', type=str)
    p.add_argument('classes_json', type=str)
    p.add_argument('base_tfrecord', type=str)
    p.add_argument('-nv', '--novel_tfrecord', type=str, default=None,
        help='second output for novel classes')
    p.add_argument('-lv', '--lowshot_value', type=int, default=20,
        help='number of images per novel class')
    p.add_argument('-s', '--img_size', type=int, default=224,
        help='sidelength of images')
    p.add_argument('-c', '--center_crop', type=bool, default=True,
        help='whether or not to perform a center crop before resize')
    p.add_argument('-t', '--truncate', type=int, default=False,
        help='whether to truncate the total number of examples')
    d = p.parse_args().__dict__
    main(**d)
        
    
