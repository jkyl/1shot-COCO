import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import glob
import json
import tqdm
import os

def crop_to_aspect_ratio(img, aspect_ratio=1.):
    ''''''
    shape = tf.shape(img)
    H, W = shape[0], shape[1]
    W = tf.cond(
            tf.equal(H, tf.minimum(H, W)),
            lambda: tf.cast(tf.cast(H, tf.float32)*aspect_ratio, tf.int32), 
            lambda: W)
    H = tf.cond(
            tf.equal(W, tf.minimum(H, W)),
            lambda: tf.cast(tf.cast(W, tf.float32)/aspect_ratio, tf.int32), 
            lambda: H)
    return tf.image.resize_image_with_crop_or_pad(img, H, W)

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

def read_imgs(img_files, size=256, center_crop=True):
    t_files = tf.train.string_input_producer(img_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(t_files)
    img = tf.image.decode_jpeg(img_bytes)
    if center_crop:
        img = crop_to_aspect_ratio(img, 1.)
    rs = resize_lanczos(img, (size, size))
    img_bytes = tf.image.encode_jpeg(rs)
    return img_bytes
    
def read_captions(captions_json):
    j = json.loads(open(captions_json).read())
    _, captions = zip(*sorted(j.items()))
    captions = np.array(captions)
    return tf.train.input_producer(captions, shuffle=False).dequeue()

def read_classes(classes_json):
    j = json.loads(open(classes_json).read())
    d = {d['image_id']: d['category_id'] for d in j['annotations']}
    _, classes = zip(*sorted(d.items()))
    return tf.train.input_producer(classes, shuffle=False).dequeue()

def main(imgs_path, captions_json, classes_json, output_tfrecord,
         img_size=256, center_crop=True):
    ''''''
    img_files = sorted(glob.glob(os.path.join(imgs_path, '*.jpg')))
    n = len(img_files)
    print('found {} images'.format(n))
    coord = tf.train.Coordinator()
    print('made coord')
    img_op = read_imgs(img_files)
    print('got img tensor')
    caption_op = read_captions(captions_json)
    print('got captions')
    class_op = read_classes(classes_json)
    print('got classes')
    writer = tf.python_io.TFRecordWriter(output_tfrecord)
    print('starting')
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for _ in tqdm.trange(n):
                img, caption, class_ = sess.run([img_op, caption_op, class_op])
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(img),
                    'caption': _bytes_feature(caption.tostring()),
                    'class': _int64_feature(class_)
                }))
                writer.write(example.SerializeToString())
            writer.close()
        except:
            raise
    
if __name__== '__main__':
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('imgs_path', type=str)
    p.add_argument('captions_json', type=str)
    p.add_argument('classes_json', type=str)
    p.add_argument('output_tfrecord', type=str)
    p.add_argument('-s', '--img_size', type=int, default=256,
        help='sidelength of images')
    p.add_argument('-c', '--center_crop', type=bool, default=True,
        help='whether or not to perform a center crop before resize')
    d = p.parse_args().__dict__
    main(**d)
        
    
