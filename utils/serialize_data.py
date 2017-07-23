import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import glob
import json
import tqdm
import os

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

def read_imgs(img_files, size=64):
    t_files = tf.train.string_input_producer(img_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(t_files)
    img = tf.image.decode_jpeg(img_bytes)
    rs = resize_lanczos(img, (size, size))
    img_bytes = tf.image.encode_jpeg(rs)
    return img_bytes
    
def read_captions(captions_json):
    j = json.loads(open(captions_json).read())
    d = {d['image_id']: d['caption'] for d in j['annotations']}
    _, captions = zip(*sorted(d.items()))
    return tf.train.string_input_producer(captions, shuffle=False).dequeue()

def read_classes(classes_json):
    j = json.loads(open(classes_json).read())
    d = {d['image_id']: d['category_id'] for d in j['annotations']}
    _, classes = zip(*sorted(d.items()))
    return tf.train.input_producer(classes, shuffle=False).dequeue()

def main(imgs_path, captions_json, classes_json, output_tfrecord):
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
                    'caption': _bytes_feature(caption.encode()),
                    'class': _int64_feature(class_)
                }))
                writer.write(example.SerializeToString())
            writer.close()
        except:
            raise

    
if __name__== '__main__':
     main('/home/paperspace/data/ms_coco/val2014', 
     '/home/paperspace/data/ms_coco/captions_val2014.json',
     '/home/paperspace/data/ms_coco/instances_val2014.json',
     '/home/paperspace/data/ms_coco/val.tfrecord')
    
    
