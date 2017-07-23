from __future__ import print_function, division
from scipy.misc import imresize
import tensorflow as tf
import threading
import warnings
import glob
import tqdm
import time
import Queue
import os

__author__ = 'Jonathan Kyl'

def glob_whitelist_files(path, children=False, whitelist=['jpg', 'jpeg', 'png', 'webp']):
    ''''''    
    return [f for f in glob.glob(os.path.join(path, '*/*' if children else '*')) \
            if any([f.lower().endswith(ext) for ext in whitelist])]

def crop_to_aspect_ratio(img, aspect_ratio):
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
    ''''''
    def rs(x):
        return imresize(x, target_shape, interp='lanczos')
    return tf.py_func(rs, [img], tf.uint8)
    
    
def read_batch_multithread(image_directories, image_shapes, batch_size, descend='auto',
                           crop_aspect_ratios=None, shuffle=True, n_threads=4, capacity=8):
    ''''''
    with tf.device('/cpu:0'):
        imgs = []
        if type(image_directories) is str:
            image_directories = [image_directories]
            
        if len(image_shapes) != len(image_directories):
            image_shapes = [image_shapes]*len(image_directories)
            
        if not hasattr(crop_aspect_ratios, '__iter__'):
            crop_aspect_ratios = [crop_aspect_ratios]*len(image_directories)
        
        reader = tf.WholeFileReader()
        for i, image_directory in enumerate(image_directories):
            image_shape = image_shapes[i]
            crop_aspect_ratio = crop_aspect_ratios[i]
            
            if descend=='auto':
                parent_files = glob_whitelist_files(image_directory)
                child_files = glob_whitelist_files(image_directory, children=True)
                print('found {} images in parent and {} images in children (descend=`auto`)'\
                      .format(len(parent_files), len(child_files)))
                if len(parent_files) == len(child_files) and len(child_files) == 0:
                    raise ValueError(
                        'found zero whitelist image files in parent and child directories')
                elif len(parent_files) > len(child_files):
                    print('using parent directory')
                    files = parent_files
                else:
                    print('using child directories')
                    files = child_files
            else:
                files = glob_whitelist_files(image_directory, children=descend)
                print('found {} images in {} (descend={})'\
                      .format(len(files), 'children' if descend else 'parent', descend))
            
            files = sorted(files) if not shuffle else files
            file_gen = tf.train.string_input_producer(files, shuffle=shuffle)
            img_bytes = reader.read(file_gen)[1]
            img = tf.image.decode_jpeg(img_bytes, channels=image_shape[-1])

            if crop_aspect_ratio:
                img = crop_to_aspect_ratio(img, crop_aspect_ratio)
            img = resize_lanczos(img, image_shape[:2])
            img.set_shape(image_shape)
            #img = tf.image.resize_images(img, image_shape[:2], 
            #                             method=tf.image.ResizeMethod.BICUBIC)
            imgs.append(img)
            
        if shuffle:
            return tf.train.shuffle_batch_join([imgs]*n_threads, 
                                               batch_size=batch_size, 
                                               capacity=batch_size*capacity, 
                                               min_after_dequeue=0)
        return tf.train.batch_join([imgs]*n_threads, 
                                   batch_size=batch_size, 
                                   capacity=batch_size*capacity, 
                                   min_after_dequeue=0)
        
def start_staging_threads(sess, put_op, n_threads=1):
    ''''''
    stage_threads = []
    stage_stop = threading.Event()
    
    def threads_job():
        with sess.graph.as_default():
            while not stage_stop.is_set():
                try:
                    sess.run(put_op)
                except:
                    stage_stop.set()
                    raise
    try:
        for i in range(n_threads):
            thread = threading.Thread(target=threads_job)
            stage_threads.append(thread)
            thread.daemon = True
            thread.start()
    except:
        stage_stop.set()
        raise

    return stage_stop, stage_threads

def read_record(tfrecord):
    reader = tf.train.RecordReader()
    
