from __future__ import division, print_function
import os
import json
import threading
import numpy as np
import tensorflow as tf
from mobilenet import *
from inception_v3 import *
from tensorflow.contrib.keras import layers, models, backend, applications, losses
from tensorflow.contrib.staging import StagingArea

__author__ = 'Jonathan Kyl'

class BaseModel(models.Model):
    ''''''
    def read_tfrecord(self, tfrecord, batch_size=64, capacity=8, n_threads=4):
        ''''''
        with tf.device('/cpu:0'):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(
                tf.train.string_input_producer([tfrecord]))
            feature = {'image': tf.FixedLenFeature([], tf.string),
                       'caption': tf.FixedLenFeature([], tf.string),
                       'class': tf.FixedLenFeature([], tf.int64),
                       'image_size': tf.FixedLenFeature([], tf.int64),
                       'vocab_size': tf.FixedLenFeature([], tf.int64),
                       'length': tf.FixedLenFeature([], tf.int64)} 
            f = tf.parse_single_example(serialized_example, features=feature)
            img, caption, class_ = f['image'], f['caption'], f['class']
            img = tf.image.decode_jpeg(img)
            img.set_shape((self.img_size, self.img_size, 3))
            caption = tf.decode_raw(caption, tf.int64)
            caption.set_shape((7*self.length,))
            caption = tf.reshape(caption, (7, self.length))
            class_.set_shape([])
            return tf.train.shuffle_batch_join(
                [[img, caption, class_]]*n_threads, 
                batch_size=batch_size, 
                capacity=batch_size*capacity, 
                min_after_dequeue=0)

    def stage_data(self, batch, capacity=8):
        ''''''
        with tf.device('/gpu:0'):
            dtypes = [t.dtype for t in batch]
            shapes = [t.get_shape() for t in batch]
            SA = StagingArea(dtypes, shapes=shapes, capacity=capacity)
            return SA.get(), SA.size(), SA.put(batch)
    
    def start_threads(self, sess, coord, put_op, n_stage_threads=4):
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
            for i in range(n_stage_threads):
                thread = threading.Thread(target=threads_job)
                stage_threads.append(thread)
                thread.daemon = True
                thread.start()
        except:
            stage_stop.set()
            raise
        tf.train.start_queue_runners(sess=sess, coord=coord)
        return stage_stop, stage_threads

    def make_summary(self, output_path, img_dict={}, scalar_dict={}, text_dict={}, n_images=5):
        ''''''
        summaries = []
        for k, v in img_dict.items():
            summaries.append(tf.summary.image(k, v, n_images))
        for k, v in scalar_dict.items():
            summaries.append(tf.summary.scalar(k, v))
        for k, v in text_dict.items():
            summaries.append(tf.summary.text(k, v))
        summary_op = tf.summary.merge(summaries)
        summary_writer = tf.summary.FileWriter(
            output_path, graph=self.graph)
        return summary_op, summary_writer
    
    def save_h5(self, output_path, n):
        ''''''
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save(os.path.join(
            output_path, 'ckpt_update-{}.h5'\
            .format(str(int(n)).zfill(10))))
        
    def preproc_img(self, img):
        ''''''
        img = tf.cast(img, tf.float32)
        img -= 127.5
        img /= 127.5
        return img

    def postproc_img(self, img):
        ''''''
        img = tf.clip_by_value(img, -1, 1)
        img *= 127.5
        img += 127.5
        return tf.cast(img, tf.uint8)
    
    def preproc_caption(self, caption):
        ''''''
        def per_batch_inds(cap, minus_ones=False):
            start = cap[:, 1]
            good = tf.where(tf.greater(start, -1))
            rand = good[tf.random_uniform(minval=0, maxval=tf.shape(good)[0], 
                                          shape=[], dtype=tf.int32)]
            cap = cap[tf.squeeze(rand)]
            if not minus_ones:
                return tf.where(tf.less(cap, 0), (self.words-1)*tf.ones_like(cap), cap)
            return cap
        def per_batch_onehots(cap):
            cap = per_batch_inds(cap, minus_ones=True)
            nonpad = tf.squeeze(tf.where(tf.greater(cap, -1)))
            cap_nonpad = tf.gather(cap, nonpad)
            eye = tf.eye(self.words, dtype=tf.float32)
            onehot = tf.gather(eye, cap_nonpad)
            padded = tf.pad(onehot, [[0, self.length-tf.shape(onehot)[0]], [0, 0]], 
                            mode='CONSTANT', constant_values=0)
            return padded
        inds_mapped = tf.map_fn(per_batch_inds, caption, dtype=tf.int64)
        onehots_mapped = tf.map_fn(per_batch_onehots, caption, dtype=tf.float32)
        inds_mapped.set_shape((None, self.length))
        onehots_mapped.set_shape((None, self.length, self.words))
        return onehots_mapped, inds_mapped
    
def create_table(inds_to_words_json):
    return tf.Variable(zip(*sorted(json.loads(open(inds_to_words_json)\
        .read()).items(), key=lambda x: int(x[0])))[1])
    
def postproc_caption(caption, table):
    captions = tf.unstack(caption)
    strings = tf.gather(table, caption)
    return tf.reduce_join(strings, axis=[1], separator=' ')

def clip_gradients_by_norm(grads_and_vars, clip_gradients):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
    return list(zip(clipped_gradients, variables))

def SELU(x):
    ''''''
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * layers.ELU(alpha=alpha)(x)

def rnn_decoder(input_dim, length, code_dim, output_dim, 
                rnn='LSTM', activations=['linear', 'linear']):
    
    for i, a in enumerate(activations):
        if a == 'selu': activations[i] = SELU
    
    inp = layers.Input(shape=(input_dim,), name='lstm_input')
    def repeat_timedim(x):
        x = tf.expand_dims(x, 1)
        return tf.tile(x, [1, length, 1])
    rep = layers.Lambda(repeat_timedim)(inp)
    rnn = eval('layers.'+rnn)(code_dim, return_sequences=True, 
                unroll=True, activation=activations[0])(rep)
    out = layers.Conv1D(output_dim, 1, activation=activations[1])(rnn)
    return models.Model(inp, out)

def cnn(kind='xception', classes=None, weights=None, include_top=False, pooling='max'):
    ''''''
    if kind == 'inception':
        return InceptionV3(classes=classes, pooling=pooling,
            include_top=include_top, weights=weights)
    elif kind == 'mobilenet':
        return MobileNet(classes=classes, pooling=pooling,
            include_top=include_top, weights=weights)

