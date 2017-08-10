from __future__ import division, print_function
import os
import json
import time
import threading
import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from tensorflow.contrib.keras import layers, models, backend, applications

__author__ = 'Jonathan Kyl'

class BaseModel(models.Model):
    ''''''
    def read_tfrecord(self, tfrecord, batch_size=64, capacity=8, 
                      n_threads=4, n_epochs=None, shuffle=True):
        ''''''
        with tf.device('/cpu:0'):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(
                tf.train.string_input_producer([tfrecord], shuffle=True, 
                                               num_epochs=n_epochs))
            feature = {
                'image': tf.FixedLenFeature([], tf.string),
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
            if shuffle:
                return tf.train.shuffle_batch_join(
                    [[img, caption, class_]]*n_threads, 
                    batch_size=batch_size, 
                    capacity=batch_size*capacity, 
                    min_after_dequeue=0)
            return tf.train.batch(
                [img, caption, class_], 
                num_threads=n_threads,
                batch_size=batch_size, 
                capacity=batch_size*capacity)

    def stage_data(self, batch, capacity=8):
        ''''''
        with tf.device('/gpu:0'):
            dtypes = [t.dtype for t in batch]
            shapes = [t.get_shape() for t in batch]
            SA = StagingArea(dtypes, shapes=shapes, capacity=capacity)
            return SA.get(), SA.size(), SA.put(batch)
    
    def start_threads(self, sess, put_op, size, capacity, 
                      n_stage_threads=4):
        ''''''        
        stage_threads = []
        stage_stop = threading.Event()
        def threads_job():
            with sess.graph.as_default():
                while not stage_stop.is_set():
                    try:
                        if sess.run(size) < capacity:
                            sess.run(put_op)
                        else:
                            time.sleep(0.01)
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
        return stage_stop, stage_threads

    def make_summary(self, output_path, img_dict={},
                     scalar_dict={}, text_dict={}, n_images=1):
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
    
    def preproc_caption(self, caption, random=True):
        ''''''
        def per_batch_inds(cap, minus_ones=False, random=random):
            if random:
                start = cap[:, 2]
                good = tf.where(tf.greater(start, -1))
                rand = good[tf.random_uniform(minval=0, maxval=tf.shape(good)[0], 
                                              shape=[], dtype=tf.int32)]
                cap = cap[tf.squeeze(rand)]
            else:
                cap = cap[0]
            if minus_ones:
                return cap
            return tf.where(tf.less(cap, 0), (self.words-1)*tf.ones_like(cap), cap)
        
        def per_batch_onehots(cap):
            cap = per_batch_inds(cap, minus_ones=False)
            return tf.one_hot(cap, depth=self.words)
        
        inds_mapped = tf.map_fn(per_batch_inds, caption, dtype=tf.int64)
        onehots_mapped = tf.map_fn(per_batch_onehots, caption, dtype=tf.float32)
        inds_mapped.set_shape((None, self.length))
        onehots_mapped.set_shape((None, self.length, self.words))
        return onehots_mapped, inds_mapped
    
    def create_table(self, inds_to_words_json):
        ''''''
        return tf.Variable(zip(*sorted(json.loads(open(inds_to_words_json)\
                           .read()).items(), key=lambda x: int(x[0])))[1])
    
    def postproc_caption(self, captions, table):
        ''''''
        captions = tf.argmax(captions, axis=-1)
        captions = tf.unstack(captions)
        strings = tf.gather(table, captions)
        return tf.reduce_join(strings, axis=[1], separator=' ')

    def clip_gradients_by_norm(self, grads_and_vars, clip_gradients):
        ''''''
        g, v = zip(*grads_and_vars)
        clipped, _ = tf.clip_by_global_norm(g, clip_gradients)
        return list(zip(clipped, v))

    
    # # # # # # # # # # # # # # # # # # 
    # END OF BASE MODEL CLASS METHODS #
    # ------------------------------- #
    #      START OF KERAS MODELS      #
    # # # # # # # # # # # # # # # # # #


def cnn(input_shape, kind='InceptionV3', pooling='max'):
    ''''''
    if kind in ('InceptionV3', 'Xception'):
        cnn = eval('applications.'+kind)
    else:
        raise ValueError, 'no such CNN method "{}"'.format(kind)
    return cnn(input_shape=input_shape, classes=None,
               pooling=pooling, include_top=False, weights=None)
    
def rnn_generator(static_dim, sequence_dim, length, code_dim, kind='LSTM'):
    ''''''
    if kind in ('LSTM', 'GRU'):
        rnn = eval('layers.'+kind)
    else:
        raise ValueError, 'no such RNN method "{}"'.format(kind)
    static = layers.Input((static_dim,))
    repeat = layers.RepeatVector(length)(static)
    sequence = layers.Input((length, sequence_dim))
    sequence_emb = layers.Conv1D(static_dim, 1)(sequence)
    code = layers.concatenate([repeat, sequence_emb])
    emb = rnn(code_dim, recurrent_dropout=0.5, unroll=True,
              return_sequences=True, activation='linear')(code)
    out = layers.Conv1D(sequence_dim, 1)(emb)
    return models.Model([static, sequence], out)
    
def rnn_discriminator(sequence_dim, length, code_dim, kind='LSTM'):
    ''''''
    if kind in ('LSTM', 'GRU'):
        rnn = eval('layers.'+kind)
    else:
        raise ValueError, 'no such RNN method "{}"'.format(kind)
    sequence = layers.Input((length, sequence_dim))
    emb = rnn(code_dim, recurrent_dropout=0.5, unroll=True,
              return_sequences=False, activation='linear')(sequence)
    out = layers.Dense(1)(emb)
    return models.Model(sequence, out)

