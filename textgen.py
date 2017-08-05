from __future__ import print_function, division
import os
import time
import tqdm
import numpy as np
import tensorflow as tf
from utils.models import *

__author__ = 'Jonathan Kyl'

class TextGen(BaseModel):
    def __init__(self, img_size=224, 
                 code_dim=128, 
                 length=20, 
                 words=9412, 
                 pooling=None, rnn='LSTM'):
        ''''''
        self.img_size = img_size
        self.length = length
        self.words = words
        self.code_dim = code_dim
        
        with tf.variable_scope('ImgFeat'):
            self.phi = cnn(
                input_shape=(img_size, img_size, 3),
                kind='inception', 
                pooling=pooling)
            self.phi_dim = self.phi.output_shape[1:]

        with tf.variable_scope('TextGen'):
            self.gen = spatial_attention_rnn(
                input_shape=self.phi_dim, 
                length=length, 
                code_dim=code_dim,
                output_dim=words, 
                rnn=rnn)
            
        with tf.variable_scope('TextDisc'):
            self.disc = rnn_discriminator(
                length=length, words=words, 
                img_features=self.phi_dim,
                code_dim=code_dim, rnn=rnn)

        super(BaseModel, self).__init__(
            self.phi.inputs + self.gen.inputs + self.disc.inputs,
            self.phi.outputs + self.gen.outputs + self.disc.outputs)

    def G(self, x):
        ''''''
        return self.gen(self.phi(x))
    
    def D(self, y, x):
        ''''''
        return self.disc([y, self.phi(x)])
    
    def L(self, y, x, label):
        pred = self.D(y, x)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label*tf.ones_like(pred), logits=pred))
    
    def xent(self, y, x):
        pred = self.G(x)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=pred))
    
    def train(self, 
              train_record, 
              val_record,
              output_path,
              lambda_m=1,
              lambda_x=1,
              lambda_d=1,
              clip_gradients=False,
              batch_size=16, 
              lr_init=1e-4, 
              g_updates=1, 
              cnn_trainable=True,
              random_captions=True,
              n_read_threads=4,
              n_stage_threads=2,
              capacity=16,
              epoch_size=100,
              validate_every=100,
              save_every=1000,
              decay_every=np.inf,
              cnn_ckpt='pretrained/inception_v3_weights.h5',
              ckpt=None,
              inds_to_words_json='/home/paperspace/data/ms_coco/preproc/preproc_vocab-9412_threshold-5_length-20/inds_to_words.json'):
        '''''' 
        
        with tf.variable_scope('BatchReader'):
            print('creating batch reader')
            
            coord =  tf.train.Coordinator()
            x_train, y_train, _ = self.read_tfrecord(train_record, 
                batch_size=batch_size, capacity=capacity, n_threads=n_read_threads)
            x_train = self.preproc_img(x_train)
            y_train, _ = self.preproc_caption(y_train, random=random_captions)
        
        with tf.variable_scope('StagingArea'):
            print('creating staging area')

            get, SA_size, put_op = self.stage_data(
                [x_train, y_train], capacity=capacity)
            x_train, y_train = get
            step = tf.Variable(0, name='step')
            update_step = tf.assign_add(step, 1)
            learning = backend.learning_phase()

        with tf.variable_scope('Optimizer'):
            print('creating optimizer')

            D_vars = self.disc.trainable_variables
            G_vars = self.gen.trainable_variables
            if cnn_trainable:
                G_vars += self.phi.trainable_variables
                
            y_hat = self.G(x_train)
            L_m = self.L(tf.random_shuffle(y_train), x_train, 0)
            L_x = self.xent(y_train, x_train)
            L_D = self.L(y_train, x_train, 1) + self.L(y_hat, x_train, 0) + lambda_m*L_m
            L_G = lambda_d*self.L(y_hat, x_train, 1) + lambda_x*L_x
                              
            lr = lr_init * 0.5**tf.floor(tf.cast(step, tf.float32) / decay_every)
            D_opt = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(L_D, var_list=D_vars)
            G_opt = tf.train.AdamOptimizer(lr, beta1=0.5)
            G_grad = G_opt.compute_gradients(L_G, var_list=G_vars)
            if clip_gradients:
                G_grad = clip_gradients_by_norm(G_grad, clip_gradients)
            G_opt = G_opt.apply_gradients(G_grad)

        with tf.variable_scope('Summary'):
            print('creating summary')

            x_val, y_val, _ = self.read_tfrecord(val_record, 
                batch_size=batch_size, n_threads=1)
            
            x_val = self.preproc_img(x_val)
            y_val, _ = self.preproc_caption(y_val, random=random_captions)
            y_hat_val = self.G(x_val)
            
            L_d_val = self.L(y_hat_val, x_val, 1)
            L_m_val = self.L(tf.random_shuffle(y_val), x_val, 0)
            L_x_val = self.xent(y_val, x_val)
            
            table = create_table(inds_to_words_json)
            real_str = postproc_caption(tf.argmax(y_val, axis=-1), table)
            gen_str = postproc_caption(tf.argmax(y_hat_val, axis=-1), table)

            scalar_dict = {'L_d': L_d_val, 
                           'L_m': L_m_val,
                           'L_x': L_x_val,
                           'SA_size': SA_size, 
                           'lr': lr}
            text_dict={'real': real_str, 'gen': gen_str}
        summary_op, summary_writer = self.make_summary(
            output_path, scalar_dict=scalar_dict, text_dict=text_dict)

        with tf.Session(graph=self.graph) as sess:
            
            print('starting threads')
            stage_stop, stage_threads = self.start_threads(
                sess, coord, put_op, n_stage_threads=n_stage_threads)
            
            print('initializing variables')
            sess.run(tf.global_variables_initializer())
            
            if save_every != np.inf:
                print('saving weights')
                self.save_h5(output_path, 0)
                
            if ckpt: 
                print('loading weights from '+ckpt)
                self.load_weights(ckpt)
                
            elif cnn_ckpt:
                print('loading cnn weights from '+cnn_ckpt)
                self.phi.load_weights(cnn_ckpt)
                
            print('finalizing graph')
            self.graph.finalize()
            try:
                epoch = 1
                while True:
                    print('epoch: ' + str(epoch)); time.sleep(0.2)
                    for _ in tqdm.trange(epoch_size, disable=False):
                        if not (coord.should_stop() or stage_stop.is_set()):

                            # update discriminator
                            _, n = sess.run([D_opt, update_step], 
                                            {learning: True})
                            
                            # update generator  
                            for _ in range(g_updates):
                                sess.run(G_opt, {learning: True})

                            # validate
                            if n % validate_every == 1:
                                smry, rst, gst = sess.run(
                                    [summary_op, real_str, gen_str],
                                    {learning: False})
                                print(rst[0])
                                print(gst[0])
                                summary_writer.add_summary(smry, n)
                                summary_writer.flush()

                            # write checkpoint
                            if n % save_every == 0:
                                self.save_h5(output_path, n)
                    epoch += 1
            except:
                coord.request_stop()
                stage_stop.set()
                raise
                
    def infer(self, ckpt, val_tfrecord, output_path, n=1,
              inds_to_words_json='',
              batch_size=16, 
              n_threads=4):
        ''''''
        with tf.variable_scope('Inference'):
            table = create_table(inds_to_words_json)
            coord = tf.train.Coordinator()
            x_val, _, _ = self.read_tfrecord(val_tfrecord, 
                batch_size=batch_size, n_threads=n_threads)
            x_val = self.preproc_img(x_val)
            y_hat = self.G(x_val)
            y_hat_inds = tf.argmax(y_hat, axis=-1)
            strings = postproc_caption(y_hat_inds, table)

        with tf.Session() as sess:
            tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.global_variables_initializer())
            self.load_weights(ckpt)
            rv = []
            for i in tqdm.trange(n):
                rv.append(sess.run([x_val, strings], {backend.learning_phase(): False}))
            return rv
                
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('train_record', type=str, 
        help='tfrecord file containing preprocessed and serialized training data')
    p.add_argument('val_record', type=str, 
        help='tfrecord file containing preprocessed and serialized validation data')
    p.add_argument('output_path', type=str, 
        help='path in which to save weights and tensorboard summaries')
    d = p.parse_args().__dict__
    tg = TextGen()
    tg.train(**d)
