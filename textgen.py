from __future__ import print_function, division
import os
import time
import tqdm
import numpy as np
import tensorflow as tf
from utils.models import *

__author__ = 'Jonathan Kyl'

class TextGen(BaseModel):
    def __init__(self, 
                 img_size=299, 
                 code_dim=1024, 
                 length=20, 
                 words=9412, 
                 cnn_type='inception',
                 pooling=None,
                 rnn_type='LSTM'):
        ''''''
        self.img_size = img_size
        self.length = length
        self.words = words
        self.code_dim = code_dim
        
        # build image feature extractor
        with tf.variable_scope('CNN'):
            self.phi = cnn(
                input_shape=(img_size, img_size, 3),
                kind=cnn_type, 
                pooling=pooling)
            self.phi_dim = self.phi.output_shape[1:]

        # build text generator
        with tf.variable_scope('TextGen'):
            self.gen = spatial_attention_rnn(
                input_shape=self.phi_dim, 
                length=length, 
                code_dim=code_dim,
                output_dim=words, 
                rnn=rnn_type)
            
        # build text discriminator
        with tf.variable_scope('TextDisc'):
            self.disc = rnn_discriminator(
                length=length, words=words, 
                img_features=self.phi_dim,
                code_dim=code_dim, rnn=rnn)

        # inherit as one model
        super(BaseModel, self).__init__(
            self.phi.inputs + self.gen.inputs + self.disc.inputs,
            self.phi.outputs + self.gen.outputs + self.disc.outputs)

    def G(self, x):
        ''''''
        return self.gen(self.phi(x))
    
    def D(self, y, x):
        ''''''
        return self.disc([y, self.phi(x)])
    
    def L_adv(self, y, x, label):
        ''''''
        pred = self.D(y, x)
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label*tf.ones_like(pred), 
                logits=pred))
    
    def xent(self, y_inds, x):
        ''''''
        pred = self.G(x)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_inds, 
                logits=pred))
    
    def weight_norm(self, weights):
        ''''''
        return tf.reduce_sum([tf.reduce_sum(w**2) for w in weights])
    
    def train(self, 
              train_record, 
              val_record,
              output_path,
              batch_size=16,
              optimizer='adam',
              lr_init=1e-4, 
              decay_every=250000,
              g_updates=1, 
              d_updates=0,
              clip_gradients=5.0,
              lambda_x=1,
              lambda_r=0.001,
              lambda_d=0,
              lambda_m=0,
              cnn_trainable=False,
              random_captions=True,
              n_read_threads=6,
              n_stage_threads=6,
              capacity=16,
              epoch_size=10000,
              save_every=10000,
              validate_every=100,
              ckpt=None,
              cnn_ckpt='pretrained/inception_v3_weights.h5',
              inds_to_words_json='/home/paperspace/data/ms_coco/preproc/preproc_vocab-9412_threshold-5_length-20/inds_to_words.json',
             ):
        '''''' 
        
        with tf.variable_scope('BatchReader'):
            print('creating batch reader')
            
            coord =  tf.train.Coordinator()
            
            # read and preprocess training records
            x_train, y_train, cls_train = self.read_tfrecord(
                train_record, 
                batch_size=batch_size, 
                capacity=capacity, 
                n_threads=n_read_threads)
            x_train = self.preproc_img(x_train)
            y_train_onehots, y_train_inds = self.preproc_caption(
                y_train, random=random_captions)
            
            # read and preprocess validation records
            x_val, y_val, cls_val = self.read_tfrecord(
                val_record, 
                batch_size=batch_size, 
                capacity=1, 
                n_threads=1)
            x_val = self.preproc_img(x_val)
            y_val_onehots, y_val_inds = self.preproc_caption(
                y_val, random=random_captions)
        
        with tf.variable_scope('StagingArea'):
            print('creating staging area')

            # create training queue on GPU
            train_get, train_size, train_put = self.stage_data(
                [x_train, y_train_onehots, y_train_inds],
                capacity=capacity)
            x_train, y_train_onehots, y_train_inds = train_get
            
            # create validation queue on GPU
            val_get, val_size, val_put = self.stage_data(
                [x_val, y_val_onehots, y_val_inds], 
                capacity=1)
            x_val, y_val_onehots, y_val_inds = val_get
            
            # global step and learning phase flag
            step = tf.Variable(0, name='step')
            update_step = tf.assign_add(step, 1)
            learning = backend.learning_phase()

        with tf.variable_scope('Optimizer'):
            print('creating optimizer')

            # get variables
            D_vars = self.disc.trainable_variables
            G_vars = self.gen.trainable_variables
            if cnn_trainable:
                G_vars += self.phi.trainable_variables
                
            # the generated caption
            y_hat = self.G(x_train)
            
            # define loss terms:
            # fake as fake, fake as real, and real as real
            L_ff = self.L_adv(y_hat, x_train, 0)
            L_fr = self.L_adv(y_hat, x_train, 1)            
            L_rr = self.L_adv(y_train_onehots, x_train, 1)
            
            # mismatch, xentropy, and weight regularizer
            L_m = self.L_adv(tf.random_shuffle(y_train_onehots), x_train, 0)
            L_x = self.xent(y_train_inds, x_train)
            L_r = self.weight_norm(self.gen.trainable_weights)

            # total discriminator loss
            L_D = L_ff + L_rr + lambda_m*L_m
                
            # total generator loss
            L_G = lambda_d*L_fr \
                + lambda_x*L_x \
                + lambda_r*L_r 
                              
            # learning rate with periodic halving
            lr = lr_init * 0.5**tf.floor(tf.cast(step, tf.float32) / decay_every)
                
            # optimizer(s): one of {adam, sgd}
            if optimizer == 'adam':
                G_opt = tf.train.AdamOptimizer(lr, beta1=0.5)
                if d_updates: D_opt = tf.train.AdamOptimizer(lr, beta1=0.5)\
                                      .minimize(L_D, var_list=D_vars)
            elif optimizer == 'sgd':
                G_opt = tf.train.GradientDescentOptimizer(lr)
                if d_updates: D_opt = tf.train.GradientDescentOptimizer(lr)\
                                      .minimize(L_D, var_list=D_vars)
            else:
                raise ValueError, 'optimizer can be "adam" or "sgd"'
            
            # rescale by L2 norm, if specified
            G_grad = G_opt.compute_gradients(L_G, var_list=G_vars)
            if clip_gradients:
                G_grad = clip_gradients_by_norm(G_grad, clip_gradients)
            G_opt = G_opt.apply_gradients(G_grad)

        with tf.variable_scope('Summary'):
            print('creating summary')
            
            # validation caption
            y_hat_val = self.G(x_val)
            
            # validation loss terms
            L_ff_val = self.L_adv(y_hat_val, x_val, 0)
            L_fr_val = self.L_adv(y_hat_val, x_val, 1)
            L_rr_val = self.L_adv(y_val_onehots, x_val, 1)
            L_m_val = self.L_adv(tf.random_shuffle(y_val_onehots), x_val, 0)
            L_x_val = self.xent(y_val_inds, x_val)
            L_D_val = L_ff_val + L_rr_val + lambda_m*L_m_val
            L_G_val = lambda_d*L_fr_val \
                    + lambda_x*L_x_val \
                    + lambda_r*L_r 
            
            # decode predictions back to words
            table = create_table(inds_to_words_json)
            real_str = postproc_caption(y_val_inds, table)[0]
            gen_str = postproc_caption(tf.argmax(y_hat_val, axis=-1), table)[0]
            txtfile = os.path.join(output_path, 'output.txt')

            # associate scalars with display names
            scalar_dict = {'L_ff': L_ff_val, 
                           'L_fr': L_fr_val,
                           'L_rr': L_rr_val,
                           'L_m': L_m_val,
                           'L_x': L_x_val,
                           'L_D': L_D_val,
                           'L_G': L_G_val,
                           'L_r': L_r,
                           'SA_size': train_size, 
                           'lr': lr}
        
        # summarize 
        summary_op, summary_writer = self.make_summary(
            output_path, scalar_dict=scalar_dict)

        # start the session 
        with tf.Session(graph=self.graph) as sess:
            
            print('starting threads')
            tf.train.start_queue_runners(sess=sess, coord=coord)
            train_stop, train_threads = self.start_threads(
                sess, train_put, n_stage_threads=n_stage_threads)
            val_stop, val_threads = self.start_threads(
                sess, val_put, n_stage_threads=1)
            
            print('initializing variables')
            sess.run(tf.global_variables_initializer())
            
            if save_every < np.inf:
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
                    print('epoch: ' + str(epoch))
                    time.sleep(0.2)
                    epoch += 1
                    
                    for _ in tqdm.trange(epoch_size, disable=False):
                        if not (coord.should_stop() 
                            or train_stop.is_set() 
                            or val_stop.is_set()):

                            # update discriminator
                            for _ in range(d_updates):
                                sess.run(D_opt, {learning: True})
                            
                            # update generator  
                            for _ in range(g_updates):
                                sess.run(G_opt, {learning: True})                                
                            
                            # update global step
                            n = sess.run(update_step)

                            # validate
                            if n % validate_every == 1:
                                smry, rst, gst = sess.run(
                                    [summary_op, real_str, gen_str],
                                    {learning: False})
                                rst = 'Real: "{}"\n'.format(rst)
                                rst += 'Fake: "{}"\n'.format(gst)
                                print(rst, file=open(txtfile, 'a'))
                                summary_writer.add_summary(smry, n)
                                summary_writer.flush()

                            # write checkpoint
                            if n % save_every == 0:
                                self.save_h5(output_path, n)
                        else:
                            raise IOError, 'queues closed'
                            
            # exit behaviour: request thread stop and wait for 
            # them to recieve message before exiting session context
            except:
                coord.request_stop()
                train_stop.set()
                val_stop.set()
                time.sleep(0.2)
                raise
                
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
