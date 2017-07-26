from __future__ import print_function, division
import os
import tqdm
import numpy as np
import tensorflow as tf
from utils.models import *

__author__ = 'Jonathan Kyl'

class TextGen(BaseModel):
    def __init__(self, img_size=224, code_dim=128, length=20, words=9509):
        ''''''
        self.img_size = img_size
        self.length = length
        self.words = words
        self.code_dim = code_dim
        
        with tf.variable_scope('CNN'):
            self.phi = cnn(kind='mobilenet', include_top=False, weights=None)
            self.phi_dim = self.phi.output_shape[-1]

        with tf.variable_scope('LSTM'):
            self.lstm = lstm_decoder(input_dim=self.phi_dim, 
                                     length=length, 
                                     code_dim=code_dim,
                                     output_dim=words, 
                                     activation='softmax')

        super(BaseModel, self).__init__(
            self.phi.inputs + self.lstm.inputs,
            self.phi.outputs + self.lstm.outputs)

    def forward_pass(self, x):
        ''''''
        return self.lstm(self.phi(x))
    
    def xent_loss(self, x, y):
        ''''''
        return tf.reduce_mean(losses.categorical_crossentropy(
                    y, self.forward_pass(x)))
    
    def grad_loss(self, L, w):
        grads = tf.gradients(L, w)
        n_params = tf.reduce_sum([tf.cast(tf.reduce_prod(v.shape), tf.float32) for v in w])
        sum_of_squared = tf.reduce_sum([tf.reduce_sum(g**2) for g in grads])
        return sum_of_squared / n_params
    
    def l2_loss(self, x):
        return tf.reduce_mean(self.phi(x)**2)

    def train(self, 
              train_record, 
              val_record,
              output_path,
              batch_size=16, 
              lr_init=1e-4, 
              lambda_Lg=0,
              lambda_L2=0,
              n_read_threads=4,
              n_stage_threads=2,
              capacity=16,
              epoch_size=100,
              validate_every=100,
              save_every=1000,
              decay_every=np.inf,
              cnn_ckpt='/home/paperspace/data/pretrained/mobilenet_1_0_224_tf_no_top.h5',
              ckpt=None):
        '''''' 
        backend.set_learning_phase(True)
        
        with tf.variable_scope('BatchReader'):
            print('creating batch reader')
            
            coord =  tf.train.Coordinator()
            x_train, y_train, _ = self.read_tfrecord(train_record, 
                batch_size=batch_size, capacity=capacity, n_threads=n_read_threads)
            x_train = self.preproc_img(x_train)
            y_train_onehots, y_train_inds = self.preproc_caption(y_train)
        
        with tf.variable_scope('StagingArea'):
            print('creating staging area')

            get, SA_size, put_op = self.stage_data(
                [x_train, y_train_inds], capacity=capacity)
            x_train, y_train_inds = get
            step = tf.Variable(0, name='step')
            update_step = tf.assign_add(step, 1)

        with tf.variable_scope('Optimizer'):
            print('creating optimizer')

            L = self.xent_loss(x_train, y_train_onehots)
            Lg = self.grad_loss(L, self.lstm.trainable_weights)
            L2 = self.l2_loss(x_train)
            Ltot = L + lambda_Lg*Lg + lambda_L2*L2
            lr = lr_init * 0.5**tf.floor(
                tf.cast(step, tf.float32) / decay_every)
            opt = tf.train.AdamOptimizer(lr, beta1=0.8, beta2=0.999)\
                .minimize(Ltot, var_list=self.trainable_weights)

        with tf.variable_scope('Summary'):
            print('creating summary')

            x_val, y_val, _ = self.read_tfrecord(val_record, 
                batch_size=batch_size, n_threads=1)
            x_val = self.preproc_img(x_val)
            y_val_onehots, y_val_inds = self.preproc_caption(y_val)
            sx = tf.Variable(x_val)
            L_val = self.xent_loss(x_val, y_val_onehots)
            Lg_val = self.grad_loss(L_val, self.lstm.trainable_weights)
            L2_val = self.l2_loss(x_val)
            Ltot_val = L_val + lambda_Lg*Lg_val + lambda_L2*L2_val
            scalar_dict = {'xent_loss': L_val, 
                           'grad_loss': Lg_val,
                           'l2_loss': L2_val,
                           'tot_loss': Ltot_val,
                           'SA_size': SA_size, 
                           'lr': lr}
        summary_op, summary_writer = self.make_summary(
            output_path, img_dict={}, 
            scalar_dict=scalar_dict, text_dict={})

        with tf.Session(graph=self.graph) as sess:
            print('starting threads')
            stage_stop, stage_threads = self.start_threads(
                sess, coord, put_op, n_stage_threads=n_stage_threads)
            print('initializing variables')
            sess.run(tf.global_variables_initializer())
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
                    for _ in tqdm.trange(epoch_size, disable=False):
                        if not (coord.should_stop() or stage_stop.is_set()):

                            # update weights                        
                            _, n = sess.run([opt, update_step])

                            # validate
                            if n % validate_every == 1:
                                s = sess.run(summary_op)
                                summary_writer.add_summary(s, n)
                                summary_writer.flush()

                            # write checkpoint
                            if n % save_every == 0:
                                self.save_h5(output_path, n)
                    epoch += 1
            except:
                coord.request_stop()
                stage_stop.set()
                raise
                
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input_path', type=str, 
        help='directory containing jpg or png files on which to train')
    p.add_argument('output_path', type=str, 
        help='output directory to put checkpoints and samples')
    p.add_argument('-g', '--gpu', type=int, default=0,
        help='identity of the GPU to use')
    d = p.parse_args().__dict__
    os.environ['CUDA_VISIBLE_DEVICES'] = str(d.pop('gpu'))
    s = TextGen(
        gan_depth=d.pop('gan_depth'),
        ae_depth=d.pop('ae_depth'))
    s.train(**d)
