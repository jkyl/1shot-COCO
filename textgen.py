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
                 code_dim=512, 
                 length=20, 
                 words=9413, 
                 cnn_type='inception',
                 pooling='avg',
                 rnn_type='LSTM'):
        ''''''
        self.img_size = img_size
        self.length = length
        self.words = words
        self.code_dim = code_dim
        
        # build image feature extractor
        with tf.variable_scope('CNN'):
            phi = cnn(
                input_shape=(img_size, img_size, 3),
                kind=cnn_type, 
                pooling=pooling)
            out = phi.output
            if len(phi.output_shape) > 2:
                out = layers.Flatten()(out)
            self.phi = models.Model(phi.inputs, out)
            self.phi_dim = self.phi.output_shape[-1]

        # build text generator
        with tf.variable_scope('TextGen'):
            self.gen = rnn_generator(
                static_dim=self.phi_dim,
                sequence_dim=words,
                length=length, 
                code_dim=code_dim,
                kind=rnn_type)
        
        # inherit as one model
        super(BaseModel, self).__init__(
            self.phi.inputs + self.gen.inputs, # + self.disc.inputs,
            self.phi.outputs + self.gen.outputs) # + self.disc.outputs)


    def G(self, x, y):
        '''generate next words given previous words and image'''
        return self.gen([self.phi(x), y])
    
    def D(self, y):
        '''discriminator logits on words'''
        return self.disc(y)
    
    def L_adv(self, y):
        '''discriminator loss against real/fake labels'''
        return tf.reduce_mean(self.D(y))
        
    def xent(self, x, y, sparse=True):
        '''logistic loss between generated and real next words'''
        pred = self.G(x, y)
        labels = tf.concat([y[:, 1:], y[:, -1:]], axis=1)
        if sparse:
            labels = tf.argmax(labels, axis=-1)
            return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=pred))
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=pred))
    
    def weight_norm(self, weights):
        '''L2 weight regularizer'''
        return tf.reduce_sum([tf.reduce_sum(w**2) for w in weights])

    def sample_recursively(self, x, batch_size=1, continuous=False):
        ''''''
        # predict the first word given a "start" token
        x = x[:batch_size]
        y_hat = np.zeros((1, self.length, self.words))
        y_hat[:, 0, self.words-2] = 1
        y_hat = tf.convert_to_tensor(y_hat, dtype=tf.float32)
        y_hat = tf.tile(y_hat, [batch_size, 1, 1])

        # append most likely word given previous word(s) to next round's input
        for t in range(self.length-1):
            g = self.G(x, y_hat)
            if continuous:
                pred = layers.Activation('softmax')(g)
            else:
                m = tf.reduce_max(g[:, t], axis=-1)
                m = tf.expand_dims(tf.expand_dims(m,  -1), -1)
                pred = tf.where(
                    tf.equal(g-m, 0), tf.ones_like(g), tf.zeros_like(g))
            pred = tf.concat(
                [tf.zeros((batch_size, t+1, self.words)), pred[:, t:t+1], 
                 tf.zeros((batch_size, self.length-(t+2), self.words))], axis=-2)
            y_hat += pred
        return y_hat
    
    def train(self, 
              train_record, 
              val_record,
              base_record,
              novel_record,
              output_path,
              batch_size=16,
              optimizer='adam',
              lr_init=1e-4, 
              decay_every=250000,
              clip_gradients=5.0,
              lambda_x=1,
              lambda_r=0.001,
              lambda_g=0,
              lambda_f=0,
              cnn_trainable=False,
              n_read_threads=5,
              n_stage_threads=5,
              capacity=16,
              epoch_size=10000,
              save_every=10000,
              validate_every=100,
              ckpt=None,
              cnn_ckpt='pretrained/inception_v3_weights.h5',
              inds_to_words_json='/home/paperspace/data/ms_coco/SOS_preproc_vocab-9413_threshold-5_length-20/inds_to_words.json',
             ):
        '''''' 
        
        with tf.variable_scope('BatchReader'):
            print('creating batch reader')

            # coordinator for tf queues
            coord =  tf.train.Coordinator()
            
            # read and preprocess training record
            x_train, y_train, cls_train, id_train = self.read_tfrecord(train_record, 
                batch_size=batch_size, capacity=capacity, n_threads=n_read_threads)
            
            # read and preprocess validation record
            x_val, y_val, cls_val, id_val = self.read_tfrecord(val_record, 
                batch_size=batch_size, capacity=1, n_threads=1)
            
            # base classes
            x_base, y_base, cls_base, id_base = self.read_tfrecord(base_record, 
                batch_size=batch_size, capacity=1, n_threads=1)
            
            # novel classes
            x_novel, y_novel, cls_novel, id_novel = self.read_tfrecord(novel_record, 
                batch_size=batch_size, capacity=1, n_threads=1)
        
        with tf.variable_scope('StagingArea'):
            print('creating staging area')

            # create training queue on GPU
            x_train, y_train = self.stage_data([x_train, y_train], 
                memory_gb=1, n_threads=n_stage_threads)
            
            # create validation queue on GPU
            x_val, y_val, x_base, y_base, x_novel, y_novel = self.stage_data(
                [x_val, y_val, x_base, y_base, x_novel, y_novel], 
                    memory_gb=0.1, n_threads=1)
            
            # global step and learning phase flag
            step = tf.Variable(0, name='step')
            update_step = tf.assign_add(step, 1)
            learning = backend.learning_phase()

        with tf.variable_scope('Optimizer'):
            print('creating optimizer')

            # get variables
            G_vars = self.gen.trainable_variables
            if cnn_trainable:
                G_vars += self.phi.trainable_variables
                
            # generated caption given previous words
            y_hat = self.G(x_train, y_train)
            
            # generated caption given only "start" token
            y_hat_cont = self.sample_recursively(x_train, continuous=True)
            
            # xentropy between real & generated next words
            L_x = self.xent(x_train, y_train, 
                            sparse=(False if lambda_g else True))
            
            # weight regularizer
            L_r = self.weight_norm(self.gen.trainable_variables)
            
            # feature vector regularizer
            L_f = tf.reduce_mean(tf.reduce_sum(self.phi(x_train)**2, axis=-1))
                
            # xentropy plus regularizers
            L_G = lambda_x*L_x + lambda_r*L_r + lambda_f*L_f
                
            # gradient penalty from arxiv.org/abs/1606.02819
            grad_norm = self.weight_norm(tf.gradients(L_G, G_vars))
            if lambda_g:
                L_G += lambda_g*grad_norm
                              
            # learning rate with periodic halving
            lr = lr_init*0.5**tf.floor(tf.cast(step, tf.float32)/decay_every)
                
            # optimizer: one of {"adam", "sgd"}
            if optimizer == 'adam':
                G_opt = tf.train.AdamOptimizer(lr, beta1=0.5)
                
            elif optimizer == 'sgd':
                G_opt = tf.train.GradientDescentOptimizer(lr)
                
            else:
                raise ValueError, 'optimizer can be "adam" or "sgd"'
            
            # clip by L2 norm, if specified
            G_grad = G_opt.compute_gradients(L_G, var_list=G_vars)
            if clip_gradients:
                G_grad = self.clip_gradients_by_norm(G_grad, clip_gradients)
            G_opt = G_opt.apply_gradients(G_grad)

        with tf.variable_scope('Summary'):
            print('creating summary')
            
            # validation loss terms
            L_x_val = self.xent(x_val, y_val)
            L_x_base = self.xent(x_base, y_base)
            L_x_novel = self.xent(x_novel, y_novel)
            L_f_val = tf.reduce_mean(tf.reduce_sum(self.phi(x_val)**2, axis=-1))
            
            # validation generated captions
            y_hat_val = self.G(x_val, y_val)
            y_hat_base = self.G(x_base, y_base)
            y_hat_novel = self.G(x_novel, y_novel)
            y_hat_samp_base = self.sample_recursively(x_base, continuous=False)
            y_hat_samp_novel = self.sample_recursively(x_novel, continuous=False)
            
            # decode predictions to words
            table = self.create_table(inds_to_words_json)
            real_base = self.postproc_caption(y_base, table)[0]
            real_novel = self.postproc_caption(y_novel, table)[0]
            fake_base = self.postproc_caption(y_hat_samp_base, table)[0]
            fake_novel = self.postproc_caption(y_hat_samp_novel, table)[0]
            txtfile = os.path.join(output_path, 'output.txt')            
            def write_func(lxv, rb, fb, rn, fn):
                lxv = float(lxv)
                rb, fb, rn, fn = [str(s) for s in [rb, fb, rn, fn]]
                st = '\n'.join([
                     'Loss: {}',
                     'Real base: "{}"',
                     'Fake base: "{}"',
                     'Real novel: "{}"',
                     'Fake novel: "{}"\n'])\
                     .format(lxv, rb, fb, rn, fn)\
                     .replace('SOS ', '').replace(' EOS', '')
                print(st, file=open(txtfile, 'a'))
                return 0
            printer = tf.py_func(write_func, 
                [L_x_val, real_base, fake_base, real_novel, fake_novel], Tout=tf.int64)

            # associate scalars with display names
            scalar_dict = {'L_x_train': L_x, 
                           'L_x_val': L_x_val,
                           'L_x_base': L_x_base,
                           'L_x_novel': L_x_novel,
                           'L_f': L_f_val,
                           'L_r': L_r,
                           'lr': lr, 
                           'printer': printer}
        # summarize 
        summary, writer = self.make_summary(
            output_path, scalar_dict=scalar_dict)

        # start the session 
        with tf.Session(graph=self.graph) as sess:
            
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
                
            print('starting threads')
            tf.train.start_queue_runners(sess=sess, coord=coord)
                
            # batch norm EMA updates
            if cnn_trainable:
                _x = layers.Input(tensor=x_train, shape=x_train.shape)
                super(BaseModel, self).__init__(
                    [_x] + self.gen.inputs,
                    [self.phi(_x)] + self.gen.outputs)
                G_opt = tf.group(G_opt, *self.updates)
                
            print('finalizing graph')
            self.graph.finalize()
            try:
                epoch = 1
                while True:                    
                    print('epoch: ' + str(epoch)); epoch += 1
                    time.sleep(0.2) # wait for print buffer to flush
                    for _ in tqdm.trange(epoch_size, disable=False):

                        # check for dead threads
                        if not coord.should_stop():

                            # update generator  
                            _, n = sess.run([G_opt, update_step], 
                                            {learning: True})
                            # validate
                            if n % validate_every == 1:
                                s = sess.run(summary, {learning: False})
                                writer.add_summary(s, n)
                                writer.flush()

                            # save checkpoint
                            if n % save_every == 0:
                                self.save_h5(output_path, n)
                        else:
                            raise IOError, 'queues closed'
                            
            # exit behaviour: request thread stop, then wait for 
            # them to recieve message before exiting session context
            except:
                coord.request_stop()
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
    