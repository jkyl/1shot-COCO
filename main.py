from __future__ import print_function, division
import os
import tqdm
import tensorflow as tf
from utils.models import *

__author__ = 'Jonathan Kyl'

class TextGen(BaseModel):
    def __init__(self, length=128, chars=256):
        ''''''
        with tf.variable_scope('CNN'):
            self.phi = cnn(kind='mobilenet', include_top=False, weights=None)
            self.c_dim = self.phi.output_shape[-1]

        with tf.variable_scope('LSTM'):
            self.lstm = lstm_decoder(self.c_dim, length=length, chars=chars)

        super(BaseModel, self).__init__(
            self.phi.inputs + self.lstm.inputs,
            self.phi.outputs + self.lstm.outputs)

    def forward_pass(self, x):
        ''''''
        return self.lstm(self.phi(x))

    def train(self, train_record, val_record, output_path):
        ''''''        
        with tf.variable_scope('StagingArea'):
            print('creating staging area')

            coord =  tf.train.Coordinator()
            train_batch = self.read_data(train_record)
            get, SA_size, put_op = self.stage_data(train_batch)
            x_train, y_train, _ = get
            step = tf.Variable(0, name='step')
            update_step = tf.assign_add(step, 1)
                        
        with tf.variable_scope('Optimizer'):
            print('creating optimizer')

            y_hat = self.forward_pass(self, x)
            L = tf.reduce_mean((y_hat - y)**2)
            lr = lr_init * 0.5**tf.floor(
                tf.cast(step, tf.float32) / decay_every)
            opt = tf.train.AdamOptimizer(lr, beta1=0, beta2=0.9)\
                .minimize(L, var_list=self.trainable_variables)
            
        with tf.variable_scope('Summary'):
            print('creating summary')

            x_val, y_val, _ = self.read_data(val_record)
            y_hat_val = self.forward_pass(x_val)
            sx = tf.Variable(x_val, name='sample_x')
            sy = tf.Variable(y_val, name='sample_y')
            sy_hat = self.lstm(self.phi(sx))
            L_val = tf.reduce_mean((y_hat_val, y_val)**2)
            img_dict = {'x': self.postproc_img(sx)}
            scalar_dict = {'train_loss': L, 'val_loss': L_val, 'SA_size': SA_size}
            text_dict = {'y': sy, 'y_hat': sy_hat}
        summary_op, summary_writer = self.make_summary(
            output_path, img_dict, scalar_dict, text_dict)
        
        with tf.Session() as sess:
            print('starting threads')
            stage_stop, stage_threads = self.start_threads(
                sess, coord, put_op, n_stage_threads=4)
            print('initializing variables')
            sess.run(tf.global_variables_initializer())
            print('saving weights')
            self.save_h5(output_path, 0, ae=True)
            if ckpt: 
                print('loading weights from '+ckpt)
                self.load_weights(ckpt)
            print('finalizing graph')
            self.graph.finalize()
            try:
                epoch = 1
                while True:
                    print('epoch: ' + str(epoch))
                    for _ in tqdm.trange(epoch_size, disable=False):
                        if not (coord.should_stop() or stage_stop.is_set()):

                            # evaluate a training step
                            _, n = sess.run([opt, update_step])
                            
                            # write image and scalar summaries to disk
                            if n % summary_every == 1:
                                s = sess.run(summary_op)
                                summary_writer.add_summary(s, n)
                                summary_writer.flush()
                                
                            # save weights,  style
                            if n % ckpt_every == 0:
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
