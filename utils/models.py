from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
import image_queuing as iq
from seq2seq.models import Seq2Seq
from keras import layers, models, applications, backend
from tensorflow.contrib.staging import StagingArea

__author__ = 'Jonathan Kyl'

class BaseModel(models.Model):
    '''
    Wrapper for keras `Model` with additional methods shared across
    our models. 
    '''
    def read_data(self, input_path, img_shape, batch_size,
                  capacity=8, n_threads=4, descend='auto',
                  crop_aspect_ratio=None,):
        '''
        args:
            input_path: directory containing png or jpg files
            img_shape: desired (H, W, C) of output
            batch_size: no. of images per batch tensor
            capacity: max no. of batches to stage on cpu/gpu
            n_threads: no. of workers reading and preprocessing
            descend: explicitly descend into subdirectories
            crop_aspect_ratio: aspect ratio of center crop 
            
        returns:
            batch: 1-element list containing 
                   (BS, H, W, C) tensor of preprocessed images
        '''
        batch = iq.read_batch_multithread(input_path, 
            img_shape, batch_size, capacity=capacity, 
            n_threads=n_threads, descend=descend, 
            crop_aspect_ratios=crop_aspect_ratio)
        if type(batch) not in {tuple, list}:
            return [batch]
        return batch
        
    def stage_data(self, batches, capacity=8, device='gpu'):
        '''
        args:
            batches: list of tensors that are batches of preprocessed images
            capacity: max number of batch lists to stage at once
            device: whether to put the StagingArea on cpu or gpu
        
        returns:
            `get` tensor, `size` tensor, and `put` op
        '''
        with tf.device('/{}:0'.format(device)):
            preproc_batches = [self.preproc_img(b) for b in batches]
            dtypes = [t.dtype for t in preproc_batches]
            shapes = [t.get_shape() for t in preproc_batches]
            SA = StagingArea(dtypes, shapes=shapes, capacity=capacity)
            return SA.get(), SA.size(), SA.put(preproc_batches)
    
    def start_threads(self, sess, coord, put_op, n_stage_threads=4):
        '''
        args:
            sess: `tf.Session` for threads to run 
            coord: `tf.train.Coordinator` for reader threads
            put_op: `StagingArea`'s put operation for staging threads
            n_stage_threads: no. of staging threads
        
        returns:
            stage_stop, stage_threads
        '''
        tf.train.start_queue_runners(sess=sess, coord=coord)
        return iq.start_staging_threads(
            sess, put_op, n_threads=n_stage_threads)
            
    def make_summary(self, output_path, img_dict, scalar_dict, n_images=5):
        '''
        args:
            img_dict: dictionary of names and tensors of images to save
            scalar_dict: dictionary of names and tensors of scalars to save
            n_images: number of images to save, up to batch_size

        returns:
            `summary_op` and `summary_writer`
        '''
        summaries = []
        for k, v in img_dict.items():
            summaries.append(tf.summary.image(k, v, n_images))
        for k, v in scalar_dict.items():
            summaries.append(tf.summary.scalar(k, v))
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            output_path, graph=self.graph)
        return summary_op, summary_writer
    
    def save_h5(self, output_path, n, gen=True, ae=False):
        '''
        args:
            output_path: path in which to save .h5 files
            n: number of updates with which to name the .h5 files
        
        returns:
            none (writes to disk)
        '''

        self.save(os.path.join(
            output_path, 'ckpt_update-{}.h5'\
            .format(str(int(n)).zfill(10))))
        if gen:
            self.gen.save(os.path.join(
                output_path, 'gen_update-{}.h5'\
                .format(str(int(n)).zfill(10))))
        if ae:
            self.ae.save(os.path.join(
                output_path, 'ae_update-{}.h5'\
                .format(str(int(n)).zfill(10))))
        
    def save_grid(self, x, output_path, step):
        '''
        args:
            x: (BS, H, W, C) tensor to turn into a grid
            output_path: path to save images
            step: flat integer tensor of global step
        
        returns:
            `tf.py_func` tensor which writes grid to disk
        '''
        imgs = tf.unstack(x)
        n_imgs = len(imgs)
        side = int(np.ceil(float(n_imgs)**.5))
        imgs += [-tf.ones_like(imgs[0]) for _ in range(side**2 - n_imgs)]
        imgs = tf.concat(imgs, axis=1)
        imgs = tf.split(imgs, side, axis=1)
        imgs = tf.concat(imgs, axis=0)
        imgs = self.postproc_img(imgs)
        imgs = tf.image.encode_png(imgs)
        def write_fn(imgs, step):
            fname = os.path.join(output_path, 
                'grid_step-{}.png'.format(str(step).zfill(10)))
            with open(fname, 'w+') as f: f.write(imgs)
            return np.ones([], dtype=np.bool)
        return tf.py_func(write_fn, [imgs, step], tf.bool)
    
    def ae_filters(self, x, posterize=True, bnw=True):
        ''''''
        if bnw:
            x = tf.tile(tf.expand_dims(tf.reduce_sum(
                x*[0.299, 0.587, 0.114], axis=-1), axis=-1), [1, 1, 1, 3])
        if posterize:
            x = tf.where(tf.greater(x, tf.expand_dims(tf.expand_dims(
                    tf.reduce_mean(x, axis=[1, 2]), axis=1), axis=1)),
                         tf.ones_like(x), -tf.ones_like(x))
        return x
        
    def preproc_img(self, img):
        ''''''
        img = tf.cast(img, tf.float32)
        img /= 127.5
        img -= 1
        return img

    def postproc_img(self, img):
        ''''''
        img = tf.clip_by_value(img, -1, 1)
        img *= 127.5
        img += 127.5
        return tf.cast(img, tf.uint8)
    
def began_gen(size, chans, z_dim, n_filters=128, 
              n_blocks=4, n_per_block=2, concat=True, name=''):
    '''
    Simple image generator/decoder used in BEGAN. See arxiv.org/abs/1703.10717
    '''
    z = x = layers.Input(shape=(z_dim,),name=name+'z_input')
    
    proj_size = size // 2**(n_blocks-1)
    x = layers.Dense(proj_size*proj_size*n_filters)(x)
    h0 = x = layers.Reshape((proj_size, proj_size, n_filters))(x)
    
    for i in range(n_blocks):
        for j in range(n_per_block):
            x = layers.Conv2D(n_filters, 3, padding='same', activation='elu',
                              name=name+'conv_{}-{}'.format(i+1, j+1))(x)

        if i < n_blocks-1:
            x = layers.UpSampling2D(size=(2, 2))(x)
            if concat:
                h0 = layers.UpSampling2D(size=(2, 2))(h0)
                x = layers.concatenate([x, h0])

    x = layers.Conv2D(chans, 3, padding='same', name=name+'conv-output')(x)
    return models.Model(inputs=z, outputs=x)

def began_enc(size, chans, code_dim, n_filters=128, 
              n_blocks=4, n_per_block=2, name=''):
    '''
    Simple image encoder used in BEGAN. See arxiv.org/abs/1703.10717
    '''
    inp = x = layers.Input(shape=(size, size, chans),name=name+'img_input')
    x = layers.Conv2D(n_filters, 3, padding='same', activation='elu',
                      name=name+'conv_input')(x)
    for i in range(n_blocks):
        for j in range(n_per_block):
            if i == n_blocks - 1 or j < n_per_block - 1:
                strides = (1, 1)
                filters = n_filters*(i+1)
            elif i < n_blocks - 1:
                strides = (2, 2)
                filters = n_filters*(i+2)
                
            x = layers.Conv2D(filters, 3, padding='same',
                    activation='elu', strides=strides,
                    name=name+'conv_{}-{}'.format(i+1, j+1))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(code_dim, name=name+'fc')(x)
    return models.Model(inputs=inp, outputs=x)

def began_ae(size, chans, z_dim, n_filters=128, 
             n_blocks=4, n_per_block=2, concat=True, name=''):
    '''
    Simple autoencoder used in BEGAN. See arxiv.org/abs/1703.10717
    '''
    enc = began_enc(size, chans, z_dim, n_filters, 
                    n_blocks, n_per_block, name+'enc_')
    dec = began_gen(size, chans, z_dim, n_filters, 
                    n_blocks, n_per_block, concat, name+'dec_')
    img = enc.inputs[0]
    z_rec = enc(img)
    ae = dec(z_rec)
    return models.Model(inputs=img, outputs=[ae, z_rec])

def dcgan_gen(size, chans, z_dim, depth=1024, n_blocks=4, 
              selu=False, batch_norm=True, tanh=True, cmap=False, name=''):
    '''
    '''
    z = layers.Input(shape=(z_dim,), name=name+'z_input')
    side = size // 2**n_blocks 
    x = layers.Dense(side*side*depth, name=name+'fc')(z)
    x = layers.Reshape((side, side, depth))(x)
    for i in range(n_blocks):
        depth = max(128, depth//2) if i < n_blocks - 1 else chans
        if batch_norm:
            x = layers.BatchNormalization(epsilon=1e-5, 
                    momentum=0.9, name=name+'bn'+str(i+1))(x)
            x = layers.Activation('relu')(x)
        elif selu:
            x = layers.Lambda(SELU)(x)
        x = layers.Conv2DTranspose(depth, 5, strides=(2, 2), 
                padding='same', name=name+'conv'+str(i+1))(x)
    outputs = [x]
    if cmap:
        #js: attempt to smooth/clamp cmap output
        #regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        #outputs.append(layers.Conv2D(chans, 1, activity_regularizer = regularizer)(x))
        outputs.append(layers.Conv2D(chans, 1)(x), trainable=True)
        # doesn't have to be trainable jk
    if tanh:
        outputs = [layers.Activation('tanh')(x) for x in outputs]
    return models.Model(inputs=z, outputs=outputs)

def dcgan_gen_cond(size, chans, z_dim, c_dim, depth=1024, n_blocks=4, 
                   selu=False, batch_norm=True, tanh=True, cmap=False, name=''):
    '''
    '''
    z = layers.Input(shape=(z_dim,), name=name+'z_input')
    c = layers.Input(shape=(c_dim,), name=name+'c_input')
    x = layers.concatenate([z, c], axis=-1)
    side = size // 2**n_blocks 
    x = layers.Dense(side*side*depth, name=name+'fc')(x)
    x = layers.Reshape((side, side, depth))(x)
    for i in range(n_blocks):
        depth = max(128, depth//2) if i < n_blocks - 1 else chans
        if batch_norm:
            x = layers.BatchNormalization(epsilon=1e-5, 
                    momentum=0.9, name=name+'bn'+str(i+1))(x)
            x = layers.Activation('relu')(x)
        elif selu:
            x = layers.Lambda(SELU)(x)
        x = layers.Conv2DTranspose(depth, 5, strides=(2, 2), 
                padding='same', name=name+'conv'+str(i+1))(x)
    outputs = [x]
    if cmap:
        outputs.append(layers.Conv2D(chans, 1)(x))
    if tanh:
        outputs = [layers.Activation('tanh')(x) for x in outputs]
    return models.Model(inputs=[z, c], outputs=outputs)

def dcgan_disc(size, chans, z_dim, depth=1024, n_blocks=4, 
               selu=False, batch_norm=True, name=''):
    '''
    '''
    inp = x = layers.Input(shape=(size, size, chans), name=name+'img_input')
    for i in range(n_blocks):
        filters = max(128, depth//2**(n_blocks-(i+1)))
        x = layers.Conv2D(filters, 5, strides=(2, 2),
                padding='same', name=name+'conv'+str(i+1))(x)
        if i > 0 and batch_norm and not selu: 
            x = layers.BatchNormalization(epsilon=1e-5, 
                    momentum=0.9, name=name+'bn{}'+str(i+1))(x)
        if not selu:
            x = layers.LeakyReLU(alpha=0.2)(x)
        else:
            x = layers.Lambda(SELU)(x)
    side = size // 2**n_blocks
    x = layers.Reshape((side*side*depth,))(x)
    logits = layers.Dense(1, name=name+'fc')(x)
    return models.Model(inputs=inp, outputs=logits)

def SELU(x):
    ''''''
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * layers.ELU(alpha=alpha)(x)

def seq2seq(n_chars, length, code_dim, depth=4):
    ''''''
    return Seq2Seq(input_dim=n_chars, hidden_dim=code_dim, output_dim=n_chars,
                   input_length=length, output_length=length, depth=depth)

def lstm_decoder(code_dim, length, chars, activation='linear'):#lambda x: layers.LeakyReLU()(x)):
    inp = layers.Input(shape=(code_dim,), name='lstm_input')
    rep = layers.RepeatVector(length)(inp)
    out = layers.LSTM(chars, return_sequences=True, activation=activation)(rep)
    return models.Model(inp, out)

def mlp(dims, activation='relu'):
    inp = x = layers.Input(shape=[dims[0]])
    for d in dims:
        x = layers.Dense(d, activation=activation)(x)
    return models.Model(inp, x)
    
def cnn(kind='Xception', classes=None, weights=None, include_top=False, pooling='max'):
    ''''''
    if kind == 'xception':
        return applications.Xception(classes=classes, pooling=pooling,
            include_top=include_top, weights=weights)
    elif kind == 'mobilenet':
        return applications.MobileNet(classes=classes, pooling=pooling,
            include_top=include_top, weights=weights)

def variational_encoder(size, channels, depth, code_dim=64, n_blocks=4,
                        n_per_block=2, name='VE_'):
    ''''''
    inp = x = layers.Input(shape=(size, size, channels), name=name+'img_input')
    x = layers.Conv2D(depth, 3, padding='same', activation='elu',
                      name=name+'conv_input')(x)
    for i in range(n_blocks):
        for j in range(n_per_block):
            if i == n_blocks - 1 or j < n_per_block - 1:
                strides = (1, 1)
                filters = depth*(i+1)
            elif i < n_blocks - 1:
                strides = (2, 2)
                filters = depth*(i+2)
            x = layers.Lambda(SELU)(x)
            x = layers.Conv2D(filters, 3, padding='same',
                    activation=None, strides=strides,
                    name=name+'conv_{}-{}'.format(i+1, j+1))(x)

    x = layers.Flatten()(x)
    mu = layers.Dense(code_dim, name=name+'mu')(x)
    log_sigma = layers.Dense(code_dim, name=name+'log_sigma')(x)
    def sample(args):
        mu, log_sigma = args
        return mu + tf.exp(log_sigma)*tf.random_normal(shape=tf.shape(mu))
    code = layers.Lambda(sample)([mu, log_sigma])
    return models.Model(inputs=inp, outputs=[code, mu, log_sigma])

