from __future__ import print_function, division

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution


disable_eager_execution()



def batches(x, batch_size):
    """
    :x: shape (N, d)
    :return: Yields batches
    """
    x = x[np.random.permutation(len(x))][:int(len(x)/batch_size)*batch_size]
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size]

        
        
class DCGAN():
    def __init__(self, prior, img_shape=(28, 28, 1), load_from=None):
        self.img_shape = img_shape
        self.channels = img_shape[-1]
        self.latent_dim = prior.d
        self.prior = prior
        
          

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.G = self.generator
        self.D = self.discriminator
        
        
        if load_from:
          self.generator.load_weights(load_from+"_g.h5")
          self.discriminator.load_weights(load_from+"_d.h5")

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.prior.d))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.prior.d,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train, epochs, batch_size=128, callbacks=[], d_steps=1, g_steps=1):

        X_train = np.reshape(X_train, [-1]+list(self.img_shape))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            for batch_id, imgs in enumerate(batches(X_train, batch_size)):

                # Sample noise and generate a batch of new images
                for _ in range(d_steps):
                    noise = self.prior(batch_size)
                    gen_imgs = self.generator.predict(noise)
                    fakes_val = gen_imgs[:10]

                    # Train the discriminator (real classified as ones and generated as zeros)
                    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                #assert np.allclose(self.generator.predict(noise)[:10], fakes_val), "D update effects G"

                # ---------------------
                #  Train Generator
                # ---------------------
                for _ in range(g_steps):
                    # Train the generator (wants discriminator to mistake images as real)
                    y_val = self.discriminator.predict(imgs[:10])
                    g_loss = self.combined.train_on_batch(noise, valid)
                    #assert np.allclose(y_val, self.discriminator.predict(imgs[:10])), "G update effects D"

                # Plot the progress
                print ("\n%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                for callback, period in callbacks:
                    if (epoch+1) % period == 0:
                        callback()
    
    def save(self, file_prefix='dcgan'):
        self.generator.save_weights(f"{file_prefix}_g.h5")
        self.discriminator.save_weights(f"{file_prefix}_d.h5")


class Uniform():
    def __init__(self, d=100):
        self.d = d
        self.low = -1
        self.high = 1
    
    def __call__(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, self.d))


class Unconnected(Uniform):
    def __init__(self, d=100):
        self.d = d
        self.low = -1.1
        self.high = 1.1
        
    def __call__(self, n):
        z = np.random.uniform(low=-1, high=1, size=(n, self.d))
        z += np.sign(z)*0.1
        return z


class GradientInverser():
    def __init__(self, gan):
        self.gan = gan
        g = gan.generator
        z = Input([gan.prior.d])
        reals = Input(gan.img_shape)
        fakes = g(z)
        loss = K.square(g.outputs[0]-reals)
        gradient = K.gradients(loss, g.inputs[0])[0]
        self.iterate = K.function([g.inputs[0], reals], [loss, gradient])

    def invert(self, x):
        z = self.gan.prior(1000)
        fakes = self.gan.generator.predict(z).reshape([1000, -1])
        x_ = x.reshape([-1, fakes.shape[1]])
        z=z[cdist(fakes, x.reshape([-1, fakes.shape[1]])).argmin(0)]
        lr = z*0+0.1
        momentum = 0.5
        old_gradient = 0
        for i in range(40):
            loss, gradient = self.iterate([z, x])
            gradient = (1-momentum)*gradient + momentum*old_gradient
            old_gradient = gradient
            loss = np.sum(loss, axis=(1, 2, 3))[...,None]
            new_z = z - lr*gradient
            new_z = np.clip(new_z, -1, 1)
            new_loss, _ = self.iterate([new_z, x])
            new_loss = np.sum(new_loss, axis=(1, 2, 3))[...,None]
            z = np.where(new_loss<loss, new_z, z+np.random.normal(0, 0.01, z.shape))
            lr = np.where(new_loss>loss, lr/2, lr*1.1)*0.9
        return z
    
    def __call__(self, x):
        return self.invert(x)