import tensorflow as tf
from tensorflow.keras import layers


class Discriminator():
    
    # TODO: Make it flexible to any input: (1) train sample shape, (2) number of layers, (3) kind of convolutions.
    def __init__():    
        self.model = tf.keras.Sequential()

        self.model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))

        return model
    
    def optimizer():
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def discriminator_loss(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss