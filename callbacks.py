import numpy as np
from matplotlib import pyplot as plt
from gan import GradientInverser
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow import keras


class PMetrics():
    def __init__(self, gan_, x_test):
        self.gan = gan_
        self.x_test = x_test
        self.z_test = self.gan.prior(len(x_test))
        self.p_fake, self.fake_std = [], []
        self.p_real, self.real_std = [], []
        
    def track(self, *ignored_args, **ignored_kwargs):
        y_fake = self.gan.D.predict(self.gan.G.predict(self.z_test))
        y_real = self.gan.D.predict(self.x_test)
        self.p_fake += [y_fake.mean()]
        self.p_real += [y_real.mean()]
        self.fake_std += [y_fake.std()]
        self.real_std += [y_real.std()]
        
    def plot(self, *ignored_args, **ignored_kwargs):
        epochs = range(len(self.p_fake))
        plt.errorbar(epochs, self.p_fake, self.fake_std, label='D(G(z))', color='blue')
        #plt.plot(epochs, np.array(self.p_fake) + np.array(self.fake_std), '--', color='blue')
        #plt.plot(epochs, np.array(self.p_fake) - np.array(self.fake_std), '--', color='blue')
        plt.errorbar(epochs, self.p_real, self.real_std, label='D(x)', color='orange', alpha=0.75)
        #plt.plot(epochs, np.array(self.p_real) + np.array(self.real_std), '--', color='orange')
        #plt.plot(epochs, np.array(self.p_real) - np.array(self.real_std), '--', color='orange')
        plt.title('P(real)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()



def show(images):
    images = np.reshape(images, [-1, 28, 28])
    n = images.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(n, 2),
                                    sharex=True, sharey=True)
    for i, img, ax in zip(range(n), images, axes):
        ax.imshow((img+1)/2, cmap='Greys')
        ax.axis('off')
    plt.show()
    

class EvolvingImageCallback():
    def __init__(self, gan):
        self.gan = gan
        self.z = gan.prior(10) # for plotting how the projection evolves
        
    def plot(self):
        fakes = self.gan.G.predict(self.z)
        self.show(fakes)

    def show(self, images):
        images = np.reshape(images, [-1, 28, 28])
        n = images.shape[0]
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(n, 2),
                                        sharex=True, sharey=True)
        for i, img, ax in zip(range(n), images, axes):
            ax.imshow((img+1)/2, cmap='Greys')
            ax.axis('off')
        plt.show()


class InterpolationCallback(EvolvingImageCallback):
    def __init__(self, gan):
        self.gan = gan
        z = gan.prior(2) # for plotting how the projection evolves
        b = z[0][None,...]
        a = z[1][None,...] - b
        x = np.arange(0, 1.05, 0.1)[...,None]
        self.z = a*x + b


class InterpolationCallback2D():
    def __init__(self, gan):
        self.gan = gan
        #r = np.arange(gan.prior.low, gan.prior.high+0.01, 0.1)
        r = np.arange(-1, 1+0.01, 0.05)
        grid = np.zeros([len(r), len(r), 2])
        grid[:,:,0] = r[None,...]
        grid[:,:,1] = r[...,None]
        self.grid = grid
        
    def plot(self):
        x_grid = self.gan.generator.predict(self.grid.reshape([-1, 2])).reshape(self.grid.shape)
        for i in range(len(x_grid)):
            plt.plot(x_grid[i,:,0], x_grid[i,:,1], color='blue', linewidth=1, alpha=0.5)
            plt.plot(x_grid[:,i,0], x_grid[:,i,1], color='blue', linewidth=1, alpha=0.5)
            plt.plot(x_grid[i,:,0], x_grid[i,:,1], '.', color='blue', linewidth=1)
            plt.plot(x_grid[:,i,0], x_grid[:,i,1], '.', color='blue', linewidth=1)
        plt.show()


class EvolvingCallback2D():
    def __init__(self, gan, x):
        self.gan = gan
        self.z = gan.prior(100) # for plotting how the projection evolves
        self.z_at_epoch = [gan.G.predict(self.z)]
        self.x = x
    
    def track(self):
        self.z_at_epoch += [self.gan.G.predict(self.z)]
        
    def plot(self):
        t = np.stack(self.z_at_epoch, axis=0)
        epochs = len(self.z_at_epoch)
        plt.clf()
        plt.plot(self.x[:,0], self.x[:,1], 'o')
        for p in range(t.shape[1]):
            plt.plot(t[:,p,0], t[:,p,1], alpha=0.5, color='orange')
        plt.plot(t[-1,:,0], t[-1,:,1], 'o', color='red')
        plt.title(f"Epoch {epochs}")
        plt.show()


class InverseDistributionCallback():
    def __init__(self, gan, x, title="", show_samples=show):
        self.gan = gan
        self.x = x
        self.inverse = GradientInverser(gan)
        self.title=title
        self.show_samples = show_samples
        
    def plot(self):
        z = self.gan.prior(1000)
        x_inv = self.inverse(self.x)
        plt.plot(z[:,0], z[:,1], "o", label='Prior')
        plt.plot(x_inv[:,0], x_inv[:,1], "o", label='Reconstruction')
        plt.title(self.title)
        plt.show()

        if self.show_samples:
            print('Reconstruction', self.title)
            self.show_samples(self.gan.G.predict(x_inv[:10]))
            print('Original', self.title)
            self.show_samples(self.x[:10])


class ModeCollapseObserver():
    def __init__(self, gan, mode_predictor):
        self.gan = gan
        self.mode_predictor = mode_predictor
      
    def __call__(self):
        z = self.gan.prior(1000)
        img = self.gan.generator.predict(z)
        modes = self.mode_predictor.predict(img).argmax(1)
        plt.hist(modes)
        plt.show()
        
        
def train_mnist_predictor():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=128,
              epochs=12,
              verbose=1,
              validation_data=(X_test, Y_test))
    model.save(f"{PROJECT_PATH}/mnist_predictor.h5")

    
def get_mnist_predictor():
    return load_model(f"{PROJECT_PATH}/mnist_predictor.h5")