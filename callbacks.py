import numpy as np
from matplotlib import pyplot as plt



        
class PMetrics():
    def __init__(self, gan, x_test):
        self.gan = gan
        self.x_test = x_test
        self.z_test = gan.prior(len(x_test))
        self.p_fake, self.fake_std = [], []
        self.p_real, self.real_std = [], []
        
    def track(self, *ignored_args, **ignored_kwargs):
        y_fake = self.gan.D.predict(self.gan.G(self.z_test))
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
        fakes = self.gan.G(self.z)
        show(fakes)


class InterpolationCallback(EvolvingImageCallback):
    def __init__(self, gan):
        self.gan = gan
        z = gan.prior(2) # for plotting how the projection evolves
        b = z[0][None,...]
        a = z[1][None,...] - b
        x = np.arange(0, 1.05, 0.1)[...,None]
        self.z = a*x + b


class Evolving2DCallback():
    def __init__(self, gan):
        self.gan = gan
        self.z = gan.prior(100) # for plotting how the projection evolves
        self.z_at_epoch = [gan.G(self.z)]
    
    def track(self):
        self.z_at_epoch += [self.gan.G(self.z)]
        
    def plot(self):
        t = np.stack(self.z_at_epoch, axis=0)
        epochs = len(self.z_at_epoch)
        plt.clf()
        plt.plot(x[:,0], x[:,1], 'o')
        for p in range(t.shape[1]):
            plt.plot(t[:,p,0], t[:,p,1], alpha=0.5, color='orange')
        plt.plot(t[-1,:,0], t[-1,:,1], 'o', color='red')
        plt.title(f"Epoch {epochs}")
        plt.show()