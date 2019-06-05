import numpy as np
from matplotlib import pyplot as plt
from keras import layers
from gan import *

# Data generation
na = None

def show_2d(x_real, x_fake=None, title=None, space='x'):
    plt.plot(x_real[:, 0], x_real[:, 1], '.', label='Real')
    if x_fake is not None:
        plt.plot(x_fake[:, 0], x_fake[:, 1], '.', label='Fake')
        plt.legend()
    if title is not None:
        plt.title(title)
    plt.xlabel(space+'_1')
    plt.ylabel(space+'_2')
    plt.show()


def get_mode(center, n_per_mode, r):
    samples = np.random.normal(center, r, size=(3*n_per_mode, 2))
    dist = np.linalg.norm(samples-center[na,...], axis=1)
    return samples[dist<r][:n_per_mode]

def generate_2d(n_per_mode, radius=0.2):
    centers = [
        np.array([-1, -1]),
        np.array([-1, 1]),
        np.array([1, -1]),
        np.array([1, 1])
    ]
    return np.concatenate([get_mode(c, n_per_mode, radius) for c in centers], axis=0)


x = np.load('data/2d.npy')


uniform = Uniform()

def show_learned_distribution(prior, G):
    z = prior(500)
    show_2d(z, title='Prior', space='z')
    show_2d(G.predict(z), title='G(z)')
    

def make_grid():
    ticks = np.arange(-1.5, 1.5, 0.05)
    return np.dstack(np.meshgrid(ticks, ticks)).reshape(-1, 2)

def grid_labels():
    return range(0, 61, 6), [f"{i/50:.2g}" for i in range(-75, 76, 15)] 

grid = make_grid()

def colors_of(x):
    c = np.zeros([x.shape[0], 3])
    c[:, :2] = (x+1.5)/3
    return np.clip(c, 0, 1)

def color_plot(mapping, title=None):
    z = make_grid()
    x = mapping(z)
    
    fig, ax = plt.subplots(ncols=2)
    
    ax[0].imshow(colors_of(z).reshape(60, 60, 3))
    ax[0].set_title('C(Z)')
    ticks, labels = grid_labels()
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(labels)
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels(labels)
    ax[0].set_xlabel('z0')
    ax[0].set_ylabel('z1')
    
    ax[1].imshow(colors_of(x).reshape(60, 60, 3))
    ax[1].set_title('C(G(Z))')
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(labels)
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels(labels)
    ax[1].set_xlabel('z0')
    ax[1].set_ylabel('z1')
    
    if title is not None:
        fig.suptitle(title)
    
    plt.show()
  

def score_over_z(G, D, title=None):
    z = make_grid()
    dgz = D.predict(G.predict(z))
    plt.imshow(dgz.reshape(60, 60))
    ticks, labels = grid_labels()
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel('z0')
    plt.ylabel('z1')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
    


def d_landscape(D, x, title=None):
    # make grid
    x0_min = x[:,0].min() - 0.1
    x0_max = x[:,0].max() + 0.1

    x1_min = x[:,1].min() - 0.1
    x1_max = x[:,1].max() + 0.1

    x0_ticks = np.arange(x0_min, x0_max+0.0025, 0.05)

    x1_ticks = np.arange(x1_min, x1_max+0.0025, 0.05)[::-1]
    grid = np.dstack(np.meshgrid(x0_ticks, x1_ticks))

    # plot the landscape
    plt.imshow(D.predict(grid.reshape(-1, 2)).reshape(grid.shape[:-1]))

    # axis annotation
    xticks = [0, int(len(x0_ticks)/2), len(x0_ticks)-1]
    xlabels = [f"{x0_ticks[i]:.1f}" for i in xticks]

    yticks = [0, int(len(x1_ticks)/2), len(x1_ticks)-1]
    ylabels = [f"{x1_ticks[i]:.1f}" for i in yticks]

    plt.xticks(xticks, xlabels)
    plt.yticks(yticks, ylabels)

    plt.xlabel('x0')
    plt.ylabel('x1')
    
    if title is not None:
        plt.title(title)

    plt.colorbar()
    plt.show()


class DLandscapeCallback():
    def __init__(self, gan, data=x):
        self.gan = gan
        self.data = x
    
    def plot(self, *args, **kwargs):
        d_landscape(self.gan.D, np.concatenate([self.data,self.gan.G.predict(self.gan.prior(400))], axis=0))
 



