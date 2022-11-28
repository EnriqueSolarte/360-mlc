import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


class GaussianModel_1D:
    def __init__(self, mean=None, sigma=None):
        self.mean = mean
        self.sigma = sigma
        self.samples = [(mean, sigma)]
        self.mean_hist = [mean]
        self.sigma_hist = [sigma]

    def add_measurement(self, mean, sigma):
        self.samples.append((mean, sigma))
        # for m, s in self.samples:
        #     self.update(m, s)
        self.update(mean, sigma)

    def update(self, mean, sigma):
        new_mean = (self.sigma**2) * mean + (sigma**2) * (self.mean)
        new_mean /= sigma**2 + self.sigma**2

        new_sigma = (sigma * self.sigma) ** 2
        new_sigma /= sigma**2 + self.sigma**2

        self.mean = new_mean
        self.sigma = np.sqrt(new_sigma)

        self.mean_hist.append(self.mean)
        self.sigma_hist.append(self.sigma)

    def sampling(self, samples=100):
        return np.random.normal(self.mean, self.sigma, samples)

    @staticmethod
    def visual_model(x, mean, sigma):
        fnt = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        fnt /= np.sum(fnt)
        # mask = 0.02
        # fnt[fnt > mask] = mask
        return fnt

    def eval(self, x):
        fnt = np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)
        fnt *= 1 / (self.sigma * np.sqrt(2 * np.pi))
        return fnt

    def force_positive(self):
        if self.mean < 0:
            self.mean = 2 * np.pi + self.mean

    def force_pi2pi_domain(self):
        if self.mean > np.pi:
            self.mean = self.mean - 2 * np.pi


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def get_2D_gaussian_kernel(shape, sigma=1):
    
    kernel_1 = np.linspace(-(shape[0] // 2), shape[0] // 2, shape[0])
    kernel_1 = dnorm(kernel_1, 0, sigma)
    
    kernel_2 = np.linspace(-(shape[1] // 2), shape[1] // 2, shape[1])
    kernel_2 = dnorm(kernel_2, 0, sigma)
    
    kernel_2D = np.outer(kernel_1.T, kernel_2.T)
 
    kernel_2D *= 1.0 / kernel_2D.sum()
    
    # plt.imshow(kernel_2D, interpolation='none',cmap='gray')
    # plt.title("Image")
    # plt.show()
    return kernel_2D

def apply_kernel(image_map, size=(512, 10), sigma=10):
    kernel = get_2D_gaussian_kernel(
        shape=size, sigma=sigma
    )
    filter_map = convolve(image_map, kernel, mode='constant')
    return filter_map / filter_map.max(axis=0)