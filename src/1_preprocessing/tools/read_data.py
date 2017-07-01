# Read articulation.pckl and acoustics.pckl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np
import scipy.io as sio
import pickle
import random
import csv

def load_data(acous_dir, artic_dir):
    acous_raw = sio.loadmat(acous_dir)['acoustics'][0,0]
    artic_raw = sio.loadmat(artic_dir)['articulation'][0,0]
    acous_raw = acous_raw.tolist()
    artic_raw = artic_raw.tolist()
    return acous_raw, artic_raw

def check_original_plot(x, y):
    '''
    e.g.
    x: acoustics[data_num], (dimension)x(time)
    y: articulation[data_num], (dimension)x(time)  
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    img = ax1.imshow(np.flipud(x), aspect='auto')
    ax1.set_ylabel('MFCC')
    pp, = ax1.plot([], [], 'k-')
    
    ax2 = fig.add_subplot(212)
    ax2.set_xlim((-60, 20))
    ax2.set_ylim((-30, 30))
    dots, = ax2.plot([], [], 'bo')
    ax2.set_ylabel('Articulation')
    
    def plot_frames(n_frame):
        dots.set_data(y[::2, n_frame], 
                      y[1::2, n_frame])
        pp.set_data(n_frame, [0, x.shape[0]])
        return ([dots, pp],)

    anim = animation.FuncAnimation(fig, plot_frames, 
                                   frames=x.shape[1], 
                                   interval=50, blit=False, 
                                   repeat=False)
    return HTML(anim.to_html5_video())