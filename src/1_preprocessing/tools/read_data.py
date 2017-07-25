import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
import numpy as np
import tensorflow as tf
import scipy.io as sio


def load_data(acous_dir, artic_dir):
    acous_raw = sio.loadmat(acous_dir)['acoustics'][0, 0]
    artic_raw = sio.loadmat(artic_dir)['articulation'][0, 0]
    labels = list(acous_raw.dtype.fields.keys())

    acous_raw = acous_raw.tolist()
    artic_raw = artic_raw.tolist()
    return acous_raw, artic_raw, labels


def zero_padding(data):
    # data: list of (num_feature x num_seq)
    # padded: (num_example x num_seq x num_feature)
    length = [data[i].shape[1] for i in range(len(data))]
    max_length = max(length)
    padded = np.zeros((len(data), max_length, data[0].shape[0]))
    for i in range(len(data)):
        padded[i, :data[i].shape[1], :data[0].shape[0]] = np.transpose(data[i])
    return padded


def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def divide_timestep(data, time_step, max_time):
    out = []
    for i in range(max_time - time_step + 1):
        out.append(data[:, i:i + time_step, :])
    return out


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
