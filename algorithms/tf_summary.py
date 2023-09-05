import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from complexity_regularized_dcca.algorithms.utils import plot_to_image

def create_channelwise_figures(x_1, y_1, x_2, y_2, label):
    num_channels = x_1.shape[0]
    fig, axes = plt.subplots(num_channels, 2, figsize=(10, 15))
    for c in range(num_channels):
        axes[c, 0].title.set_text(f'View {0} Channel {c}')
        axes[c, 0].scatter(x_1[c], tf.transpose(y_1)[c], label=label)

        axes[c, 1].title.set_text(f'View {1} Channel {c}')
        axes[c, 1].scatter(x_2[c], tf.transpose(y_2)[c], label=label)

    plt.tight_layout()

    return fig

class TensorboardWriter():
    def __init__(self, root_dir):
        folders = list()
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if os.path.isdir(file_path):
                folders.append(file_path)

        curr_number = 0
        while True:
            num_str = str(curr_number)
            if len(num_str) == 1:
                num_str = "0"+num_str

            folder = os.path.join(root_dir, num_str)
            if not os.path.exists(folder):
                break
            else:
                curr_number = curr_number + 1

        os.makedirs(folder)

        self.writer = tf.summary.create_file_writer(folder)
        self.dir = folder

    def write_scalar_summary(self, epoch, list_of_tuples):
        with self.writer.as_default():
            for tup in list_of_tuples:
                tf.summary.scalar(tup[1], tup[0], step=epoch)
        self.writer.flush()

    def write_image_summary(self, epoch, Az_1, Az_2, y_1, y_2, fy_1, fy_2, yhat_1, yhat_2):
        with self.writer.as_default():
            # Inverse learning plot
            inv_fig = create_channelwise_figures(Az_1, fy_1, Az_2, fy_2, r'$\mathrm{f}\circledast\mathrm{g}$')
            tf.summary.image("Inverse learning", plot_to_image(inv_fig), step=epoch)
            self.writer.flush()
            
            # Function f plot
            f_fig = create_channelwise_figures(y_1, fy_1, y_2, fy_2, r'$\mathrm{f}$')
            tf.summary.image("Network function", plot_to_image(f_fig), step=epoch)
            self.writer.flush()

            # Reconstruction plot
            rec_fig = create_channelwise_figures(y_1, yhat_1, y_2, yhat_2, None)
            tf.summary.image("Reconstruction", plot_to_image(rec_fig), step=epoch)
            self.writer.flush()
