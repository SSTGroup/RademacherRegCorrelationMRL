import io
from decimal import Decimal
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def logdir_update_from_params(
        log_dir,
        shared_dim,
        num_neurons,
        lambda_l1,
        lambda_l2,
        lambda_rec=None,
        lambda_rad=None,
        topk=None):
    """
    Update directory name with experiment parameters for easier comparison in tensorboard
    """
    shared_dim_str = f"_shdim_{shared_dim}"
    netw_size_str = f"_netwdim_{num_neurons}"
    l1_str = f"_l1_{Decimal(lambda_l1):.0E}"
    l2_str = f"_l2_{Decimal(lambda_l2):.0E}"
    log_dir = log_dir + netw_size_str + shared_dim_str + l1_str + l2_str

    # Add optional parameters
    if lambda_rec is not None:
        rec_str = f"_rec_{Decimal(lambda_rec):.0E}"
        log_dir = log_dir + rec_str

    if lambda_rad is not None:
        radem_str = f"_radem_{Decimal(lambda_rad):.0E}"
        log_dir = log_dir + radem_str

    if topk is not None:
        topk_str = f"_topk_{Decimal(topk)}"
        log_dir = log_dir + topk_str

    return log_dir
