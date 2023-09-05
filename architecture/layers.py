import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from complexity_regularized_dcca.algorithms.losses_metrics import compute_l1, compute_l2


class EncoderChannel(layers.Layer):
    def __init__(self, channel_config, view_ind, channel_ind, **kwargs):
        name = f'View_{view_ind}_Encoder_Channel_{channel_ind}'
        super(EncoderChannel, self).__init__(name=name, **kwargs)

        self.dense_layers = [
            layers.Dense(
                dim,
                activation=activ,
            ) for (dim, activ) in channel_config
        ]

    def get_single_layer_complexity(self, channel, topk, max_perc):
        layer_kernel, layer_bias = channel.trainable_variables[0], channel.trainable_variables[1]
        layer_norms = tf.norm(tf.concat([layer_kernel, layer_bias[None]], axis=0), ord=np.inf, axis=0)
        topk_max = int(channel.trainable_variables[0].shape[1]*max_perc)
        top_k_values, _ = tf.math.top_k(layer_norms, k=tf.math.minimum(topk_max,topk))
        layer_max = tf.reduce_sum(top_k_values)

        k_inp, k_layer = channel.trainable_variables[0].shape
        k_inp, k_layer = tf.cast(k_inp, dtype=tf.float32), tf.cast(k_layer, dtype=tf.float32)

        layer_compl = layer_max * tf.sqrt( k_inp + k_layer )

        return layer_compl

    def get_channel_complexity(self, topk, max_perc):
        return tf.math.reduce_prod([self.get_single_layer_complexity(layer, topk=topk, max_perc=max_perc) for layer in self.dense_layers])        

    def get_l1(self):
        return tf.math.reduce_sum([compute_l1(layer.trainable_variables[0]) for layer in self.dense_layers])

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for layer in self.dense_layers])

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)

        return x


class DecoderChannel(layers.Layer):
    def __init__(self, channel_config, view_ind, channel_ind, **kwargs):
        name = f'View_{view_ind}_Decoder_Channel_{channel_ind}'
        super(DecoderChannel, self).__init__(name=name, **kwargs)

        self.dense_layers = [
            layers.Dense(
                dim,
                activation=activ,
            ) for (dim, activ) in channel_config
        ]

    def get_l1(self):
        return tf.math.reduce_sum([compute_l1(layer.trainable_variables[0]) for layer in self.dense_layers])

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for layer in self.dense_layers])

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)

        return x


class ChannelwiseEncoder(tf.keras.Model):
    def __init__(self, channel_config, num_channels, view_ind, **kwargs):
        super(ChannelwiseEncoder, self).__init__(name=f'ChannelwiseEncoder_view_{view_ind}', **kwargs)
        self.config = channel_config
        self.num_channels = num_channels
        self.view_index = view_ind

        # Build channels
        self.channels = {str(i): EncoderChannel(
            channel_config=channel_config,
            view_ind=view_ind,
            channel_ind=i) for i in range(self.num_channels)
        }

    def get_rademacher_complexity(self, dim_samples, num_samples, topk, max_perc):
        layer_complexity = tf.math.reduce_prod([self.channels[key].get_channel_complexity(topk=topk, max_perc=max_perc) for key in self.channels.keys()])

        L = len(self.config)
        multiplicator = 2**L * tf.sqrt(tf.math.log(tf.cast(dim_samples, dtype=tf.float32)) / tf.cast(num_samples, dtype=tf.float32))

        return tf.math.multiply(layer_complexity, multiplicator)

    def get_l1(self):
        return tf.math.reduce_sum([self.channels[key].get_l1() for key in self.channels.keys()])

    def get_l2(self):
        return tf.math.reduce_sum([self.channels[key].get_l2() for key in self.channels.keys()])

    def call(self, inputs):
        return tf.concat(
            [self.channels[key](inputs[:, int(key), None]) for key in self.channels.keys()],
            axis=1)


class ChannelwiseDecoder(tf.keras.Model):
    def __init__(self, channel_config, num_channels, view_ind, **kwargs):
        super(ChannelwiseDecoder, self).__init__(name=f'ChannelwiseDecoder_view_{view_ind}', **kwargs)
        self.num_channels = num_channels
        self.view_index = view_ind

        # Build channels
        self.channels = {str(i): DecoderChannel(
            channel_config=channel_config,
            view_ind=view_ind,
            channel_ind=i) for i in range(num_channels)
        }

    def get_l1(self):
        return tf.math.reduce_sum([self.channels[key].get_l1() for key in self.channels.keys()])

    def get_l2(self):
        return tf.math.reduce_sum([self.channels[key].get_l2() for key in self.channels.keys()])

    def call(self, inputs):
        return tf.concat(
            [self.channels[key](inputs[:, int(key), None]) for key in self.channels.keys()],
            axis=1)


class Encoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(Encoder, self).__init__(name=f'Encoder_view_{view_ind}', **kwargs)
        self.config = config
        self.view_index = view_ind

        self.dense_layers = {
            str(i): layers.Dense(
                dim,
                activation=activ,
            ) for i, (dim, activ) in enumerate(self.config)
        }

    def get_denselayer_complexity(self, channel, topk, max_perc):
        
        layer_kernel, layer_bias = channel.trainable_variables[0], channel.trainable_variables[1]
        layer_norms = tf.norm(tf.concat([layer_kernel, layer_bias[None]], axis=0), ord=np.inf, axis=0)
        topk_max = int(channel.trainable_variables[0].shape[1]*max_perc)
        top_k_values, _ = tf.math.top_k(layer_norms, k=tf.math.minimum(topk_max,topk))
        layer_max = tf.reduce_sum(top_k_values)

        k_inp, k_layer = channel.trainable_variables[0].shape
        k_inp, k_layer = tf.cast(k_inp, dtype=tf.float32), tf.cast(k_layer, dtype=tf.float32)

        layer_compl = layer_max * tf.sqrt( k_inp + k_layer )

        return layer_compl

    def get_rademacher_complexity(self, dim_samples, num_samples, topk, max_perc):
        
        layer_complexity = tf.math.reduce_prod([self.get_denselayer_complexity(layer, topk=topk, max_perc=max_perc) for id, layer in self.dense_layers.items()])

        L = len(self.config)
        multiplicator = 2**L * tf.sqrt(tf.math.log(tf.cast(dim_samples, dtype=tf.float32)) / tf.cast(num_samples, dtype=tf.float32 ))

        return tf.math.multiply(layer_complexity, multiplicator)

    def get_l1(self):
        return tf.math.reduce_sum([compute_l1(layer.trainable_variables[0]) for id, layer in self.dense_layers.items()])

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for id, layer in self.dense_layers.items()])

    def call(self, inputs):
        x = inputs
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[str(i)](x)

        return x


class Decoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(Decoder, self).__init__(name=f'Decoder_view_{view_ind}', **kwargs)
        self.config = config
        self.view_index = view_ind

        self.dense_layers = [
            layers.Dense(
                dim,
                activation=activ,
            ) for (dim, activ) in self.config
        ]

    def get_l1(self):
        return tf.math.reduce_sum([compute_l1(layer.trainable_variables[0]) for layer in self.dense_layers])

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for layer in self.dense_layers])

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)

        return x
