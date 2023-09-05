import tensorflow as tf

from complexity_regularized_dcca.architecture.layers import ChannelwiseEncoder, ChannelwiseDecoder
from complexity_regularized_dcca.algorithms.correlation import CCA


class ChannelwiseAutoencoder(tf.keras.Model):
    def __init__(
        self,
        channel_encoder_config,
        channel_decoder_config,
        num_channels,
        cca_reg,
        num_shared_dim,
        name="ChannelwiseAutoencoder",
        **kwargs
    ):
        super(ChannelwiseAutoencoder, self).__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.channel_encoder_config = channel_encoder_config
        self.channel_decoder_config = channel_decoder_config
        self.cca_reg = tf.Variable(cca_reg, dtype=tf.float32, trainable=False)
        self.num_shared_dim = tf.Variable(num_shared_dim, dtype=tf.int32, trainable=False)

        # Encoder
        self.encoder_v0 = ChannelwiseEncoder(
            channel_config=self.channel_encoder_config,
            num_channels=self.num_channels,
            view_ind=0,
        )
        self.encoder_v1 = ChannelwiseEncoder(
            channel_config=self.channel_encoder_config,
            num_channels=self.num_channels,
            view_ind=1,
        )
        # Decoder
        self.decoder_v0 = ChannelwiseDecoder(
            channel_config=self.channel_decoder_config,
            num_channels=self.num_channels,
            view_ind=0,
        )
        self.decoder_v1 = ChannelwiseDecoder(
            channel_config=self.channel_decoder_config,
            num_channels=self.num_channels,
            view_ind=1,
        )
        
        data_dim = self.num_channels
        
        self.mean_v0 = tf.Variable(tf.random.uniform((data_dim,)), trainable=False)
        self.mean_v1 = tf.Variable(tf.random.uniform((data_dim,)), trainable=False)
        
        self.B1 = tf.Variable(tf.random.uniform((self.num_shared_dim, data_dim)), trainable=False)
        self.B2 = tf.Variable(tf.random.uniform((self.num_shared_dim, data_dim)), trainable=False)

    @tf.function
    def get_rademacher_complexity(self, dim_samples, num_samples, topk, max_perc):
        radem_enc_0 = self.encoder_v0.get_rademacher_complexity(dim_samples, num_samples, topk, max_perc)
        radem_enc_1 = self.encoder_v1.get_rademacher_complexity(dim_samples, num_samples, topk, max_perc)
        return tf.math.reduce_prod([radem_enc_0, radem_enc_1])

    @tf.function
    def get_l1(self):
        l1_enc_0 = self.encoder_v0.get_l1()
        l1_enc_1 = self.encoder_v1.get_l1()
        l1_dec_0 = self.decoder_v0.get_l1()
        l1_dec_1 = self.decoder_v1.get_l1()
        return tf.math.reduce_sum([l1_enc_0, l1_enc_1, l1_dec_0, l1_dec_1])

    @tf.function
    def get_l2(self):
        l2_enc_0 = self.encoder_v0.get_l2()
        l2_enc_1 = self.encoder_v1.get_l2()
        l2_dec_0 = self.decoder_v0.get_l2()
        l2_dec_1 = self.decoder_v1.get_l2()
        return tf.math.reduce_sum([l2_enc_0, l2_enc_1, l2_dec_0, l2_dec_1])

    def call(self, inputs, training=True):
        # Check that input matches architecture
        # We expect the data to be of shape [num_samples, num_channels]
        inp_view_0 = inputs['nn_input_0']
        inp_view_1 = inputs['nn_input_1']

        # Compute latent variables
        latent_view_0 = self.encoder_v0(inp_view_0)
        latent_view_1 = self.encoder_v1(inp_view_1)
        # Reconstruct via decoder
        reconst_view_0 = self.decoder_v0(latent_view_0)
        reconst_view_1 = self.decoder_v1(latent_view_1)

        if training == True:
            B1, B2, epsilon, omega, ccor, mean_v0, mean_v1 = CCA(
                latent_view_0,
                latent_view_1,
                num_shared_dim=self.num_shared_dim,
                r1=self.cca_reg,
                r2=self.cca_reg
            )
            self.mean_v0.assign(mean_v0)
            self.mean_v1.assign(mean_v1)
            self.B1.assign(B1)
            self.B2.assign(B2)
        else:
            m = tf.cast(tf.shape(latent_view_0)[:1], tf.float32)
            v0_bar = tf.subtract(latent_view_0, self.mean_v0)
            v1_bar = tf.subtract(latent_view_1, self.mean_v1)
            epsilon = self.B1@tf.transpose(v0_bar)
            omega = self.B2@tf.transpose(v1_bar)
            diagonal = tf.linalg.diag_part(epsilon@tf.transpose(omega))
            ccor = diagonal / m
        
        return {
            'latent_view_0':latent_view_0, 
            'latent_view_1':latent_view_1, 
            'cca_view_0':tf.transpose(epsilon),
            'cca_view_1':tf.transpose(omega),
            'ccor':ccor,
            'reconst_view_0':reconst_view_0, 
            'reconst_view_1':reconst_view_1
        }


class ChannelwiseAutoencoder_paper(tf.keras.Model):
    def __init__(
        self,
        channel_encoder_config,
        channel_decoder_config,
        num_channels,
        cca_reg,
        num_shared_dim,
        name="ChannelwiseAutoencoder_paper",
        **kwargs
    ):
        super(ChannelwiseAutoencoder_paper, self).__init__(name=name, **kwargs)
        self.channel_encoder_config = channel_encoder_config
        self.channel_decoder_config = channel_decoder_config
        self.cca_reg = tf.Variable(cca_reg, dtype=tf.float32, trainable=False)
        self.num_shared_dim = tf.Variable(num_shared_dim, dtype=tf.int32, trainable=False)
        self.num_channels = num_channels
        # Encoder
        self.encoder_v0 = ChannelwiseEncoder(
            channel_config=channel_encoder_config, 
            num_channels=self.num_channels, 
            view_ind=0)
        self.encoder_v1 = ChannelwiseEncoder(
            channel_config=channel_encoder_config,
            num_channels=self.num_channels,
            view_ind=1)
        # Matrices for CCA
        self.B_1 = tf.Variable(tf.eye(self.num_channels), dtype=tf.float32)
        self.B_2 = tf.Variable(tf.eye(self.num_channels), dtype=tf.float32)
        # Decoder
        self.decoder_v0 = ChannelwiseDecoder(
            channel_config=channel_decoder_config,
            num_channels=self.num_channels,
            view_ind=0)
        self.decoder_v1 = ChannelwiseDecoder(
            channel_config=channel_decoder_config,
            num_channels=self.num_channels,
            view_ind=1)

        self.U = None

    @tf.function
    def get_rademacher_complexity(self, dim_samples, num_samples, topk, max_perc):
        radem_enc_0 = self.encoder_v0.get_rademacher_complexity(dim_samples, num_samples, topk, max_perc)
        radem_enc_1 = self.encoder_v1.get_rademacher_complexity(dim_samples, num_samples, topk, max_perc)
        return tf.math.reduce_prod([radem_enc_0, radem_enc_1])

    @tf.function
    def get_l1(self):
        l1_enc_0 = self.encoder_v0.get_l1()
        l1_enc_1 = self.encoder_v1.get_l1()
        l1_dec_0 = self.decoder_v0.get_l1()
        l1_dec_1 = self.decoder_v1.get_l1()
        return tf.math.reduce_sum([l1_enc_0, l1_enc_1, l1_dec_0, l1_dec_1])

    @tf.function
    def get_l2(self):
        l2_enc_0 = self.encoder_v0.get_l2()
        l2_enc_1 = self.encoder_v1.get_l2()
        l2_dec_0 = self.decoder_v0.get_l2()
        l2_dec_1 = self.decoder_v1.get_l2()
        return tf.math.reduce_sum([l2_enc_0, l2_enc_1, l2_dec_0, l2_dec_1])

    def update_U(self, network_output):
        num_samples = network_output['latent_view_0'].shape[0]
        I_t = tf.cast(num_samples, dtype=tf.float32)
        W = tf.eye(num_samples, num_samples) - tf.matmul(tf.ones([num_samples, 1]), tf.transpose(tf.ones([num_samples, 1])))/I_t

        Z_1 = tf.matmul(self.architecture.B_1, tf.transpose(network_output['latent_view_0']))
        Z_2 = tf.matmul(self.architecture.B_2, tf.transpose(network_output['latent_view_1']))

        Z = tf.add_n([Z_1, Z_2])
        U_tmp = tf.matmul(Z, W)

        # singular values - left singular vectors - right singular vectors
        D, P, Q = tf.linalg.svd(U_tmp, full_matrices=False, compute_uv=True)

        return tf.sqrt(I_t)*tf.matmul(P, tf.transpose(Q))

    def call(self, inputs):
        # Check that input matches architecture
        # We expect the data to be of shape [num_samples, num_channels]
        inp_view_0 = inputs['nn_input_0']
        inp_view_1 = inputs['nn_input_1']

        # Compute latent variables
        latent_view_0 = self.encoder_v0(inp_view_0)
        latent_view_1 = self.encoder_v1(inp_view_1)
        # Reconstruct via decoder
        reconst_view_0 = self.decoder_v0(latent_view_0)
        reconst_view_1 = self.decoder_v1(latent_view_1)

        return {
            'latent_view_0': latent_view_0,
            'latent_view_1': latent_view_1,
            'reconst_view_0': reconst_view_0,
            'reconst_view_1': reconst_view_1
        }
