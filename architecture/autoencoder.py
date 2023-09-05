import tensorflow as tf

from complexity_regularized_dcca.architecture.layers import Encoder, Decoder
from complexity_regularized_dcca.algorithms.correlation import CCA

class MVAutoencoder(tf.keras.Model):
    def __init__(
        self, 
        encoder_config_v1,
        encoder_config_v2,
        decoder_config_v1,
        decoder_config_v2,
        cca_reg,
        num_shared_dim,
        name="TwoViewAutoencoder",
        **kwargs
    ):
        super(MVAutoencoder, self).__init__(name=name, **kwargs)
        self.encoder_config_v1 = encoder_config_v1
        self.encoder_config_v2 = encoder_config_v2
        self.decoder_config_v1 = decoder_config_v1
        self.decoder_config_v2 = decoder_config_v2
        self.cca_reg = tf.Variable(cca_reg, dtype=tf.float32, trainable=False)
        self.num_shared_dim = tf.Variable(num_shared_dim, dtype=tf.int32, trainable=False)
        
        # Encoder
        self.encoder_v0 = Encoder(self.encoder_config_v1, view_ind=0)
        self.encoder_v1 = Encoder(self.encoder_config_v2, view_ind=1)
        # Decoder
        self.decoder_v0 = Decoder(self.decoder_config_v1, view_ind=0)
        self.decoder_v1 = Decoder(self.decoder_config_v2, view_ind=1)
        
        v1_data_dim = self.encoder_config_v1[-1][0]
        v2_data_dim = self.encoder_config_v2[-1][0]
        
        self.mean_v0 = tf.Variable(tf.random.uniform((v1_data_dim,)), trainable=False)
        self.mean_v1 = tf.Variable(tf.random.uniform((v2_data_dim,)), trainable=False)
        
        self.B1 = tf.Variable(tf.random.uniform((self.num_shared_dim, v1_data_dim)), trainable=False)
        self.B2 = tf.Variable(tf.random.uniform((self.num_shared_dim, v2_data_dim)), trainable=False)
      
    
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
