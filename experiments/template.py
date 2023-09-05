from abc import ABC
import os
import tensorflow as tf
from tqdm.auto import tqdm
from complexity_regularized_dcca.architecture.encoder import MVEncoder
from complexity_regularized_dcca.architecture.autoencoder import MVAutoencoder, ConvMVAutoencoder
from complexity_regularized_dcca.architecture.channelwise_autoencoder import ChannelwiseAutoencoder, ChannelwiseAutoencoder_paper
from complexity_regularized_dcca.algorithms.losses_metrics import MetricDict, get_rec_loss
from complexity_regularized_dcca.algorithms.tf_summary import TensorboardWriter
from complexity_regularized_dcca.algorithms.utils import logdir_update_from_params
from complexity_regularized_dcca.algorithms.losses_metrics import EmptyWatchdog


class Experiment(ABC):
    """
    Experiment meta class
    """
    def __init__(self, architecture, dataprovider, shared_dim, optimizer, log_dir, summary_writer, eval_epochs, watchdog, val_default_value=0.0, convergence_threshold=0.001):
        self.architecture = architecture
        self.dataprovider = dataprovider
        self.optimizer = optimizer
        self.summary_writer = self.create_summary_writer(summary_writer, log_dir)
        self.log_dir = self.summary_writer.dir
        self.shared_dim = shared_dim
        self.watchdog = watchdog
        self.moving_metrics = self.get_moving_metrics()
        self.eval_epochs = eval_epochs

        self.epoch = 1
        self.continue_training = True
        self.best_val = val_default_value
        self.best_val_view0 = val_default_value
        self.best_val_view1 = val_default_value
        self.best_val_avg = val_default_value
        # Convergence criteria
        self.loss_threshold = convergence_threshold
        self.prev_loss = 1e5
        self.prev_epoch = 0
        
    def get_moving_metrics(self):
        raise NotImplementedError

    def create_summary_writer(self, summary_writer, log_dir):
        # Define writer for tensorboard summary
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        return summary_writer(log_dir)

    def train_multiple_epochs(self, num_epochs):
        # Load training data once
        #training_data = self.dataprovider.training_data

        # Iterate over epochs
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            # Train one epoch
            self.train_single_epoch()

            if not self.continue_training:
                break
                
        self.save_weights('latest')

    def save_weights(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        else:
            save_path = self.log_dir

        self.architecture.save_weights(filepath=save_path)

    def load_weights_from_log(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
        else:
            save_path = self.log_dir

        self.architecture.load_weights(filepath=save_path)

    def load_best(self):
        self.architecture.load_weights(filepath=self.log_dir)

    def train_single_epoch(self):
        #for data in tqdm(training_data, desc='Batch', leave=False):
        for data in self.dataprovider.training_data:
            with tf.GradientTape() as tape:
                # Feed forward
                network_output = self.architecture(inputs=data, training=True)
                # Compute loss
                loss = self.compute_loss(network_output, data)

            # Compute gradients
            gradients = tape.gradient(loss, self.architecture.trainable_variables)
            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))

        # Write metric summary
        self.log_metrics()

        # Increase epoch counter
        self.epoch += 1

        if self.epoch % 100 == 0:
            self.prev_loss = self.prev_loss + 1e5
            next_loss = loss + 1e5
            if tf.abs(self.prev_loss - next_loss) < self.loss_threshold:
                #self.shared_dim += 1
                #self.architecture.update_num_shared_dim(self.shared_dim)
                self.continue_training = False
            self.prev_loss = loss
            self.prev_epoch = self.epoch

    def predict(self, data_to_predict):
        outputs = MetricDict()
        for data in data_to_predict:
            network_output = self.architecture(inputs=data, training=True)
            outputs.update(network_output)

        return outputs.output()

    def save(self):
        self.architecture.save(self.log_dir)

    def compute_loss(self, network_output, data):
        raise NotImplementedError

    def log_metrics(self):
        raise NotImplementedError


class DeepCCAExperiment(Experiment):
    """
    Experiment class for DeepCCA
    """
    def __init__(
            self,
            log_dir,
            encoder_config_v1,
            encoder_config_v2,
            dataprovider,
            shared_dim,
            lambda_rad,
            topk,
            max_perc=1,
            lambda_l1=0,
            lambda_l2=0,
            cca_reg=0,
            eval_epochs=10,
            optimizer=None,
            val_default_value=0.0,
            convergence_threshold=0.001,
            watchdog=None
    ):

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()

        if watchdog is None:
            watchdog = EmptyWatchdog()

        architecture = MVEncoder(
            encoder_config_v1=encoder_config_v1,
            encoder_config_v2=encoder_config_v2,
            cca_reg=cca_reg,
            num_shared_dim=shared_dim
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=encoder_config_v1[0][0],
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_rad=lambda_rad,
            topk=topk
        )

        super(DeepCCAExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=watchdog,
            val_default_value=val_default_value,
            convergence_threshold=convergence_threshold
        )

        # Dimensions and lambdas
        self.shared_dim = shared_dim

        self.lambda_rad = lambda_rad
        self.topk = topk
        self.max_perc = max_perc
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def compute_loss(self, network_output, data):
        # Compute CCA loss
        ccor = network_output['ccor']
        cca_loss = -1*tf.reduce_sum(ccor) / len(ccor)

        # Rademacher
        num_samples = self.dataprovider.num_samples
        dim_samples = self.dataprovider.dim_samples
        rad_loss = self.architecture.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
        rad_loss *= self.lambda_rad

        l1_loss = self.architecture.get_l1()
        l1_loss *= self.lambda_l1
        l2_loss = self.architecture.get_l2()
        l2_loss *= self.lambda_l2

        loss = cca_loss + rad_loss + l1_loss + l2_loss

        if self.epoch % 10 == 0:
            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (loss, 'Loss/Total'),
                    (cca_loss, 'Loss/CCA'),
                    (rad_loss, 'Loss/Rademacher'),
                    (l1_loss, 'Loss/L1'),
                    (l2_loss, 'Loss/L2'),
                    (self.shared_dim, 'MovingMean/Dimensions')
                ]
            )
        return loss


class DeepCCAEExperiment(Experiment):
    """
    Experiment class for DeepCCAE
    """
    def __init__(
            self,
            log_dir,
            encoder_config_v1,
            encoder_config_v2,
            decoder_config_v1,
            decoder_config_v2,
            dataprovider,
            shared_dim,
            lambda_rad,
            topk,
            max_perc=1,
            lambda_l1=0,
            lambda_l2=0,
            lambda_rec=1e-9,
            cca_reg=0,
            eval_epochs=10,
            optimizer=tf.keras.optimizers.Adam(),
            val_default_value=0.0,
            convergence_threshold=0.001
    ):

        architecture = MVAutoencoder(
            encoder_config_v1=encoder_config_v1,
            encoder_config_v2=encoder_config_v2,
            decoder_config_v1=decoder_config_v1,
            decoder_config_v2=decoder_config_v2,
            cca_reg=cca_reg,
            num_shared_dim=shared_dim
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=encoder_config_v1[0][0],
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_rad=lambda_rad,
            topk=topk,
            lambda_rec=lambda_rec
        )

        super(DeepCCAEExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=EmptyWatchdog(),
            val_default_value=val_default_value,
            convergence_threshold=convergence_threshold
        )

        # Dimensions and lambdas
        self.shared_dim = shared_dim

        self.lambda_rad = lambda_rad
        self.topk = topk
        self.max_perc = max_perc
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_rec = lambda_rec

    def compute_loss(self, network_output, data):
        # Compute CCA loss
        ccor = network_output['ccor']
        cca_loss = -1*tf.reduce_sum(ccor) / len(ccor)

        # Compute reconstruction loss
        rec_loss = get_rec_loss(
            tf.cast(data['nn_input_0'], tf.float32),
            tf.cast(data['nn_input_1'], tf.float32),
            network_output['reconst_view_0'],
            network_output['reconst_view_1']
        )
        rec_loss *= self.lambda_rec

        # Rademacher
        num_samples = self.dataprovider.num_samples
        dim_samples = self.dataprovider.dim_samples
        rad_loss = self.architecture.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
        rad_loss *= self.lambda_rad

        l1_loss = self.architecture.get_l1()
        l1_loss *= self.lambda_l1
        l2_loss = self.architecture.get_l2()
        l2_loss *= self.lambda_l2

        loss = cca_loss + rad_loss + l1_loss + l2_loss + rec_loss

        if self.epoch % 10 == 0:
            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (loss, 'Loss/Total'),
                    (cca_loss, 'Loss/CCA'),
                    (rad_loss, 'Loss/Rademacher'),
                    (rec_loss, 'Loss/Reconstruction'),
                    (l1_loss, 'Loss/L1'),
                    (l2_loss, 'Loss/L2'),
                    (self.shared_dim, 'MovingMean/Dimensions')
                ]
            )
        return loss


class ConvDeepCCAEExperiment(DeepCCAEExperiment):
    """
    Experiment class for DeepCCAE
    """
    def __init__(
            self,
            log_dir,
            encoder_config_v1,
            encoder_config_v2,
            decoder_config_v1,
            decoder_config_v2,
            dataprovider,
            shared_dim,
            lambda_rad,
            topk,
            max_perc=1,
            lambda_l1=0,
            lambda_l2=0,
            lambda_rec=1e-9,
            cca_reg=0,
            eval_epochs=10,
            optimizer=tf.keras.optimizers.Adam(),
            val_default_value=0.0,
            convergence_threshold=0.001
    ):

        architecture = ConvMVAutoencoder(
            encoder_config_v1=encoder_config_v1,
            encoder_config_v2=encoder_config_v2,
            decoder_config_v1=decoder_config_v1,
            decoder_config_v2=decoder_config_v2,
            cca_reg=cca_reg,
            num_shared_dim=shared_dim
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=-1,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_rad=lambda_rad,
            topk=topk,
            lambda_rec=lambda_rec
        )

        super(DeepCCAEExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=EmptyWatchdog(),
            val_default_value=val_default_value,
            convergence_threshold=convergence_threshold
        )

        # Dimensions and lambdas
        self.shared_dim = shared_dim

        self.lambda_rad = lambda_rad
        self.topk = topk
        self.max_perc = max_perc
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_rec = lambda_rec

class DeepCCAEChannelwiseExperiment(Experiment):
    """
    Experiment class for DeepCCAE
    """
    def __init__(
            self,
            log_dir,
            encoder_config,
            decoder_config,
            dataprovider,
            shared_dim,
            lambda_rad,
            topk,
            max_perc=1,
            lambda_l1=0,
            lambda_l2=0,
            lambda_rec=1e-9,
            cca_reg=0,
            eval_epochs=10,
            optimizer=tf.keras.optimizers.Adam(),
            val_default_value=0.0,
            convergence_threshold=0.001
    ):

        architecture = ChannelwiseAutoencoder(
            channel_encoder_config=encoder_config,
            channel_decoder_config=decoder_config,
            num_channels=dataprovider.num_channels,
            cca_reg=cca_reg,
            num_shared_dim=shared_dim
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=encoder_config[0][0],
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_rad=lambda_rad,
            topk=topk,
            lambda_rec=lambda_rec
        )

        super(DeepCCAEChannelwiseExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=EmptyWatchdog(),
            val_default_value=val_default_value,
            convergence_threshold=convergence_threshold
        )

        # Dimensions and lambdas
        self.shared_dim = shared_dim

        self.lambda_rad = lambda_rad
        self.topk = topk
        self.max_perc = max_perc
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_rec = lambda_rec

    def compute_loss(self, network_output, data):
        # Compute CCA loss
        ccor = network_output['ccor']
        cca_loss = -1*tf.reduce_sum(ccor) / len(ccor)

        # Compute reconstruction loss
        rec_loss = get_rec_loss(
            tf.cast(data['nn_input_0'], tf.float32),
            tf.cast(data['nn_input_1'], tf.float32),
            network_output['reconst_view_0'],
            network_output['reconst_view_1']
        )
        rec_loss *= self.lambda_rec

        # Rademacher
        num_samples = self.dataprovider.num_samples
        dim_samples = self.dataprovider.dim_samples
        rad_loss = self.architecture.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
        rad_loss *= self.lambda_rad

        l1_loss = self.architecture.get_l1()
        l1_loss *= self.lambda_l1
        l2_loss = self.architecture.get_l2()
        l2_loss *= self.lambda_l2

        loss = cca_loss + rad_loss + l1_loss + l2_loss + rec_loss

        if self.epoch % 10 == 0:
            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (loss, 'Loss/Total'),
                    (cca_loss, 'Loss/CCA'),
                    (rad_loss, 'Loss/Rademacher'),
                    (rec_loss, 'Loss/Reconstruction'),
                    (l1_loss, 'Loss/L1'),
                    (l2_loss, 'Loss/L2'),
                    (self.shared_dim, 'MovingMean/Dimensions')
                ]
            )
        return loss


class DeepCCAEPaperExperiment(Experiment):
    """
    Experiment class for DeepCCAE, the paper implementation
    """
    def __init__(
            self,
            log_dir,
            encoder_config,
            decoder_config,
            dataprovider,
            shared_dim,
            lambda_rad,
            topk,
            max_perc=1,
            lambda_l1=0,
            lambda_l2=0,
            lambda_rec=1e-9,
            eval_epochs=10,
            cca_reg=0,
            optimizer=tf.keras.optimizers.Adam(),
            val_default_value=0.0,
            convergence_threshold=0.001
    ):

        architecture = ChannelwiseAutoencoder_paper(
            channel_encoder_config=encoder_config,
            channel_decoder_config=decoder_config,
            num_channels=dataprovider.num_channels,
            cca_reg=cca_reg,
            num_shared_dim=shared_dim
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=encoder_config[0][0],
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_rad=lambda_rad,
            topk=topk,
            lambda_rec=lambda_rec
        )

        super(DeepCCAEPaperExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=EmptyWatchdog(),
            val_default_value=val_default_value,
            convergence_threshold=convergence_threshold
        )

        # Dimensions and lambdas
        self.shared_dim = shared_dim

        self.lambda_rad = lambda_rad
        self.topk = topk
        self.max_perc = max_perc
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_rec = lambda_rec

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

    def compute_loss(self, network_output, data):
        # Compute CCA loss
        tmp_1 = tf.square(tf.norm(self.U - tf.matmul(self.architecture.B_1, tf.transpose(network_output['latent_view_0'])), axis=0))
        tmp_2 = tf.square(tf.norm(self.U - tf.matmul(self.architecture.B_2, tf.transpose(network_output['latent_view_1'])), axis=0))
        cca_loss = tf.reduce_mean(tf.add(tmp_1,tmp_2))

        # Compute reconstruction loss
        rec_loss = get_rec_loss(
            tf.cast(data['nn_input_0'], tf.float32),
            tf.cast(data['nn_input_1'], tf.float32),
            network_output['reconst_view_0'],
            network_output['reconst_view_1']
        )
        rec_loss *= self.lambda_rec

        # Rademacher
        num_samples = self.dataprovider.num_samples
        dim_samples = self.dataprovider.dim_samples
        rad_loss = self.architecture.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
        rad_loss *= self.lambda_rad

        l1_loss = self.architecture.get_l1()
        l1_loss *= self.lambda_l1
        l2_loss = self.architecture.get_l2()
        l2_loss *= self.lambda_l2

        loss = cca_loss + rad_loss + l1_loss + l2_loss + rec_loss

        if self.epoch % 10 == 0:
            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (loss, 'Loss/Total'),
                    (cca_loss, 'Loss/CCA'),
                    (rad_loss, 'Loss/Rademacher'),
                    (rec_loss, 'Loss/Reconstruction'),
                    (l1_loss, 'Loss/L1'),
                    (l2_loss, 'Loss/L2'),
                    (self.shared_dim, 'MovingMean/Dimensions')
                ]
            )
        return loss

    def train_multiple_epochs(self, num_epochs):
        # Load training data once
        training_data = self.dataprovider.training_data

        # Initialize U
        for data in training_data:
            # Feed forward
            network_output = self.architecture(data)

        self.U = self.update_U(network_output=network_output)

        # Iterate over epochs
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            for _ in range(100):
                # Train for fixed U
                outputs = MetricDict()
                for data in training_data:
                    with tf.GradientTape() as tape:
                        # Feed forward
                        network_output = self.architecture(data)
                        # Compute loss
                        loss = self.compute_loss(network_output, data)

                    # Compute gradients
                    gradients = tape.gradient(loss, self.architecture.trainable_variables)
                    # Apply gradients
                    self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))

                    outputs.update(network_output)

                # Write metric summary
                _ = self.compute_metrics(outputs.output(), training=True)

                # Increase epoch counter
                self.epoch += 1

            self.U = self.update_U(network_output=network_output)

            if not self.continue_training:
                break