import os
import tensorflow as tf
import pickle as pkl

from complexity_regularized_dcca.algorithms.losses_metrics import get_similarity_metric_v1, MetricDict, MovingMetric, EpochWatchdog
from complexity_regularized_dcca.algorithms.correlation import CCA
from complexity_regularized_dcca.experiments.template import Experiment, DeepCCAExperiment, DeepCCAEExperiment, DeepCCAEPaperExperiment, DeepCCAEChannelwiseExperiment

class SyntheticExperiment(Experiment):
    def get_moving_metrics(self):
        if isinstance(self.watchdog, EpochWatchdog):
            cor_movmetr = {
                'cor_' + str(num): MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean) 
                    for num in range(self.architecture.min_out_dim)
            }
        else:
            cor_movmetr = {
                'cor_' + str(num): MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean) 
                    for num in range(self.shared_dim)
            }

        simi_movmetr = {
            'sim_v0' : MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean),
            'sim_v1' : MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean),
            'sim_avg' : MovingMetric(window_length=50, history_length=100, fun=tf.math.reduce_mean),
        }

        return {**cor_movmetr, **simi_movmetr}
    
    def compute_similarity_scores(self):

        training_data = self.dataprovider.training_data
                
        outputs = MetricDict()
        for data in training_data:
            network_output = self.architecture(data, training=False)
            outputs.update(network_output)

        network_output = outputs.output()

        sim_v0 = get_similarity_metric_v1(
            S=tf.transpose(self.dataprovider.z_0)[:self.dataprovider.true_dim],
            U=tf.transpose(network_output['cca_view_0']),
            dims=self.dataprovider.true_dim
        )
        sim_v1 = get_similarity_metric_v1(
            S=tf.transpose(self.dataprovider.z_1)[:self.dataprovider.true_dim],
            U=tf.transpose(network_output['cca_view_1']),
            dims=self.dataprovider.true_dim
        )

        return sim_v0, sim_v1

    def log_metrics(self):
        self.watchdog.decrease_counter()

        if self.epoch % self.eval_epochs == 0:
            # Compute correlation values on training data
            training_outp = self.predict(self.dataprovider.training_data)
            ccor = training_outp['ccor']

            sim_v0, sim_v1 = self.compute_similarity_scores()
            sim_avg = (sim_v0+sim_v1)/2

            # Rademacher
            num_samples = self.dataprovider.num_samples
            dim_samples = self.dataprovider.dim_samples
            
            radem_total = self.architecture.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
            radem_enc_0 = self.architecture.encoder_v0.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
            radem_enc_1 = self.architecture.encoder_v1.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)

            l1 = self.architecture.get_l1()
            l2 = self.architecture.get_l2()

            self.moving_metrics['sim_v0'].update_window(sim_v0)
            smoothed_sim_v0 = self.moving_metrics['sim_v0'].get_metric()
            self.moving_metrics['sim_v1'].update_window(sim_v1)
            smoothed_sim_v1 = self.moving_metrics['sim_v1'].get_metric()
            self.moving_metrics['sim_avg'].update_window(sim_avg)
            smoothed_sim_avg = self.moving_metrics['sim_avg'].get_metric()

            if smoothed_sim_v0 < self.best_val_view0:
                self.best_val_view0 = smoothed_sim_v0
                self.save_weights(subdir='view0')

            if smoothed_sim_v1 < self.best_val_view1:
                self.best_val_view1 = smoothed_sim_v1
                self.save_weights(subdir='view1')

            if smoothed_sim_avg < self.best_val_avg:
                self.best_val_avg = smoothed_sim_avg
                self.save_weights(subdir='avg')

            for num in range(self.shared_dim):
                self.moving_metrics['cor_' + str(num)].update_window(ccor[num])

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch, 
                list_of_tuples=[
                    (ccor[num], 'Correlations/' + str(num)) for num in range(self.shared_dim)] +
                    [(self.moving_metrics['cor_' + str(num)].get_metric(),
                    'MovingMean/Correlation_' + str(num)) for num in range(self.shared_dim)] +
                    [(self.moving_metrics['sim_v0'].get_metric(), 'Metrics/Smooth similarity measure 1st view'),
                    (self.moving_metrics['sim_v1'].get_metric(), 'Metrics/Smooth similarity measure 2nd view'),
                    (self.moving_metrics['sim_avg'].get_metric(), 'Metrics/Smooth similarity measure average'),
                    (self.watchdog.compute(), 'MovingMean/Watchdog'),
                    (sim_v0, 'Metrics/Similarity measure 1st view'),
                    (sim_v1, 'Metrics/Similarity measure 2nd view'),
                    (radem_total, 'Rademacher/total'),
                    (radem_enc_0, 'Rademacher/enc_0'),
                    (radem_enc_1, 'Rademacher/enc_1'),
                    (l1, 'Regularization/L1'),
                    (l2, 'Regularization/L2')
                ]
            )

            if self.watchdog.check():
                if self.watchdog.reset():
                    #if self.shared_dim < self.architecture.min_out_dim:
                    #    self.shared_dim += 1
                    #    self.architecture.update_num_shared_dim(self.shared_dim)
                    pass
                else:
                    self.continue_training = False


class SynthDeepCCAExperiment(DeepCCAExperiment, SyntheticExperiment):
    pass


class SynthDeepCCAEExperiment(DeepCCAEExperiment, SyntheticExperiment):
    pass


class SynthDeepCCAEChannelwiseExperiment(DeepCCAEChannelwiseExperiment, SyntheticExperiment):
    pass


class SynthDeepCCAEChannelwisePaperExperiment(DeepCCAEPaperExperiment, SyntheticExperiment):
    def save_weights(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        else:
            save_path = self.log_dir

        self.architecture.save_weights(filepath=save_path)
        cca_file = os.path.join(save_path, 'CCA_variables.pkl')
        with open(cca_file, 'wb') as f:
            pkl.dump(dict(U=self.architecture.U), f)

    def load_weights_from_log(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
        else:
            save_path = self.log_dir

        cca_file = os.path.join(save_path, 'CCA_variables.pkl')
        with open(cca_file, 'rb') as f:
            CCA_dict = pkl.load(f)
            U = CCA_dict['U']

        self.architecture.load_weights(filepath=save_path)
        self.architecture.U = U

    def compute_similarity_scores(self):

        training_data = self.dataprovider.training_data
                
        outputs = MetricDict()
        for data in training_data:
            network_output = self.architecture(data)
            outputs.update(network_output)

        network_output = outputs.output()

        B0, B1, epsilon, omega, ccor, _, _ = CCA(
            network_output['latent_view_0'],
            network_output['latent_view_1'],
            self.shared_dim,
            self.architecture.cca_reg,
            self.architecture.cca_reg
        )

        sim_v0 = get_similarity_metric_v1(
            S=tf.transpose(self.dataprovider.z_0)[:self.shared_dim],
            U=epsilon,
            dims=self.dataprovider.true_dim
        )
        sim_v1 = get_similarity_metric_v1(
            S=tf.transpose(self.dataprovider.z_1)[:self.shared_dim],
            U=omega,
            dims=self.dataprovider.true_dim
        )

        return sim_v0, sim_v1

    def compute_metrics(self, network_output, training=True):
        self.watchdog.decrease_counter()
        metrics = dict()

        if self.epoch % self.eval_epochs == 0:
            # Compute all metrics
            B0, B1, epsilon, omega, ccor, _, _ = CCA(
                network_output['latent_view_0'],
                network_output['latent_view_1'],
                self.shared_dim,
                self.architecture.cca_reg,
                self.architecture.cca_reg
            )

            sim_v0 = get_similarity_metric_v1(
                S=tf.transpose(self.dataprovider.z_0)[:self.shared_dim],
                U=epsilon,
                dims=self.dataprovider.true_dim
            )
            sim_v1 = get_similarity_metric_v1(
                S=tf.transpose(self.dataprovider.z_1)[:self.shared_dim],
                U=omega,
                dims=self.dataprovider.true_dim
            )

            metrics['ccor'], metrics['sim_v0'], metrics['sim_v1'] = ccor, sim_v0, sim_v1

            # Rademacher
            num_samples = self.dataprovider.num_samples
            dim_samples = self.dataprovider.dim_samples
            
            radem_total = self.architecture.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
            radem_enc_0 = self.architecture.encoder_v0.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
            radem_enc_1 = self.architecture.encoder_v1.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)

            l1 = self.architecture.get_l1()
            l2 = self.architecture.get_l2()

            self.moving_metrics['sim_v0'].update_window(sim_v0)
            smoothed_sim_v0 = self.moving_metrics['sim_v0'].get_metric()
            self.moving_metrics['sim_v1'].update_window(sim_v1)
            smoothed_sim_v1 = self.moving_metrics['sim_v1'].get_metric()

            if smoothed_sim_v0 < self.best_val_view0:
                self.best_val_view0 = smoothed_sim_v0
                self.save_weights(subdir='view0')

            if smoothed_sim_v1 < self.best_val_view1:
                self.best_val_view1 = smoothed_sim_v1
                self.save_weights(subdir='view1')

            if training:
                for num in range(self.shared_dim):
                    self.moving_metrics['cor_' + str(num)].update_window(ccor[num])

                self.summary_writer.write_scalar_summary(
                    epoch=self.epoch, 
                    list_of_tuples=[
                        (ccor[num], 'Correlations/' + str(num)) for num in range(self.shared_dim)] +
                        [(self.moving_metrics['cor_' + str(num)].get_metric(),
                        'MovingMean/Correlation_' + str(num)) for num in range(self.shared_dim)] +
                        [(self.moving_metrics['sim_v0'].get_metric(), 'Metrics/Smooth similarity measure 1st view'),
                        (self.moving_metrics['sim_v1'].get_metric(), 'Metrics/Smooth similarity measure 2nd view'),
                        (self.watchdog.compute(), 'MovingMean/Watchdog'),
                        (sim_v0, 'Metrics/Similarity measure 1st view'),
                        (sim_v1, 'Metrics/Similarity measure 2nd view'),
                        (radem_total, 'Rademacher/total'),
                        (radem_enc_0, 'Rademacher/enc_0'),
                        (radem_enc_1, 'Rademacher/enc_1'),
                        (l1, 'Regularization/L1'),
                        (l2, 'Regularization/L2')
                    ]
                )

                if self.watchdog.check():
                    if self.watchdog.reset():
                        self.shared_dim += 1
                    else:
                        self.continue_training = False
                
        return metrics