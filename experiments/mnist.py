import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from complexity_regularized_dcca.algorithms.losses_metrics import MetricDict, MovingMetric
from complexity_regularized_dcca.algorithms.clustering import spectral_clustering_acc
from complexity_regularized_dcca.experiments.template import Experiment, DeepCCAExperiment, DeepCCAEExperiment, ConvDeepCCAEExperiment
from complexity_regularized_dcca.experiments.evaluation import Evaluation


class MNISTExperiment(Experiment):
    def get_moving_metrics(self):
        cor_movmetr = {
            'cor_' + str(num): MovingMetric(window_length=5, history_length=10, fun=tf.math.reduce_mean) 
                for num in range(self.shared_dim)
        }

        acc_movmetr = {
            'acc_v0': MovingMetric(window_length=5, history_length=10, fun=tf.math.reduce_mean),
            'acc_v1': MovingMetric(window_length=5, history_length=10, fun=tf.math.reduce_mean),
        }

        return {**cor_movmetr, **acc_movmetr}

    def log_metrics(self):
        self.watchdog.decrease_counter()

        if self.epoch % 10 == 0:
            # Compute correlation values on training data
            training_outp = self.predict(self.dataprovider.training_data)
            ccor = training_outp['ccor']

            # Rademacher
            num_samples = self.dataprovider.num_samples
            dim_samples = self.dataprovider.dim_samples

            radem_total = self.architecture.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
            radem_enc_0 = self.architecture.encoder_v0.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)
            radem_enc_1 = self.architecture.encoder_v1.get_rademacher_complexity(dim_samples, num_samples, self.topk, self.max_perc)

            l1 = self.architecture.get_l1()
            l2 = self.architecture.get_l2()

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[(ccor[i], 'Correlations/'+str(i)) for i in range(self.shared_dim)] +
                [(radem_total, 'Rademacher/total'),
                 (radem_enc_0, 'Rademacher/enc_0'),
                 (radem_enc_1, 'Rademacher/enc_1'),
                 (l1, 'Regularization/L1'),
                 (l2, 'Regularization/L2')
                 ]
            )

        if self.epoch % self.eval_epochs == 0:
            acc_v0 = self.compute_clustering_accuracy(view='view0', split='eval')
            self.moving_metrics['acc_v0'].update_window(acc_v0)
            smoothed_acc_v0 = self.moving_metrics['acc_v0'].get_metric()

            acc_v1 = self.compute_clustering_accuracy(view='view1', split='eval')
            self.moving_metrics['acc_v1'].update_window(acc_v1)
            smoothed_acc_v1 = self.moving_metrics['acc_v1'].get_metric()

            acc_avg = (acc_v0 + acc_v1) / 2
            smoothed_acc_avg = (smoothed_acc_v0 + smoothed_acc_v1)/2

            if smoothed_acc_v0 > self.best_val_view0:
                self.best_val_view0 = smoothed_acc_v0
                self.save_weights(subdir='view0')
            
            if smoothed_acc_v1 > self.best_val_view1:
                self.best_val_view1 = smoothed_acc_v1
                self.save_weights(subdir='view1')

            if smoothed_acc_avg > self.best_val_avg:
                self.best_val_avg = smoothed_acc_avg
                self.save_weights(subdir='avg')

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (acc_v0, 'Accuracy/View0'),
                    (acc_v1, 'Accuracy/View1'),
                    (acc_avg, 'Accuracy/Average'),
                    (smoothed_acc_v0, 'AccuracySmoothed/View0'),
                    (smoothed_acc_v1, 'AccuracySmoothed/View1'),
                    (smoothed_acc_avg, 'AccuracySmoothed/Average'),
                ]
            )

    def compute_clustering_accuracy(self, view='view0', split='eval'):
        assert view in ['view0', 'view1']
        assert split in ['eval', 'test']
        if split == 'eval':
            data_for_acc = self.dataprovider.eval_data
        else:
            data_for_acc = self.dataprovider.test_data

        outputs_met, labels_met = MetricDict(), MetricDict()
        for data in data_for_acc:
            outputs_met.update(self.architecture(inputs=data, training=False))
            labels_met.update(dict(labels=data['labels'].numpy()))

        netw_output = outputs_met.output()
        labels = labels_met.output()['labels']

        if view == 'view0':
            latent_repr = netw_output['latent_view_0']
        elif view == 'view1':
            latent_repr = netw_output['latent_view_1']

        return spectral_clustering_acc(
            data_points=latent_repr,
            labels=labels,
            num_classes=self.dataprovider.num_classes
        )
        

    def visualize_subspace(self, data_split='eval'):
        assert data_split in ['eval', 'test']
        if data_split == 'eval':
            data_for_vis = self.dataprovider.eval_data
        else:
            data_for_vis = self.dataprovider.test_data

        outputs_met = MetricDict()
        for data in data_for_vis:
            outputs_met.update(self.architecture(inputs=data, training=False))

        netw_output = outputs_met.output()

        embedding_0_net = TSNE(n_components=2, learning_rate='auto', init='random',
                               verbose=False, n_jobs=4).fit_transform(netw_output['cca_view_0'])

        embedding_1_net = TSNE(n_components=2, learning_rate='auto', init='random',
                               verbose=False, n_jobs=4).fit_transform(netw_output['cca_view_1'])

        fig, ax = plt.subplots(2, 1, figsize=(10, 20))
        ax[0].scatter(embedding_0_net[:, 0], embedding_0_net[:, 1],
                      c=self.dataprovider.view1_eval[1], cmap=plt.cm.tab10)
        ax[1].scatter(embedding_1_net[:, 0], embedding_1_net[:, 1],
                      c=self.dataprovider.view2_eval[1], cmap=plt.cm.tab10)


class MNISTDeepCCAExperiment(DeepCCAExperiment, MNISTExperiment):
    pass


class MNISTDeepCCAEExperiment(DeepCCAEExperiment, MNISTExperiment):
    pass


class MNISTConvDeepCCAEExperiment(ConvDeepCCAEExperiment, MNISTExperiment):
    pass


class MNISTEvaluation(Evaluation):
    def eval(self, model):
        metrics = dict()

        test_data = self.dataprovider.test_data

        outputs_met, labels_met = MetricDict(), MetricDict()
        for data in test_data:
            outputs_met.update(model(inputs=data, training=False))
            labels_met.update(dict(labels=data['labels'].numpy()))

        netw_output = outputs_met.output()
        labels = labels_met.output()['labels']

        metrics['acc_v0'] = spectral_clustering_acc(
            data_points=netw_output['cca_view_0'],
            labels=labels,
            num_classes=self.dataprovider.num_classes
        )
        metrics['acc_v1'] = spectral_clustering_acc(
            data_points=netw_output['cca_view_1'],
            labels=labels,
            num_classes=self.dataprovider.num_classes
        )
        metrics['acc_avg'] = (metrics['acc_v0'] + metrics['acc_v1']) / 2

        return metrics
