import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC as SVM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from complexity_regularized_dcca.algorithms.losses_metrics import MetricDict, MovingMetric
from complexity_regularized_dcca.experiments.template import Experiment, DeepCCAExperiment, DeepCCAEExperiment

class XRMBExperiment(Experiment):
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
            acc_v0 = self.compute_svm_accuracy(view='view0', split='eval')
            self.moving_metrics['acc_v0'].update_window(acc_v0)
            smoothed_acc_v0 = self.moving_metrics['acc_v0'].get_metric()

            acc_v1 = self.compute_svm_accuracy(view='view1', split='eval')
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
                    (smoothed_acc_v0, 'AccuracySmoothed/View0'),
                    (acc_v1, 'Accuracy/View1'),
                    (acc_avg, 'Accuracy/Average'),
                    (smoothed_acc_v1, 'AccuracySmoothed/View1'),
                    (smoothed_acc_avg, 'AccuracySmoothed/Average'),
                ]
            )

    def compute_svm_accuracy(self, view='view0', split='eval'):
        assert split in ['eval', 'test']
        assert view in ['view0', 'view1']
        
        if split == 'test':
            data_splits_for_acc = self.dataprovider.test_data
        else:
            data_splits_for_acc = self.dataprovider.eval_data
        
        accs = []
        for split in data_splits_for_acc:
            outputs_met_train, labels_met_train = MetricDict(), MetricDict()
            for data in split['train']:
                outputs_met_train.update(self.architecture(data, training=False))
                labels_met_train.update(dict(labels=data['labels'].numpy()))

            netw_output_train = outputs_met_train.output()
            labels_train = labels_met_train.output()['labels']

            if view == 'view0':
                X_train = netw_output_train['latent_view_0']
            else:
                X_train = netw_output_train['latent_view_1']

            scaler = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
            X_train = scaler.transform(X_train)

            svm_model = SVM(random_state=333)
            svm_model.fit(X_train, labels_train)

            outputs_met_val, labels_met_val = MetricDict(), MetricDict()
            for data in split['val']:
                outputs_met_val.update(self.architecture(data, training=False))
                labels_met_val.update(dict(labels=data['labels'].numpy()))

            netw_output_val = outputs_met_val.output()
            labels_val = labels_met_val.output()['labels']

            if view == 'view0':
                X_val = netw_output_val['latent_view_0']
            else:
                X_val = netw_output_val['latent_view_1']

            X_val = scaler.transform(X_val)

            predictions = svm_model.predict(X_val)
            svm_acc = accuracy_score(labels_val, predictions)
            accs.append(svm_acc)

        return np.mean(accs)

class XRMBDeepCCAExperiment(DeepCCAExperiment, XRMBExperiment):
    pass


class XRMBDeepCCAEExperiment(DeepCCAEExperiment, XRMBExperiment):
    pass
