import os

from complexity_regularized_dcca.data.mnist import MNISTData
from complexity_regularized_dcca.experiments.mnist import MNISTDeepCCAExperiment


def main():
    root_dir = ''
    data = MNISTData.from_saved_data(root_dir)
    
    for l2 in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        for _ in range(5):        
            experiment = MNISTDeepCCAExperiment(
                log_dir=os.path.join(root_dir, 'dcca'), 
                encoder_config_v1=[(1536, 'sigmoid'), (1536, 'sigmoid'), (1536, 'sigmoid'), (15, None)],
                encoder_config_v2=[(1536, 'sigmoid'), (1536, 'sigmoid'), (1536, 'sigmoid'), (15, None)],
                dataprovider=data,
                shared_dim=15,
                lambda_rad=0,
                topk=10,
                max_perc=1,
                lambda_l1=0,
                lambda_l2=l2,
                cca_reg=1e-4,
                eval_epochs=100,
                val_default_value=0.0,
                convergence_threshold=0.000,
            )

            experiment.train_multiple_epochs(30000)

            experiment.save()
    
if __name__ == '__main__':
    main()
    
