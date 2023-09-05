import os
from tqdm import tqdm
import pickle

from complexity_regularized_dcca.data.xrmb import XRMBData
from complexity_regularized_dcca.experiments.xrmb import XRMBDeepCCAExperiment

root_dir = ''
logfile = ''

xrmb_dataprovider = XRMBData(
    10000,
)

results_dict = dict()
for l2 in tqdm([1e-7, 1e-6, 1e-5, 1e-4], leave=False):
    results_dict[l2] = dict()
    for shared_dim in [90]:
        results_dict[l2][shared_dim] = dict()
        for network_dim in [1536]:
            results_dict[l2][shared_dim][network_dim] = dict(view0=dict(eval=list(), test=list()), view1=dict(eval=list(), test=list()))
            counter = 0
            while counter < 5:
                try:
                    exp = XRMBDeepCCAExperiment(
                        log_dir=os.path.join(root_dir, 'dcca'), 
                        encoder_config_v1=[(network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (shared_dim, None)],
                        encoder_config_v2=[(network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (shared_dim, None)],
                        dataprovider=xrmb_dataprovider,
                        shared_dim=shared_dim,
                        lambda_rad=0,
                        topk=30,
                        max_perc=1,
                        lambda_l1=0,
                        lambda_l2=l2,
                        cca_reg=1e-4,
                        eval_epochs=10,
                        val_default_value=0.0,
                        convergence_threshold=0.000,
                    )

                    exp.train_multiple_epochs(250)
                    
                    exp.save()
                    
                    counter = counter + 1
                    
                    # Load best setting for view0
                    exp.load_weights_from_log(subdir='view0')
                    results_dict[l2][shared_dim][network_dim]['view0']['eval'].append(
                        exp.compute_svm_accuracy(split='eval', view='view0')
                    )
                    results_dict[l2][shared_dim][network_dim]['view0']['test'].append(
                        exp.compute_svm_accuracy(split='test', view='view0')
                    )
                except Exception as e:
                    print(e)
            
                with open(logfile, 'wb') as f:
                    pickle.dump(results_dict, f)