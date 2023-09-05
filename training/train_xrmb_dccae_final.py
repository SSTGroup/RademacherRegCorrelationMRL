import os
from tqdm import tqdm
import pickle

from complexity_regularized_dcca.data.xrmb import XRMBData
from complexity_regularized_dcca.experiments.xrmb import XRMBDeepCCAEExperiment

root_dir = ''
logfile = ''

xrmb_dataprovider = XRMBData(
    10000,
)

results_dict = dict()
for l2 in tqdm([1e-7, 1e-6, 1e-5, 1e-4], leave=False):
    results_dict[l2] = dict()
    for lrec in tqdm([1e-6, 1e-8, 1e-10], leave=False):
        results_dict[l2][lrec] = dict()
        for lrad in [0]:
            results_dict[l2][lrec][lrad] = dict()
            for topk in tqdm([20], leave=False):
                results_dict[l2][lrec][lrad][topk] = dict()
                for shared_dim in [90]:
                    results_dict[l2][lrec][lrad][topk][shared_dim] = dict()
                    for network_dim in [1536]:
                        results_dict[l2][lrec][lrad][topk][shared_dim][network_dim] = dict(view0=dict(eval=list(), test=list()), view1=dict(eval=list(), test=list()))
                        counter = 0
                        while counter < 5:
                            try:
                                exp = XRMBDeepCCAEExperiment(
                                    log_dir=os.path.join(root_dir, 'dccae'),
                                    dataprovider=xrmb_dataprovider,
                                    shared_dim=shared_dim,
                                    encoder_config_v1=[(network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (shared_dim, None)],
                                    encoder_config_v2=[(network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (shared_dim, None)],
                                    decoder_config_v1=[(network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (273, None)],
                                    decoder_config_v2=[(network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (network_dim, 'sigmoid'), (112, None)],
                                    lambda_rec=lrec,
                                    lambda_rad=lrad,
                                    topk=topk,
                                    lambda_l2=l2, 
                                    cca_reg=1e-4,
                                    eval_epochs=10
                                )

                                exp.train_multiple_epochs(250)
                                
                                exp.save()
                                
                                counter = counter + 1
                                
                                # Load best setting for view0
                                exp.load_weights_from_log(subdir='view0')
                                results_dict[l2][lrec][lrad][topk][shared_dim][network_dim]['view0']['eval'].append(
                                    exp.compute_svm_accuracy(split='eval', view='view0')
                                )
                                results_dict[l2][lrec][lrad][topk][shared_dim][network_dim]['view0']['test'].append(
                                    exp.compute_svm_accuracy(split='test', view='view0')
                                )
                            except Exception as e:
                                print(e)

                            with open(logfile, 'wb') as f:
                                pickle.dump(results_dict, f)