import os
import numpy as np
import tensorflow as tf
import scipy.io as scio
from sklearn.model_selection import KFold

class XRMBData():
    def __init__(self, batch_size, data_path = "/"):
        self.batch_size = batch_size
        
        single1_mat = scio.loadmat(os.path.join(data_path, "XRMBf2KALDI_window7_single1.mat"))
        single2_mat = scio.loadmat(os.path.join(data_path, "XRMBf2KALDI_window7_single2.mat"))
        
        train_labels = single2_mat['trainLabel'][:,0]
        eval_labels = single2_mat['tuneLabel'][:,0]
        test_labels = single2_mat['testLabel'][:,0]
        
        view_1_train = single1_mat['X1']
        view_2_train = single2_mat['X2']
        train_ids = single2_mat['trainID']            
        
        train_mask = np.ones_like(train_labels, dtype=bool)
        eval_mask = np.ones_like(eval_labels, dtype=bool)
        test_mask = np.ones_like(test_labels, dtype=bool)
        
        self.view1 = dict()
        self.view1['train'] = view_1_train[train_mask]
        self.view1['eval'] = single1_mat['XV1'][eval_mask]
        self.view1['test'] = single1_mat['XTe1'][test_mask]
        self.view1['val'] = np.concatenate([self.view1['eval'], self.view1['test']], axis=0)

        self.view2 = dict()
        self.view2['train'] = view_2_train[train_mask]
        self.view2['eval'] = single2_mat['XV2'][eval_mask]
        self.view2['test'] = single2_mat['XTe2'][test_mask]
        self.view2['val'] = np.concatenate([self.view2['eval'], self.view2['test']], axis=0)

        self.labels = dict()
        self.labels['train'] = train_labels[train_mask]
        self.labels['eval'] = single2_mat['tuneLabel'][eval_mask,0]
        self.labels['test'] = single2_mat['testLabel'][test_mask,0]
        self.labels['val'] = np.concatenate([self.labels['eval'], self.labels['test']], axis=0)

        self.ids = dict()
        self.ids['train'] = train_ids[train_mask,0]
        self.ids['eval'] = single2_mat['tuneID'][eval_mask,0]
        self.ids['test'] = single2_mat['testID'][test_mask,0]
        self.ids['val'] = np.concatenate([self.ids['eval'], self.ids['test']], axis=0)
               
        self.training_data = self.get_dataset(self.view1['train'], self.view2['train'], self.labels['train'])
        self.eval_data = self.get_split_datasets(self.view1['eval'], self.view2['eval'], self.labels['eval'], k=3)
        self.test_data = self.get_split_datasets(self.view1['test'], self.view2['test'], self.labels['test'], k=3)
        
        self.num_samples = self.view1['train'].shape[0]
        self.dim_samples = self.view1['train'].shape[1]

    def get_split_datasets(self, view1, view2, labels, k):

        # Create multiple datasets from numpy array in dict
        kf = KFold(n_splits=k)
        splits = list()
        for train_ids, val_ids in kf.split(np.arange(labels.shape[0])):
            # Create dataset for each split                
            splits.append(
                dict(
                    train = self.get_dataset(view1=view1[train_ids], view2=view2[train_ids], labels=labels[train_ids]),
                    val = self.get_dataset(view1=view1[val_ids], view2=view2[val_ids], labels=labels[val_ids])
                )
            )

        return splits
        
    def get_dataset(self, view1, view2, labels):
        # Create dataset from numpy array in dict
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                'nn_input_0': view1,
                'nn_input_1': view2,
                'labels': labels,
            }
        )
        
        num_samples = view1.shape[0]
        num_batches = int(np.ceil(num_samples/self.batch_size))
        
        # Batch
        dataset = dataset.batch(self.batch_size)
        
        return dataset.take(num_batches)
