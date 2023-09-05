import tensorflow as tf
import numpy as np

class Dataprovider():

    def convert_to_tensors(self, list_of_vars):
        for var in list_of_vars:
            setattr(self, var, tf.convert_to_tensor(getattr(self, var), tf.float32))

    def generate_batches(self, data):
        # We expect the data to be in shape of (num_samples, num_channels)
        assert data.shape[0] == self.num_samples
        assert data.shape[1] == self.num_channels

        return np.reshape(
            data, 
            (self.num_batches, int(data.shape[0]/self.num_batches), data.shape[1]),
            order='C'
        )

    def nn_input(self, batch):
        raise NotImplementedError