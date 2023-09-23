import numpy as np
import h5py, os, pickle, torch
import time
from framework.utilities import calculate_scalar, scale
import framework.config as config



class DataGenerator(object):
    def __init__(self, batch_size):

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(0)

        file_path = os.path.join(os.getcwd(), 'Dataset', 'Testing_mel.pickle')
        # print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.audio_ids, self.rates, self.event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.x = data['x']

        ##################################################################################
        file_path = os.path.join(os.getcwd(), 'Dataset', 'Testing_rms.pickle')
        # print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.x_rms = data['x']

        with open(os.path.join(os.getcwd(), 'Dataset', 'normalization.pickle'), 'rb') as f:
            data = pickle.load(f)
        self.mean, self.std = data['mel_mean'], data['mel_std']
        self.mean_rms, self.std_rms = data['rms_mean'], data['rms_std']


    def generate_data(self, data_type, max_iteration=None, only_SSC=False):
        audios_num = len(self.audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y_event = self.event_label[batch_audio_indexes]
            batch_x = self.transform(batch_x)
            if only_SSC:
                yield batch_x, batch_y_event

            else:
                batch_x_rms = self.x_rms[batch_audio_indexes]
                batch_y = self.rates[batch_audio_indexes]
                batch_x_rms = self.transform(batch_x_rms, mean=self.mean_rms, std=self.std_rms)

                yield batch_x, batch_x_rms, batch_y, batch_y_event

    def transform(self, x, mean=None, std=None):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        if mean==None:
            mean, std = self.mean, self.std
        return scale(x, mean, std)


