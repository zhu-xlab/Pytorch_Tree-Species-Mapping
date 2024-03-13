import numpy as np
import pandas as pd
import torch
from bases.data_loader_base import DataLoaderBase


class MyDataLoader(DataLoaderBase):
    def __init__(self, config):
        super().__init__(config)

        train_data_path = config.train_data_path
        val_data_path = config.val_data_path
        test_data_path = config.test_data_path
        w = config.w
        h = config.h
        if config.model == 'convlstm':
            input_channels = config.input_channels  # ---> ConvLSTM
            num_length = config.num_length
            num_channels = input_channels * num_length
        else:
            num_channels = config.num_channels  # ---> ResNet

        train_data = np.load(train_data_path, allow_pickle=True)
        val_data = np.load(val_data_path, allow_pickle=True)
        test_data = np.load(test_data_path, allow_pickle=True)

        # train data
        num_train = train_data.shape[0]
        x_train = np.zeros((num_train, w, h, num_channels), dtype=float)
        for i in range(0, num_train):
            x_train[i, :] = train_data[i, 0, 0]
        y_train = np.reshape(train_data[:, :, 1], num_train).astype(int)

        x_train_fillna = np.zeros((num_train, w * h, num_channels), dtype=float)
        for i in range(0, num_train):
            x_train_fillna[i, :] = np.reshape(x_train[i, :], (w * h, num_channels))
            x_train_fillna[i, :] = np.array(pd.DataFrame(x_train_fillna[i, :]).fillna(0))

        x_train_nn = np.zeros((num_train, w, h, num_channels), dtype=float)
        for i in range(0, num_train):
            x_train_nn[i, :] = np.reshape(x_train_fillna[i, :], (w, h, num_channels))
        x_train_nn = np.transpose(x_train_nn, (0, 3, 1, 2))  # (5852, 61, 5, 5)

        # validation data
        num_val = val_data.shape[0]
        x_val = np.zeros((num_val, w, h, num_channels), dtype=float)
        for i in range(0, num_val):
            x_val[i, :] = val_data[i, 0, 0]
        y_val = np.reshape(val_data[:, :, 1], num_val).astype(int)

        x_val_fillna = np.zeros((num_val, w * h, num_channels), dtype=float)
        for i in range(0, num_val):
            x_val_fillna[i, :] = np.reshape(x_val[i, :], (w * h, num_channels))
            x_val_fillna[i, :] = np.array(pd.DataFrame(x_val_fillna[i, :]).fillna(0))

        x_val_nn = np.zeros((num_val, w, h, num_channels), dtype=float)
        for i in range(0, num_val):
            x_val_nn[i, :] = np.reshape(x_val_fillna[i, :], (w, h, num_channels))
        x_val_nn = np.transpose(x_val_nn, (0, 3, 1, 2))  # (1463, 61, 5, 5)

        # test data
        num_test = test_data.shape[0]
        x_test = np.zeros((num_test, w, h, num_channels), dtype=float)
        for i in range(0, num_test):
            x_test[i, :] = test_data[i, 0, 0]
        y_test = np.reshape(test_data[:, :, 1], num_test).astype(int)

        x_test_fillna = np.zeros((num_test, w * h, num_channels), dtype=float)
        for i in range(0, num_test):
            x_test_fillna[i, :] = np.reshape(x_test[i, :], (w * h, num_channels))
            x_test_fillna[i, :] = np.array(pd.DataFrame(x_test_fillna[i, :]).fillna(0))

        x_test_nn = np.zeros((num_test, w, h, num_channels), dtype=float)
        for i in range(0, num_test):
            x_test_nn[i, :] = np.reshape(x_test_fillna[i, :], (w, h, num_channels))
        x_test_nn = np.transpose(x_test_nn, (0, 3, 1, 2))  # (1463, 61, 5, 5)

        self.X_train = torch.tensor(x_train_nn)
        self.y_train = torch.tensor(y_train).type(torch.LongTensor)
        self.X_val = torch.tensor(x_val_nn)
        self.y_val = torch.tensor(y_val).type(torch.LongTensor)
        self.X_test = torch.from_numpy(x_test_nn).float()
        self.y_test = torch.from_numpy(y_test).type(torch.LongTensor)
        self.mean = torch.mean(self.X_train, dim=(0, 2, 3))
        self.std = torch.std(self.X_train, dim=(0, 2, 3))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_mean_std(self):
        return self.mean, self.std
