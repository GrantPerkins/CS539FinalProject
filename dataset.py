from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.svm

import torch
from torch import nn
import torch.utils.data


class MBTADataset:
    """
    General dataset object for my final project
    """

    def __init__(self, window_size):
        """
        Create windows, load df, generate csv if need be. produce labels.
        :param window_size: size of windows (days)
        """
        self.window_size = window_size
        self.df_path = Path("mbta.csv")
        if not self.df_path.exists():
            self.make_csv()
        self.df = pd.read_csv(str(self.df_path))
        self.windows, self.y = self.make_windows()

    def make_csv(self):
        """
        Transform csv from one row per reading to one row per day
        :return:
        """
        df = pd.read_csv(r"C:\Users\gcper\OneDrive\WPI\ML\final\MBTA_Reliability.csv")

        # isolate rapid transit trains during peak hours
        df = df[(df["gtfs_route_desc"] == "Rapid Transit") & (df["mode_type"] == "Rail") & (
                df["peak_offpeak_ind"] == "PEAK")]
        df["service_date"] = df["service_date"].apply(lambda s: s.split()[0])
        routes = df["gtfs_route_id"].unique()

        # reshape df
        # make template df, one row for each date
        new_df = pd.DataFrame({"service_date": df.service_date.unique()})
        for route in routes:
            # make a column for each route's numerator and denominator
            route_df = df[df.gtfs_route_id == route][["service_date", "otp_numerator", "otp_denominator"]]
            route_df = route_df.rename(
                {"otp_numerator": f"{route}_Numerator", "otp_denominator": f"{route}_Denominator"},
                errors="raise", axis=1)
            new_df = new_df.merge(route_df, left_on="service_date", right_on="service_date",
                                  how="left")  # puts NaNs where data is missing (how=left)
        new_df.service_date = pd.to_datetime(new_df.service_date, format="%Y/%m/%d",
                                             errors="raise")  # convert service date to date type
        new_df = new_df.sort_values(by="service_date", ascending=True)  # sort df by date, oldest to newest
        new_df = new_df.rename({"service_date": "Date"}, axis=1)
        new_df = new_df.set_index("Date")

        # interpolate
        for col in new_df.columns:
            new_df[col].fillna(value=new_df[col].median(), inplace=True)

        # save final df
        new_df.to_csv(str(self.df_path))

    def make_windows(self):
        """
        Convert df into collection of `window_size` windows
        :return:
        """
        # separate input and labels
        red_line = self.df["Red_Numerator"] / self.df["Red_Denominator"]
        y = red_line.to_numpy()[self.window_size:]
        variables_df = self.df[[col for col in self.df.columns if col != "Date"]]
        for col in variables_df.columns:
            series = variables_df[col]
            variables_df[col] = (series - series.min()) / (series.max() - series.min())
        # sliding window
        windows = variables_df.rolling(window=self.window_size)
        return windows, y

    def make_sklearn_dataset(self, test_split=0.2):
        """
        Train/test split in numpy format. current implementation uses polynomial coefficients as features
        :param test_split: ratio of test split
        :return: train_X, train_y, test_X, test_y
        """
        X = []
        y = self.y
        # x coordinates for polynomial
        poly_x = np.arange(self.window_size)
        for window in self.windows:
            features = []
            # only fit on full windows, not initial few
            if len(window) == self.window_size:
                for col in window.columns:
                    # fit polynomial
                    coef = np.polyfit(poly_x, window[col].to_numpy(), deg=2)
                    coef = coef[:-1]  # ignore intercept
                    features.append(coef)
                features = np.stack(features).ravel()
                X.append(features)
        X = X[:-1]  # remove last data point, can't predict that one.
        X = np.stack(X)
        # # train - validation - test split
        # # 60 : 20 : 20
        length = len(self.y)
        train_length = int((1 - test_split) * length)
        test_length = length - train_length
        print(f"{train_length} samples in train set")
        print(f"{test_length} samples in test set")
        train_X, train_y = X[:train_length], y[:train_length]
        test_X, test_y = X[train_length:], y[train_length:]
        return train_X, train_y, test_X, test_y

    def make_torch_dataset(self, val_split=0.2, test_split=0.2, batch_size=8):
        """
        Train/val/test split, where every type is a tensor.
        :param val_split: validation ratio
        :param test_split: test ratio
        :param batch_size: size of batch
        :return: train_loader, val_loader, test_loader
        """
        X = []
        for window in self.windows:
            # only use full windows
            if len(window) == self.window_size:
                X.append(window.to_numpy().astype(np.float64))
        X = X[:-1]  # remove last data point, can't predict that one.
        X = np.stack(X)
        # convert to tensor
        X = torch.tensor(X).float().cpu()

        y = self.y
        y = y.reshape((-1, 1))
        y = torch.tensor(y).float().cpu()  # convert to tensor

        # train - validation - test split
        # 60 : 20 : 20
        length = len(y)
        train_length = int((1 - val_split - test_split) * length)
        val_length = int(val_split * length)
        test_length = int(test_split * length)
        print(f"{train_length} samples in train set")
        print(f"{val_length} samples in val set")
        print(f"{test_length} samples in test set")
        train_X, train_y = X[:train_length], y[:train_length]
        val_X, val_y = X[train_length:train_length + val_length], y[train_length:train_length + val_length]
        test_X, test_y = X[train_length + val_length:], y[train_length + val_length:]

        # quirky pytorch stuff here. create dataset object from tensors,
        # and then a method to access that data in batches
        train_ds = torch.utils.data.TensorDataset(train_X, train_y)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        val_ds = torch.utils.data.TensorDataset(val_X, val_y)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_ds = torch.utils.data.TensorDataset(test_X, test_y)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader
