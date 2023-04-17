from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.svm

from abc import ABC, abstractmethod


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
        # sliding window
        windows = variables_df.rolling(window=self.window_size)
        return windows, y

    def make_sklearn_dataset(self, test_split=0.2):
        """
        Train/test split in numpy format. current implementation uses polynomial coefficients as features
        :param test_split: ratio of test split
        :param transform: any transform to be applied here
        :return: train_X, train_y, test_X, test_y
        """
        X = []
        y = self.y
        # x coordinates for polynomial
        for window in self.windows:
            # only fit on full windows, not initial few
            if len(window) == self.window_size:
                features = window.to_numpy().ravel()
                X.append(features)
        X = X[:-1]  # remove last data point, can't predict that one.
        X = np.stack(X)
        # # train - test split
        # # 80 : 20
        length = len(self.y)
        train_length = int((1 - test_split) * length)
        test_length = length - train_length
        print(f"{train_length} samples in train set")
        print(f"{test_length} samples in test set")
        train_X, train_y = X[:train_length], y[:train_length]
        test_X, test_y = X[train_length:], y[train_length:]
        return train_X, train_y, test_X, test_y


class MBTATransform(ABC):
    @abstractmethod
    def fit_transform(self):
        return None

    @abstractmethod
    def transform(self):
        return None
