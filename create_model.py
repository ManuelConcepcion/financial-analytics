# Libraries
import time
import warnings

import numpy as np
import pandas as pd

from typing import Callable, Optional
from scipy.stats.distributions import uniform, randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report

from finance_project_utils import var_cov_var, RSI, evaluate_model


# Main Code: given some parameters, run the corresponding code
class FinancialModel:
    def __init__(self,
                 data_root: Optional[str] = None,
                 raw_data_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 validation_dir: Optional[str] = None) -> None:

        self.data_root = "./data/" if data_root is None else data_root
        self.raw_data_dir = self.data_root+"raw/5m/" \
            if raw_data_dir is None else raw_data_dir
        self.processed_dir = self.data_root+"resampled/" \
            if processed_dir is None else processed_dir
        
        self.validation_dir = self.data_root+"validation/" \
            if validation_dir is None else validation_dir
        
        self.upper_limit = 0.0

    def _preprocess_data(self,
                         asset_df: pd.DataFrame,
                         tau: int,
                         alpha: float,
                         expected_return_function: Callable):
        df = asset_df.copy()

        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)

        lags = list(range(1, int(tau+1)))
        lag_columns = []  # List to store the names of lag columns

        # Calculate lagged returns based on the list of lags
        for lag in lags:
            lag_column_name = f'lag_{lag}'
            lag_columns.append(lag_column_name)

            df[lag_column_name] = np.log(df['Close'].shift(-lag) / df['Close'])

        df['ExpectedReturn'] = df[lag_columns].apply(expected_return_function,
                                                     axis=1)
        df.drop(lag_columns, axis=1, inplace=True)

        threshold = df['ExpectedReturn'].quantile((1 - alpha))
        self.upper_limit = threshold

        # Create a new column based on the condition
        df['buy_flag'] = df['ExpectedReturn'].ge(threshold).astype(int)

        return df

    @staticmethod
    def _add_variables(asset_df: pd.DataFrame):
        df = asset_df.copy()
        # List of variables to be added to the dataframe
        X_final_variables = ['Open_normalized', 'High_normalized',
                             'Close_normalized', 'Low_normalized',
                             'Open_MA_10', 'High_MA_10', 'Low_MA_10',
                             'Close_MA_10', 'Volume_MA_10', 'DayRange_MA_10',
                             'Open_MA_20', 'High_MA_20', 'Low_MA_20',
                             'Close_MA_20', 'Volume_MA_20', 'DayRange_MA_20',
                             'Open_MA_50', 'High_MA_50', 'Low_MA_50',
                             'Close_MA_50', 'Volume_MA_50', 'DayRange_MA_50',
                             'Open_MA_100', 'High_MA_100', 'Low_MA_100',
                             'Close_MA_100', 'Volume_MA_100',
                             'DayRange_MA_100',
                             'Open_MA_200', 'High_MA_200', 'Low_MA_200',
                             'Close_MA_200', 'Volume_MA_200',
                             'DayRange_MA_200',
                             '10_daysRange', '20_daysRange', '50_daysRange',
                             '100_daysRange', '200_daysRange',
                             'num_consec_days_up', 'num_consec_days_down',
                             'rsi_1', 'rsi_2', 'rsi_3', 'rsi_4', 'rsi_5',
                             'rsi_6', 'rsi_7', 'rsi_8', 'rsi_9', 'rsi_10',
                             'rsi_11', 'rsi_12', 'rsi_13', 'rsi_14', 'rsi_15',
                             'rsi_16', 'rsi_17', 'rsi_18', 'rsi_19', 'rsi_20',
                             'rets', 'rets_mean', 'rets_sigma', 'VaR']

        # Initialize each variable to 0 if it doesn't exist
        for each_var in X_final_variables:
            try:
                df[each_var]
            except KeyError:
                df[each_var] = 0

        # Calculate the daily range
        df['DayRange'] = df['High'] - df['Low']

        # Normalize open, high, close, and low relative to low
        df['Open_normalized'] = df['Open'] / df['Low'].iloc[0]
        df['High_normalized'] = df['High'] / df['Low'].iloc[0]
        df['Close_normalized'] = df['Close'] / df['Low'].iloc[0]
        df['Low_normalized'] = df['Low'] / df['Low'].iloc[0]

        # Calculate various moving averages and ranges
        print("computing 10, 20, 50, 100 and 200 days moving average for 2 "
              "variables: Open, High, Low, Close, Volume, DayRange")
        for lag in (10, 20, 50, 100, 200):
            df["Open_MA_" + str(lag)] = df["Open"].rolling(window=lag).mean()
            df["High_MA_" + str(lag)] = df["High"].rolling(window=lag).mean()
            df["Low_MA_" + str(lag)] = df["Low"].rolling(window=lag).mean()
            df["Close_MA_" + str(lag)] = df["Close"].rolling(window=lag).mean()
            df["Volume_MA_" + str(lag)] = \
                df["Volume"].rolling(window=lag).mean()
            df["DayRange_MA_" + str(lag)] = \
                df["DayRange"].rolling(window=lag).mean()

        # Calculate the 10, 20, 50, 100, and 200 days range
        print("and also, the 10, 20, 50, 100 and 200 days range")
        for lag in (10, 20, 50, 100, 200):
            df[str(lag) + "_daysRange"] = \
                (df["High"].rolling(window=lag).max()
                 - df["Low"].rolling(window=lag).min())

        # Calculate RSI for different periods
        for period in range(1, 21):
            df["rsi_"+str(period)] = RSI(df['Close'], period)

        # Calculate daily returns and Value at Risk (VaR)
        df["rets"] = df["Close"].pct_change()
        P = 1000   # 1,000 USD
        c = 0.99  # 99% confidence interval
        df["rets_mean"] = df["rets"].rolling(window=10).mean()
        df["rets_sigma"] = df["rets"].rolling(window=10).std()
        df["VaR"] = np.vectorize(var_cov_var)(P, c, df["rets_mean"],
                                              df["rets_sigma"])

        # Drop rows with NaN values after adding new variables
        df.dropna(inplace=True)

        # Additional calculations for growth and consecutive days
        df['growth'] = 0
        df.loc[df["Close"] > df["Close"].shift(1), 'growth'] = 1

        df['num_consec_days_up'] = \
            df["growth"].groupby(
                 (df["growth"] != df["growth"].shift()).cumsum()).cumcount()
        df['num_consec_days_up'] += 1
        df.loc[df['growth'] == 0.0, 'num_consec_days_up'] = 0

        df['num_consec_days_down'] = \
            df["growth"].groupby(
                 (df["growth"] != df["growth"].shift()).cumsum()).cumcount()
        df['num_consec_days_down'] += 1
        df.loc[df['growth'] == 1.0, 'num_consec_days_down'] = 0

        return df

    def _fit_model(self, asset_df: pd.DataFrame, param_grid: dict):
        df = asset_df.copy()

        target_colname = 'buy_flag'
        numeric_features = ['Open_normalized', 'High_normalized',
                            'Close_normalized', 'Low_normalized',
                            'Open_MA_10', 'High_MA_10', 'Low_MA_10',
                            'Close_MA_10', 'Volume_MA_10', 'DayRange_MA_10',
                            'Open_MA_20', 'High_MA_20', 'Low_MA_20',
                            'Close_MA_20', 'Volume_MA_20', 'DayRange_MA_20',
                            'Open_MA_50', 'High_MA_50', 'Low_MA_50',
                            'Close_MA_50', 'Volume_MA_50', 'DayRange_MA_50',
                            'Open_MA_100', 'High_MA_100', 'Low_MA_100',
                            'Close_MA_100', 'Volume_MA_100',
                            'DayRange_MA_100',
                            'Open_MA_200', 'High_MA_200', 'Low_MA_200',
                            'Close_MA_200', 'Volume_MA_200',
                            'DayRange_MA_200',
                            '10_daysRange', '20_daysRange', '50_daysRange',
                            '100_daysRange', '200_daysRange',
                            'num_consec_days_up', 'num_consec_days_down',
                            'rsi_1', 'rsi_2', 'rsi_3', 'rsi_4', 'rsi_5',
                            'rsi_6', 'rsi_7', 'rsi_8', 'rsi_9', 'rsi_10',
                            'rsi_11', 'rsi_12', 'rsi_13', 'rsi_14', 'rsi_15',
                            'rsi_16', 'rsi_17', 'rsi_18', 'rsi_19', 'rsi_20',
                            'rets', 'rets_mean', 'rets_sigma', 'VaR']
        categorical_features = ['growth']

        X_categorical = df[categorical_features]

        X = pd.concat([df[numeric_features], X_categorical], axis=1)
        y = df[target_colname]

        ts_cv = TimeSeriesSplit(n_splits=5)

        search_cv = RandomizedSearchCV(
            estimator=RandomForestClassifier(class_weight="balanced"),
            param_distributions=param_grid,
            n_iter=5,
            scoring='f1',
            cv=ts_cv,
            verbose=2
        )

        search_cv.fit(X=X, y=y)
        best_params = search_cv.best_params_

        final_model = RandomForestClassifier(
            ccp_alpha=best_params['ccp_alpha'],
            min_impurity_decrease=best_params['min_impurity_decrease'],
            min_samples_split=best_params['min_samples_split'],
            n_estimators=best_params['n_estimators']
        )

        final_model.fit(X=X, y=y)

        print("Model results on training dataset:")
        evaluate_model(final_model, X=X, y=y, cv=ts_cv)

        return final_model

    def _evaluate_model(self,
                        validation_df,
                        final_model,
                        alpha,
                        tau,
                        er_function) -> dict[str, float]:
        validation_df = self._preprocess_data(validation_df, alpha=alpha, tau=tau,
                                              expected_return_function=er_function)
        validation_df = self._add_variables(validation_df)

        # Define the validation data
        categorical_features = ['growth']
        numeric_features = ['Open_normalized', 'High_normalized',
                            'Close_normalized', 'Low_normalized',
                            'Open_MA_10', 'High_MA_10', 'Low_MA_10',
                            'Close_MA_10', 'Volume_MA_10', 'DayRange_MA_10',
                            'Open_MA_20', 'High_MA_20', 'Low_MA_20',
                            'Close_MA_20', 'Volume_MA_20', 'DayRange_MA_20',
                            'Open_MA_50', 'High_MA_50', 'Low_MA_50',
                            'Close_MA_50', 'Volume_MA_50', 'DayRange_MA_50',
                            'Open_MA_100', 'High_MA_100', 'Low_MA_100',
                            'Close_MA_100', 'Volume_MA_100',
                            'DayRange_MA_100',
                            'Open_MA_200', 'High_MA_200', 'Low_MA_200',
                            'Close_MA_200', 'Volume_MA_200',
                            'DayRange_MA_200',
                            '10_daysRange', '20_daysRange', '50_daysRange',
                            '100_daysRange', '200_daysRange',
                            'num_consec_days_up', 'num_consec_days_down',
                            'rsi_1', 'rsi_2', 'rsi_3', 'rsi_4', 'rsi_5',
                            'rsi_6', 'rsi_7', 'rsi_8', 'rsi_9', 'rsi_10',
                            'rsi_11', 'rsi_12', 'rsi_13', 'rsi_14', 'rsi_15',
                            'rsi_16', 'rsi_17', 'rsi_18', 'rsi_19', 'rsi_20',
                            'rets', 'rets_mean', 'rets_sigma', 'VaR']

        # Extract the categorical features from the validation set
        X_categorical_validation = validation_df[categorical_features]

        # Concatenate the numeric features with the categorical features
        X_validation = pd.concat([validation_df[numeric_features],
                                  X_categorical_validation], axis=1)

        predictions = final_model.predict(X_validation)

        # Evaluate the model
        print("Comparing the model to optimal strategy:")
        print(classification_report(validation_df['buy_flag'], predictions))

        validation_df['prediction'] = predictions

        # Create empty columns for position, entry, and PredictedBuyFlag
        validation_df['position'] = 0
        validation_df['entry'] = 0
        validation_df['PredictedBuyFlag'] = 0
        # Calculate the strategy
        for i in range(len(validation_df)):
            # Buy condition based on threshold
            if validation_df['Close'].iloc[i] >= self.upper_limit:
                validation_df['position'].iloc[i] = 1
                validation_df['PredictedBuyFlag'].iloc[i] = 1

            # Sell condition after tau moments
            elif validation_df['num_consec_days_up'].iloc[i] >= tau:
                validation_df['position'].iloc[i] = -1

            # Entry condition based on model predictions
            if validation_df['prediction'].iloc[i] != 0:
                validation_df['entry'].iloc[i] = 1

        # Extract only the rows where a buy/sell decision was made
        trades_df = validation_df[validation_df['position'] != 0]

        # Calculate returns based on the strategy decisions
        trades_df['return'] = np.log(trades_df['Close'] / trades_df['Close'].shift(1))

        # Calculate the cumulative strategy return
        trades_df['cum_strat_ret'] = (1 + trades_df['return'] * trades_df['position']).cumprod()
        trades_df['return'] = np.log(trades_df['Close'] / trades_df['Close'].shift(1))  # log_returns
        trades_df['strategy_return'] = trades_df['return'] * trades_df['position'].shift(1)  # log_returns
        trades_df = trades_df.dropna()

        # Assuming you have a column named 'strategy_return' in the trades_df
        avg_gain = trades_df[trades_df['strategy_return'] == 1]['strategy_return'].mean(skipna=True)
        max_gain = trades_df['strategy_return'].max()

        avg_loss = trades_df[trades_df['strategy_return'] == 0]['strategy_return'].mean(skipna=True)
        max_loss = trades_df['strategy_return'].min()

        winning_rate = (len(trades_df[trades_df['strategy_return'] > 0]) / 
                        len(trades_df))
        losing_rate = 1 - winning_rate

        expectancy = abs((winning_rate * avg_gain) / (losing_rate * avg_loss))

        out_dict = {"Winning rate": winning_rate * 100,
                    "Losing rate": losing_rate * 100,
                    "Maximum gain": max_gain * 100,
                    "Maximum loss": max_loss * 100,
                    "Average gain": avg_gain * 100,
                    "Average loss": avg_loss * 100,
                    "Expectancy": expectancy}

        return out_dict

    def run(self,
            asset: str,
            interval: str,
            resample_method: str,
            er_function: Callable,
            tau: int,
            alpha: float,
            param_grid: dict):

        valid_model_params = {"n_estimators", "min_samples_split",
                              "min_impurity_decrease", "ccp_alpha"}
        if not valid_model_params.intersection(set(param_grid.keys())):
            # If there are extra/wrong params
            raise ValueError(f"Keys in param_grid should be "
                             f"{valid_model_params}.")

        data_dir = \
            f"{self.processed_dir}{interval}/{asset}_{resample_method}.csv"

        df = pd.read_csv(data_dir)
        df = self._preprocess_data(asset_df=df, tau=tau, alpha=alpha,
                                   expected_return_function=er_function)
        df = self._add_variables(asset_df=df)
        final_model = self._fit_model(asset_df=df, param_grid=param_grid)

        # To continue with Irene's part
        if interval == "10m":
            validation_dir = \
                (f"{self.validation_dir}{interval}/{asset}_{resample_method}"
                 ".csv")
        else:
            validation_dir = \
                f"{self.validation_dir}{interval}/{asset}.csv"

        validation_df = pd.read_csv(validation_dir)
        # Apply feature engineering to the validation data
        out_dict = self._evaluate_model(validation_df=validation_df,
                                        final_model=final_model,
                                        alpha=alpha,
                                        tau=tau,
                                        er_function=er_function)

        return out_dict


def main():
    data_root = "./data/"
    raw_data_dir = data_root+"raw/5m/"
    processed_dir = data_root+"resampled/"

    fm = FinancialModel(data_root=data_root,
                        raw_data_dir=raw_data_dir,
                        processed_dir=processed_dir)

    asset = "ETHUSDT"
    interval = "10m"
    resample_method = "last"

    expected_return_function: Callable = max
    tau: int = 3                      # forecasting horizon

    alpha: float = 0.02              # Top alpha expected returns
    # Percentage of the price that if we see, we liquidate the position
    # stop_loss_fraction: float = 0.30

    # Hyperparameter tuning:
    param_grid = {
        "n_estimators": randint(low=100, high=1000),
        "min_samples_split": randint(low=2, high=20),
        "min_impurity_decrease": uniform(loc=0.0, scale=0.5),
        "ccp_alpha": [0.0, 0.5, 1.0]
    }

    return fm.run(asset=asset, interval=interval, 
                  resample_method=resample_method,
                  er_function=expected_return_function, tau=tau, alpha=alpha,
                  param_grid=param_grid)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    start = time.time()
    result = main()
    end = time.time()

    print(result)
    print(f"Time: {end-start}")
