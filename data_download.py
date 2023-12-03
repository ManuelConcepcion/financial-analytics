# Libraries
import time
import datetime
import os

import pandas as pd

from binance import Client


# Main Code: DataDownloader Class
class DataDownloader:
    def __init__(self) -> None:
        # Setup --------------
        self.dir = 'data/raw'
        # Content ------------
        self.intervals = ['5m']
        self.assets = ['BTC', 'ETH', 'DOGE']
        # interval ----------
        self.start_date = datetime.date(2023, 9, 1)
        self.end_one_day = self.start_date + datetime.timedelta(1)
        self.end_date = datetime.date(2023, 11, 15)

    @staticmethod
    def get_data(client,
                 symbol,
                 interval,
                 start_str,
                 end_str,) -> pd.DataFrame:
        # Download Binance Data
        df = pd.DataFrame(
            client.get_historical_klines(symbol=symbol,
                                         interval=interval,
                                         start_str=start_str,
                                         end_str=end_str,)
                                        )
        # Transform result
        df = df.iloc[:-1, :6]
        df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df.set_index('Time')
        df.index = pd.to_datetime(df.index, unit='ms')
        df = df.astype(float)

        return df

    def _run_interval(self,
                      interval: str,
                      client: Client,
                      mode: str,):
        # Determine the end date to use from the mode
        if mode == 'day':
            end_date = self.end_one_day
        elif mode == 'full':
            end_date = self.end_date
        else:
            raise ValueError("Argument 'mode' must be one of"
                             "('day', 'full').")

        for asset in self.assets:
            print(f"\n\tASSET {asset}...")
            date = self.start_date
            data_df = None

            day_counter = 0     # So we do not write after every single row

            symbol = asset+'USDT'
            path = self.dir+f'/{interval}'
            while (date <= end_date):
                next_day = date + datetime.timedelta(1)

                download_data = self.get_data(client=client,
                                              symbol=symbol,
                                              interval=interval,
                                              start_str=str(date),
                                              end_str=str(next_day))

                if data_df is None:
                    data_df = download_data.copy()
                else:
                    data_df = pd.concat([data_df, download_data], axis=0)

                day_counter += 1
                if day_counter > 9:
                    data_df.to_csv(f'{path}/{symbol}.csv')

                date = next_day
                time.sleep(1)

            # Finish and save
            if data_df is not None:
                path = self.dir+f'/{interval}'
                data_df.to_csv(f'{path}/{symbol}.csv')

        return f"Interval {interval} done."

    def run(self,
            mode: str = 'day'):
        """
        Mode must be one of {day, full}
        """
        # Create directories to store the data
        print("Creating folders...")
        for interval in self.intervals:
            if interval not in os.listdir(self.dir):
                os.mkdir(self.dir+'/'+interval)
        print("Folders done.\n")

        # Initialize Binance Client
        client = Client()

        timer_start = time.time()

        interval_start = time.time()
        print("Downloading data...\n")
        for interval in self.intervals:
            print(f"Downloading interval {interval}...")
            asset_result = self._run_interval(interval=interval, client=client,
                                              mode=mode)
            interval_end = time.time()
            print(asset_result+f" Time: {interval_end-interval_start}s\n")
            # "Lap" timer
            interval_start = interval_end
        timer_end = time.time()
        print(f"Data download finished. Total time: {timer_end-timer_start}s")


# Main Function
def main():
    directory = 'data/raw'
    if directory not in os.listdir():
        os.makedirs(directory)

    dd = DataDownloader()

    dd.run(mode="full")
    print("\nAll finished. Enjoy!")


if __name__ == "__main__":
    main()
