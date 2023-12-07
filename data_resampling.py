# Libraries
import os
import pandas as pd

from typing import Optional


# Main Code
class DataResampler:
    def __init__(self,
                 data_root: Optional[str] = None,
                 raw_data_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None) -> None:

        self.resample_intervals: list[int] = [10]
        self.resample_methods: list[str] = ['last', 'first', 'mean']

        self.data_root = "./data/" if data_root is None else data_root
        self.raw_data_dir = self.data_root+"raw/" \
            if raw_data_dir is None else raw_data_dir
        self.processed_dir = self.data_root+"resampled/" \
            if processed_dir is None else processed_dir

    def _resample_df(self, trading_df: pd.DataFrame, data_name: str):
        """Resample the dataframe to the frequencies provided."""
        # Instantiate relevant attributes from self for clarity.
        frequencies = self.resample_intervals
        resampling_methods = self.resample_methods
        output_dir = self.processed_dir

        df_original = trading_df.copy()

        df_original['Time'] = pd.to_datetime(df_original['Time'])
        df_original.set_index('Time', inplace=True)

        # SECTION: RESAMPLING - - -
        # Create a dictionary to store resampled DataFrames
        resampled_dfs = {}

        # Iterate through resampling intervals and methods
        for interval in frequencies:
            for method in resampling_methods:
                # Resample the data with label set to 'right' and aggregate
                # using the current method.
                df_resampled = \
                    df_original.resample(f'{interval}T', label='right')\
                    .agg(method)
                # Generate a unique name for each dataframe
                df_name = f'{data_name}_resampled_{interval}m_{method}'
                # Store the name in the dictionary for reference
                resampled_dfs[df_name] = df_resampled

        # SECTION: SAVE RESAMPLED DFS - - -
        for df_name, df in resampled_dfs.items():
            # Structure: Token, "resampled", time, resampling rule
            token, _, frequency, method = str(df_name).split("_")

            filename = f"{token}_{method}.csv"
            filedir = f"{output_dir}{frequency}/"

            df.to_csv(f"{filedir}{filename}", sep=",")

    def _create_necessary_directories(self):
        # Ensure that the necessary directories exist.
        # A bit dirty, but it works.
        for directory in [f"{self.data_root}resampled",
                          f"{self.data_root}resampled/5m",
                          f"{self.data_root}resampled/10m",
                          f"{self.data_root}resampled/15m"]:
            try:
                os.mkdir(directory)
            except FileExistsError:
                continue

    def prepare_data(self):
        """
        Resample series for all coins present in raw and return list of
        assets.
        """
        raw_directory = self.raw_data_dir

        # self._create_necessary_directories()

        asset_list = []
        for asset_filename in os.listdir(raw_directory):
            try:
                asset_df = pd.read_csv(raw_directory+asset_filename)
            except IsADirectoryError:
                continue
            # Save the crypto's name
            asset_name = asset_filename.split(".")[0]
            asset_list.append(asset_name)

            # Call for its resampling
            self._resample_df(trading_df=asset_df, data_name=asset_name)

        return asset_list


def main():
    data_root = "./data/"
    raw_data_dir = data_root+"validation/5m/"
    processed_dir = data_root+"validation/"

    dr = DataResampler(data_root=data_root,
                       raw_data_dir=raw_data_dir,
                       processed_dir=processed_dir)

    return dr.prepare_data()


if __name__ == "__main__":
    main()
