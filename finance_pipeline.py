# Dependencies
import time
import warnings
from itertools import product

import pandas as pd
from scipy.stats.distributions import uniform, randint

from create_model import FinancialModel


# Main Code
def main():
    start_timer = time.time()

    data_root = "./data/"
    raw_data_dir = data_root+"raw/5m/"
    processed_dir = data_root+"resampled/"

    # Compare models
    fm = FinancialModel(data_root=data_root,
                        raw_data_dir=raw_data_dir,
                        processed_dir=processed_dir)
    # Get all possible inputs
    assets = ["ETHUSDT", "BTCUSDT", "DOGEUSDT"]
    intervals = ["5m", "15m"]
    resample_methods = ["mean", "last"]

    expected_return_functions = [min, max]
    tau_list = [3]
    alpha_list = [0.01, 0.02]

    param_grid = {
        "n_estimators": randint(low=100, high=1000),
        "min_samples_split": randint(low=2, high=20),
        "min_impurity_decrease": uniform(loc=0.0, scale=0.5),
        "ccp_alpha": [0.0, 0.5, 1.0]
    }

    result_list = []

    total_iterations = len(list(product(assets, intervals, resample_methods,
                                        expected_return_functions, tau_list,
                                        alpha_list)))
    counter = 0

    for combination in product(assets, intervals, resample_methods,
                               expected_return_functions, tau_list,
                               alpha_list):
        counter += 1

        asset = combination[0]
        interval = combination[1]
        resample_method = combination[2]
        expected_return_function = combination[3]
        tau = combination[4]
        alpha = combination[5]

        print(f"\nCombination: {combination} ({counter}/{total_iterations})")

        result_list.append(fm.run(asset=asset, interval=interval,
                           resample_method=resample_method,
                           er_function=expected_return_function, tau=tau,
                           alpha=alpha, param_grid=param_grid))

    model_results = pd.DataFrame(result_list)

    model_results.sort_values("Winning rate", inplace=True)

    model_results.to_csv("finance_pipeline_results.csv", encoding="utf-8")

    end_timer = time.time()
    print(f"Iteration time: {end_timer-start_timer}s\n")
    return model_results


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("Model selection full pipeline start.")
    start = time.time()
    result = main()
    end = time.time()

    print(result)
    print(f"Time: {end-start}")
