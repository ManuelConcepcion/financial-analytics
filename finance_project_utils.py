import numpy as np
from sklearn.model_selection import cross_validate
from scipy.stats import norm


def var_cov_var(P, c, mu, sigma):
    # Calculate the Value-at-Risk using the Variance-Covariance method
    # P: Portfolio value, c: Confidence level, mu: Mean of returns,
    # sigma: Standard deviation of returns
    alpha = norm.ppf(1-c, mu, sigma)
    return P - P*(alpha + 1)


def RSI(prices, n=9):
    # Calculate the Relative Strength Index (RSI) for a given series of prices
    # prices: Closing prices, n: Period for RSI calculation (default is 9)
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    down = -seed[seed < 0].sum()/n
    down = 1 if down == 0 else down
    up = seed[seed >= 0].sum()/n
    rs = up/down
    rsi = np.zeros(len(prices))
    rsi[:n] = 100 - 100/(1 + rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta

        up = float((up*(n-1) + upval)/n)
        down = float((down*(n-1) + downval)/n + 0.00001)
        rs = up/down
        rsi[i] = 100 - 100/(1 + rs)

    return rsi


def evaluate_model(model, X, y, cv):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=['f1', 'roc_auc']
    )
    f1 = cv_results['test_f1']
    roc_auc = cv_results['test_roc_auc']

    print(
        f"F1 Score:     {f1.mean():.3f} +/- {f1.std():.3f}\n"
        f"ROC Area Under the Curve: {roc_auc.mean():.3f} +/- "
        f"{roc_auc.std():.3f}"
    )

    return cv_results
