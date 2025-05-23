import numpy as np
import pandas as pd

def simulate_price_curve(start_date, vol, base_price, days=14):
    np.random.seed(42)
    returns = np.random.normal(0, vol, size=days)
    prices = base_price * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.date_range(start=start_date, periods=days))