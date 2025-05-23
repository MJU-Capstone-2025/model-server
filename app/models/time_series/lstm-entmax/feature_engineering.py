import numpy as np

def create_features(df):
    df['abs_return'] = df['Coffee_Price_Return'].abs()
    df['log_return'] = np.log(df['Coffee_Price']) - np.log(df['Coffee_Price'].shift(1))
    df['target_volatility_14d'] = df['log_return'].rolling(14).std().shift(-13)
    df['volatility_5d'] = df['Coffee_Price_Return'].rolling(5).std()
    df['volatility_10d'] = df['Coffee_Price_Return'].rolling(10).std()
    df['momentum_1d'] = df['Coffee_Price'].diff(1)
    df['momentum_3d'] = df['Coffee_Price'].diff(3)
    df['momentum_5d'] = df['Coffee_Price'] - df['Coffee_Price'].shift(5)

    mean_20 = df['Coffee_Price'].rolling(20).mean()
    std_20 = df['Coffee_Price'].rolling(20).std()
    df['bollinger_width'] = (2 * std_20) / mean_20

    return_z = (df['Coffee_Price_Return'] - df['Coffee_Price_Return'].rolling(20).mean())
    df['return_zscore'] = return_z / (df['Coffee_Price_Return'].rolling(20).std() + 1e-6)
    df['volatility_ratio'] = df['volatility_5d'] / (df['volatility_10d'] + 1e-6)
    return df