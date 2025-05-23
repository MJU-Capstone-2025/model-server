import pandas as pd
from config import COFFEE_MACRO_DATA, COFFEE_WEATHER_DATA

def load_data():
    df_macro = pd.read_csv(COFFEE_MACRO_DATA)
    df_weather = pd.read_csv(COFFEE_WEATHER_DATA)
    return df_macro, df_weather