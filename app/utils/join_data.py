import pandas as pd

data1 = pd.read_csv("./data/input/weather_data.csv")
data2 = pd.read_csv("./data/input/market.csv")

print(data1.shape)
print(data2.shape)

# Date를 기준으로 data2의 Crude_Oil_Price,USD_KRW,USD_BRL,USD_COP,USD_ETB 컬름들을 data1에 추가
joined_data = data1.merge(data2[["Date", "Crude_Oil_Price", "USD_KRW", "USD_BRL", "USD_COP", "USD_ETB"]], on="Date", how="left")

print(joined_data.shape)

# 모든 USD 관련 컬럼에 대해 결측치 채우기
for column in ["Crude_Oil_Price", "USD_KRW", "USD_BRL", "USD_COP", "USD_ETB"]:
    joined_data[column] = joined_data[column].fillna(method="ffill", limit=30)

# 결과 파일 저장하기 (파일 다시 읽기 전에)
joined_data.to_csv("./data/input/joined_data_filled.csv", index=False)

# 결측치 확인
print(joined_data.isnull().sum())

# 채워진 파일을 읽기
joined_data = pd.read_csv("./data/input/joined_data_filled.csv") 
