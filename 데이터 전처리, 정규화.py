import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import datetime


# Ddata = pd.read_excel() # 파일로드
# HIdata = Ddata.고가.iloc[0] # 당일 고가
# ACCdata = sum(Ddata.거래량.iloc[1:20]) # 19일치 누적 데이터 추출

# print(HIdata)
# print(ACCdata)

Mdata = pd.read_excel() # 파일로드
Mdata = Mdata.abs() # 절댓값 적용
Temp = Mdata[:127]
Temp = max(Temp['고가'])
print(Temp)
Mdata = Mdata[:146] # 과거 데이터 삭제
Mdata = Mdata.drop(columns = []) # 필요없는데이터 제거
Mdata = Mdata.sort_index(ascending=False).reset_index(drop=True) # 내림차순, 인덱스 초기화
# Mdata['고점'] = HIdata
# Mdata['일일누적거래량'] = ACCdata
# data = data.set_index("체결시간")
# data['체결시간'] = pd.to_datetime(data['체결시간'], format='%Y-%m-%d %H:%M:%S')

for i in Mdata.index:
  Mdata.loc[i, '고점대비하락률'] = (Temp - Mdata.loc[i, '현재가']) / Temp * 100

Mdata['3분누적거래량'] = 0
for i in Mdata.index[19:]:
  Mdata.loc[i, '3분누적거래량'] = Mdata.loc[i, '거래량'] + Mdata.loc[i - 1, '3분누적거래량']
  # Mdata.loc[i, '일일누적거래량'] = Mdata.loc[i, '3분누적거래량'] + ACCdata
  # Mdata.loc[i, '일일평균거래량'] = Mdata.loc[i, '일일누적거래량'] / 20
  Mdata.loc[i, '3분평균거래량'] = sum(Mdata.거래량.iloc[i - 19:i+1]) / 20

Mdata = Mdata.drop(columns = []) # 필요없는데이터 제거
Mdata = Mdata[19:]
Mdata = Mdata.reset_index(drop=True) # 내림차순, 인덱스 초기화

### 데이터 정규화
scaler = MinMaxScaler()
scale_cols = []
df_scaled = scaler.fit_transform(Mdata[scale_cols])
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

df_scaled.to_excel()
print(Mdata)
