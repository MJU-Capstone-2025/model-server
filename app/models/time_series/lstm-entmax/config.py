"""
모델 설정 파일
"""

import os
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'input')

COFFEE_MACRO_DATA = os.path.join(DATA_DIR, '거시경제및커피가격통합데이터.csv')
COFFEE_WEATHER_DATA = os.path.join(DATA_DIR, '비수확기평균커피가격통합데이터.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'