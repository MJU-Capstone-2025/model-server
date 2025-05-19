"""
시계열 데이터셋 모듈 (Time Series Dataset Module)

이 모듈은 LSTM 모델 학습에 필요한 시계열 데이터셋 클래스를 제공한다.
PyTorch의 Dataset 클래스를 상속하여 시계열 데이터를 미니배치로 처리할 수 있도록 구현한다.

주요 기능:
1. 시계열 데이터를 슬라이딩 윈도우 방식으로 처리
2. 단일 스텝 및 다중 스텝 예측을 위한 데이터 준비
3. 데이터 샘플링 (step 파라미터를 통해) 지원

사용 예시:
    from dataset import MultiStepTimeSeriesDataset
    from torch.utils.data import DataLoader
    
    # 데이터셋 생성
    train_dataset = MultiStepTimeSeriesDataset(X_train, y_train, data_window=50, target_size=14, step=1)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiStepTimeSeriesDataset(Dataset):
    """
    다중 스텝 시계열 예측을 위한 PyTorch 데이터셋 클래스
    
    이 클래스는 시계열 데이터를 슬라이딩 윈도우 방식으로 처리하여
    입력 시퀀스(X)와 타겟 시퀀스(y)를 생성한다.
    """
    
    def __init__(self, dataset, target, data_window, target_size, step, single_step=False):
        """
        데이터셋 초기화
        
        Args:
            dataset (np.ndarray): 입력 데이터 배열
            target (np.ndarray): 타겟 데이터 배열
            data_window (int): 입력 윈도우 크기 (과거 몇 개의 데이터를 볼 것인지)
            target_size (int): 예측할 미래 기간
            step (int): 샘플링 간격 (입력 윈도우 내 데이터 간격)
            single_step (bool): 단일 스텝 예측 여부
        """
        self.data, self.labels = [], []

        start_index = data_window
        end_index = len(dataset) - target_size  # 미래 예측을 고려해 끝점 조정

        for i in range(start_index, end_index):
            indices = range(i - data_window, i, step)  # X 데이터 생성 (샘플링 적용)
            self.data.append(dataset[indices])

            if single_step:
                self.labels.append(target[i + target_size])  # 단일 값 예측
            else:
                self.labels.append(target[i:i + target_size])  # 다중 스텝 예측

        # 리스트를 PyTorch Tensor로 변환
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        """데이터셋 길이를 반환한다"""
        return len(self.data)

    def __getitem__(self, idx):
        """특정 인덱스의 샘플을 반환한다"""
        return self.data[idx], self.labels[idx]