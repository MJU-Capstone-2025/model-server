"""
데이터셋 클래스
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiStepTimeSeriesDataset(Dataset):
    """
    다중 스텝 시계열 예측을 위한 PyTorch 데이터셋 클래스.
    시퀀스 피처와 정적 피처를 분리하여 처리합니다.
    """
    
    def __init__(self, X, y, window_size, horizon, step, static_feat_idx):
        """
        Args:
            X (np.ndarray): 입력 피처 배열
            y (np.ndarray): 타겟 배열
            window_size (int): 입력 시퀀스 길이
            horizon (int): 예측 구간 길이
            step (int): 슬라이딩 윈도우 스텝 크기
            static_feat_idx (list): 정적 피처의 인덱스 리스트
        """
        self.seq_feat_idx = [i for i in range(X.shape[1]) if i not in static_feat_idx]
        
        # 시퀀스 데이터 생성
        self.data = [
            X[i:i+window_size, self.seq_feat_idx] 
            for i in range(0, len(X) - window_size - horizon + 1, step)
        ]
        
        # 정적 피처 데이터 생성
        self.static_feats = [
            X[i, static_feat_idx] 
            for i in range(0, len(X) - window_size - horizon + 1, step)
        ]
        
        # 라벨 데이터 생성
        self.labels = [
            y[i+window_size:i+window_size+horizon] 
            for i in range(0, len(X) - window_size - horizon + 1, step)
        ]
        
        # NumPy 배열로 변환
        self.data = np.array(self.data)
        self.static_feats = np.array(self.static_feats)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.static_feats[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        ) 