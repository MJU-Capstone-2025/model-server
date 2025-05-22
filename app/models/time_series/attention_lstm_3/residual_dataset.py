"""
잔차 신호를 활용한 시계열 데이터셋 모듈 (Time Series Dataset with Residuals)

이 모듈은 모델의 이전 예측 오류(residual)를 포함하는 시계열 데이터셋 클래스를 제공한다.
이전 예측 오차를 입력 특성으로 활용하여 모델의 예측 정확도를 향상시킨다.

주요 기능:
1. 시계열 데이터와 함께 잔차 신호를 처리
2. 최초 예측 후 실제 오차를 계산하여 다음 예측에 활용
3. 주기적인 재학습을 위한 데이터셋 업데이트 지원

사용 예시:
    from residual_dataset import ResidualTimeSeriesDataset
    from torch.utils.data import DataLoader
    
    # 데이터셋 생성
    train_dataset = ResidualTimeSeriesDataset(
        X_train, y_train, residuals=None,
        data_window=50, target_size=14, step=1
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ResidualTimeSeriesDataset(Dataset):
    """
    잔차 신호를 활용한 시계열 예측을 위한 PyTorch 데이터셋 클래스
    
    이 클래스는 시계열 데이터와 함께 이전 예측의 잔차 신호를 처리하여
    입력 시퀀스(X), 잔차 시퀀스(residual), 타겟 시퀀스(y)를 생성한다.
    """
    
    def __init__(self, dataset, target, residuals=None, data_window=50, 
                target_size=14, step=1, single_step=False, 
                residual_window=7, use_residuals=True):
        """
        데이터셋 초기화
        
        Args:
            dataset (np.ndarray): 입력 데이터 배열
            target (np.ndarray): 타겟 데이터 배열
            residuals (np.ndarray, optional): 모델의 이전 예측 잔차. None일 경우 0으로 초기화
            data_window (int): 입력 윈도우 크기 (과거 몇 개의 데이터를 볼 것인지)
            target_size (int): 예측할 미래 기간
            step (int): 샘플링 간격 (입력 윈도우 내 데이터 간격)
            single_step (bool): 단일 스텝 예측 여부
            residual_window (int): 사용할 잔차 윈도우 크기 (과거 몇 개의 잔차 데이터를 볼 것인지)
            use_residuals (bool): 잔차 데이터 사용 여부
        """
        self.data, self.labels, self.residuals = [], [], []
        self.use_residuals = use_residuals
        
        # 잔차 데이터가 없으면 0으로 초기화
        if residuals is None or not use_residuals:
            residuals = np.zeros((len(dataset), target_size))
        
        # 잔차 배열의 차원 확인 및 조정
        if len(residuals.shape) == 1:
            # 1차원 배열인 경우 2차원으로 변환 (각 샘플이 하나의 잔차만 가짐)
            residuals = residuals.reshape(-1, 1)
            
        # 배열 크기 확인 및 조정 (데이터셋 길이와 같아야 함)
        if len(residuals) < len(dataset):
            # 부족한 부분을 0으로 패딩
            pad_array = np.zeros((len(dataset) - len(residuals), residuals.shape[1]))
            residuals = np.vstack([pad_array, residuals])

        start_index = data_window
        end_index = len(dataset) - target_size  # 미래 예측을 고려해 끝점 조정

        for i in range(start_index, end_index):
            # 입력 데이터 준비
            indices = range(i - data_window, i, step)  # X 데이터 생성 (샘플링 적용)
            self.data.append(dataset[indices])
            
            # 타겟 데이터 준비
            if single_step:
                self.labels.append(target[i + target_size])  # 단일 값 예측
            else:
                self.labels.append(target[i:i + target_size])  # 다중 스텝 예측
            # 잔차 데이터 준비 (최근 residual_window 개수만큼)
            # 인덱스가 범위를 벗어나지 않도록 안전하게 처리
            start_res_idx = max(0, i - residual_window)  # 음수 인덱스 방지
            end_res_idx = min(i, len(residuals))  # 범위 초과 방지
            
            # 필요한 경우 앞부분을 0으로 패딩
            if i - start_res_idx < residual_window:
                pad_size = residual_window - (i - start_res_idx)
                res_slice = np.concatenate([np.zeros((pad_size, residuals.shape[1] if len(residuals.shape) > 1 else 1)), 
                                            residuals[start_res_idx:end_res_idx]])
            else:
                res_slice = residuals[start_res_idx:end_res_idx]
                
            self.residuals.append(res_slice)

        # 리스트를 PyTorch Tensor로 변환
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
        self.residuals = torch.tensor(np.array(self.residuals), dtype=torch.float32)

    def __len__(self):
        """데이터셋 길이를 반환한다"""
        return len(self.data)

    def __getitem__(self, idx):
        """특정 인덱스의 샘플을 반환한다"""
        if self.use_residuals:
            return self.data[idx], self.residuals[idx], self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]
    def update_residuals(self, new_residuals):
        """
        새로운 잔차 데이터로 기존 데이터셋을 업데이트한다.
        
        Args:
            new_residuals (np.ndarray): 새로운 잔차 데이터
        """
        if not self.use_residuals:
            return
            
        # 새 잔차가 없거나 잔차를 사용하지 않는 경우 업데이트하지 않음
        if new_residuals is None:
            return
        
        print(f"DEBUG: Updating residuals - self.residuals.shape={self.residuals.shape}, new_residuals.shape={new_residuals.shape}")
        
        # 텐서를 numpy로 변환
        residuals_np = self.residuals.detach().cpu().numpy()
        
        # 새 잔차가 텐서인 경우 numpy로 변환
        if isinstance(new_residuals, torch.Tensor):
            new_residuals = new_residuals.detach().cpu().numpy()
        
        # 새 잔차의 형태 조정
        # 만약 new_residuals의 차원이 1차원이면 2차원으로 변환
        if len(new_residuals.shape) == 1:
            new_residuals = new_residuals.reshape(-1, 1)
        
        # 3차원 residuals 처리 (batch_size, residual_window, feature_size)
        if len(residuals_np.shape) == 3:
            batch_size = min(len(residuals_np), len(new_residuals))
            window_size = residuals_np.shape[1]
            feature_size = residuals_np.shape[2]
            
            print(f"DEBUG: Handling 3D residuals with shape {residuals_np.shape}")
            
            # 각 샘플 배치에 대해 처리
            for i in range(batch_size):
                if i < len(new_residuals):
                    # 기존 윈도우를 한 칸 이동시킨다 (첫 번째 항목 버림)
                    residuals_np[i] = np.roll(residuals_np[i], -1, axis=0)
                    
                    # 새 잔차 데이터 형태 처리
                    if len(new_residuals.shape) == 2:
                        if new_residuals.shape[1] == feature_size:
                            # 새 잔차와 특성 차원이 일치하는 경우 직접 업데이트
                            residuals_np[i, -1] = new_residuals[i]
                        elif new_residuals.shape[1] == 1:
                            # 새 잔차가 스칼라인 경우 전체 feature에 복제
                            residuals_np[i, -1] = np.full(feature_size, new_residuals[i, 0])
                        else:
                            # 차원이 맞지 않을 경우 처리
                            print(f"DEBUG: Reshaping new_residuals from shape {new_residuals.shape} to match feature size {feature_size}")
                            # 특성 차원을 맞추기 위해 잘라내거나 패딩
                            if new_residuals.shape[1] > feature_size:
                                # 너무 클 경우 자르기
                                residuals_np[i, -1] = new_residuals[i, :feature_size]
                            else:
                                # 작을 경우 패딩
                                padded = np.zeros(feature_size)
                                padded[:new_residuals.shape[1]] = new_residuals[i]
                                residuals_np[i, -1] = padded
        else:
            # 기존 2차원 처리 로직 유지
            min_samples = min(len(residuals_np), len(new_residuals))
            
            # 각 샘플에 대해 처리
            for i in range(min_samples):
                # 특성 차원이 일치하지 않는 경우
                if new_residuals.shape[1] != residuals_np.shape[1]:
                    print(f"WARNING: Feature dimension mismatch: residuals_np[{i}].shape[1]={residuals_np.shape[1]}, new_residuals[{i}].shape[1]={new_residuals.shape[1]}")
                    
                    if new_residuals.shape[1] == 1:
                        # 스칼라인 경우 복제
                        new_value = new_residuals[i, 0]
                        residuals_np[i] = np.roll(residuals_np[i], -1, axis=0)
                        residuals_np[i, -1] = np.full(residuals_np.shape[1], new_value)
                    else:
                        # 차원이 맞지 않는 경우 처리
                        print(f"DEBUG: Reshaping new_residuals[{i}] from size {new_residuals.shape[1]} to {residuals_np.shape[1]}")
                        if new_residuals.shape[1] > residuals_np.shape[1]:
                            # 잘라내기
                            residuals_np[i] = np.roll(residuals_np[i], -1, axis=0)
                            residuals_np[i, -1] = new_residuals[i, :residuals_np.shape[1]]
                        else:
                            # 패딩
                            residuals_np[i] = np.roll(residuals_np[i], -1, axis=0)
                            padded = np.zeros(residuals_np.shape[1])
                            padded[:new_residuals.shape[1]] = new_residuals[i]
                            residuals_np[i, -1] = padded
                else:
                    # 차원이 일치하는 경우 그대로 업데이트
                    residuals_np[i] = np.roll(residuals_np[i], -1, axis=0)
                    residuals_np[i, -1] = new_residuals[i]
        
        # 업데이트된 잔차를 다시 텐서로 변환
        self.residuals = torch.tensor(residuals_np, dtype=torch.float32)
        print(f"DEBUG: Residuals updated successfully with shape {self.residuals.shape}")
