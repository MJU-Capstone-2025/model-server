"""
손실 함수들
"""
import torch


def directional_loss(y_pred, y_true):
    """
    방향성 손실: 예측값과 실제값의 변화 방향이 일치하는지 측정합니다.
    
    Args:
        y_pred (torch.Tensor): 예측값
        y_true (torch.Tensor): 실제값
        
    Returns:
        torch.Tensor: 방향성 손실값
    """
    pred_diff = torch.sign(y_pred[:, 1:] - y_pred[:, :-1])
    true_diff = torch.sign(y_true[:, 1:] - y_true[:, :-1])
    return torch.mean((pred_diff != true_diff).float())


def variance_loss(y_pred, y_true):
    """
    분산 손실: 예측값과 실제값의 분산 차이를 측정합니다.
    
    Args:
        y_pred (torch.Tensor): 예측값
        y_true (torch.Tensor): 실제값
        
    Returns:
        torch.Tensor: 분산 손실값
    """
    return torch.abs(torch.std(y_pred) - torch.std(y_true)) 