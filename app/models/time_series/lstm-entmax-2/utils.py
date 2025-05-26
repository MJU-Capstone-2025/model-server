"""
유틸리티 함수들
"""
import torch


def get_device():
    """
    사용 가능한 GPU 또는 CPU 장치를 반환합니다.
    
    Returns:
        str: 'cuda' (GPU 사용 가능시) 또는 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_attention_entropy(attn_weights):
    """
    어텐션 가중치의 엔트로피를 계산합니다.
    
    Args:
        attn_weights (torch.Tensor): 어텐션 가중치 텐서
        
    Returns:
        float: 평균 엔트로피 값
    """
    eps = 1e-8
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=1)
    return entropy.mean().item() 