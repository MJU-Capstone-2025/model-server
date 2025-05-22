"""
잔차 활용 어텐션 LSTM 모델 모듈 (Residual-Enhanced Attention LSTM Model Module)

이 모듈은 이전 예측 오차(residual)를 활용하여 예측 정확도를 향상시키는 
Attention LSTM 모델의 아키텍처를 정의한다.

주요 구성:
1. 주 데이터 스트림과 잔차 데이터 스트림을 위한 이중 인코더 구조
2. ResidualAttentionBlock: 잔차 데이터를 활용하는 특별한 어텐션 블록
3. ResidualAttentionLSTM: 잔차 신호를 통합하는 확장된 LSTM 모델

사용 예시:
    from residual_model import ResidualAttentionLSTM
    
    # 모델 초기화
    model = ResidualAttentionLSTM(
        input_size=input_dim, 
        residual_size=1,  # 잔차 특성 차원 (일반적으로 1)
        hidden_size=100,
        target_size=14
    )
    
    # 모델 추론 (forward pass)
    y_pred, attention_weights = model(x_batch, residual_batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from entmax import Entmax15
except ImportError:
    print("⚠️ entmax 패키지를 찾을 수 없다. 다음 명령어로 설치하자: pip install entmax")
    # Entmax15의 대체 구현으로 softmax를 사용
    class Entmax15(nn.Module):
        def __init__(self, dim=None):
            super().__init__()
            self.dim = dim
        
        def forward(self, x):
            return nn.functional.softmax(x, dim=self.dim)


class EntmaxAttention(nn.Module):
    """
    Entmax15 기반의 어텐션 메커니즘
    """
    
    def __init__(self, hidden_size, attn_dim=64):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.entmax = Entmax15(dim=1)

    def forward(self, lstm_output):
        scores = self.score_layer(lstm_output).squeeze(-1)  # (B, T)
        weights = self.entmax(scores)  # sparse attention weights
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)  # (B, H)
        return context, weights


class ResidualAttentionBlock(nn.Module):
    """
    잔차 데이터를 처리하는 어텐션 블록
    
    이 블록은 잔차 신호를 처리하여 주요 시계열 데이터의 처리를 보완한다.
    """
    
    def __init__(self, residual_size, hidden_size, attn_dim=32):
        """
        초기화
        
        Args:
            residual_size (int): 잔차 신호 입력 차원
            hidden_size (int): 은닉 상태 크기
            attn_dim (int): 어텐션 중간 차원
        """
        super().__init__()
        
        # 잔차를 처리하는 LSTM
        self.residual_lstm = nn.LSTM(
            input_size=residual_size,
            hidden_size=hidden_size // 2,  # 주요 LSTM보다 작은 크기로 설정
            batch_first=True
        )
        
        # 잔차에 대한 어텐션
        self.attention = EntmaxAttention(hidden_size // 2, attn_dim)

    def forward(self, residuals):
        """
        순방향 계산을 수행한다.
        
        Args:
            residuals (torch.Tensor): 잔차 입력 (batch_size, seq_len, residual_size)
            
        Returns:
            tuple: (context, weights) 형태의 튜플
        """
        # 잔차 LSTM 처리
        residual_out, _ = self.residual_lstm(residuals)  # (B, T, H//2)
        
        # 잔차 어텐션
        context, weights = self.attention(residual_out)  # (B, H//2)
        
        return context, weights


class ResidualAttentionLSTM(nn.Module):
    """
    잔차 활용 어텐션 LSTM 모델
    
    특징:
    1. 시계열 데이터와 잔차 데이터를 별도로 처리
    2. 두 정보 스트림을 결합하여 최종 예측
    3. 게이트 메커니즘을 통해 잔차 정보의 영향력 조절
    """
    
    def __init__(self, input_size, residual_size=1, hidden_size=100, 
                num_layers=2, target_size=14, dropout=0.2):
        """
        ResidualAttentionLSTM을 초기화한다.
        
        Args:
            input_size (int): 입력 시계열 데이터 특성 차원
            residual_size (int): 잔차 특성 차원 (일반적으로 1)
            hidden_size (int): LSTM 은닉 상태 차원
            num_layers (int): LSTM 층 수
            target_size (int): 예측할 미래 시점 수
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_size = target_size
        
        # 입력 시계열 데이터를 처리하는 LSTM
        self.main_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 주 시계열 어텐션
        self.main_attention = EntmaxAttention(hidden_size)
        
        # 잔차 처리 블록
        self.residual_block = ResidualAttentionBlock(residual_size, hidden_size)
        
        # 주 스트림과 잔차 스트림의 정보를 결합하는 게이트
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//2, 1),
            nn.Sigmoid()
        )
        
        # 최종 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, target_size)
        )
        
        # 잔차 영향력 조절 게이트
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.Sigmoid()
        )

    def forward(self, x, residuals=None):
        """
        순방향 계산을 수행한다.
        
        Args:
            x (torch.Tensor): 입력 시계열 데이터 (batch_size, seq_len, input_size)
            residuals (torch.Tensor, optional): 잔차 입력 (batch_size, residual_seq_len, residual_size)None일 경우 잔차 입력을 사용하지 않음
            
        Returns:
            tuple: (predictions, attention_weights) 형태의 튜플
        """
        batch_size = x.size(0)
        
        # 주 시계열 데이터 처리
        main_out, _ = self.main_lstm(x)  # (B, T, H)
        main_context, main_attention_weights = self.main_attention(main_out)  # (B, H)
        
        # 잔차 데이터 처리 (있을 경우)
        if residuals is not None:
            residual_context, residual_attention_weights = self.residual_block(residuals)
            
            # 잔차 영향력 조절
            residual_gate = self.residual_gate(residual_context)
            residual_context = residual_context * residual_gate
            
            # 주 스트림과 잔차 스트림 결합
            combined = torch.cat([main_context, residual_context], dim=1)  # (B, H + H//2)
            
            # 결합 가중치 계산
            alpha = self.fusion_gate(combined)  # (B, 1)
            
            # 주 컨텍스트와 잔차 컨텍스트 조합
            fused_context = torch.cat([
                main_context, 
                residual_context
            ], dim=1)  # (B, H + H//2)
            
            attention_weights = (main_attention_weights, residual_attention_weights)
        else:
            # 잔차 입력이 없는 경우
            zero_residual = torch.zeros(batch_size, self.hidden_size // 2).to(x.device)
            fused_context = torch.cat([main_context, zero_residual], dim=1)
            attention_weights = main_attention_weights
        
        # 최종 예측
        predictions = self.output_layer(fused_context)
        
        return predictions, attention_weights
