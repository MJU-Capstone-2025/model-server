"""
어텐션 LSTM 모델 모듈 (Attention LSTM Model Module)

이 모듈은 커피 가격 예측에 사용되는 Attention LSTM 모델의 아키텍처를 정의합니다.
EntmaxAttention을 사용하여 시계열 데이터에서 중요한 패턴을 감지하는 기능을 제공합니다.

주요 구성:
1. EntmaxAttention: 희소 어텐션을 제공하는 Entmax15 기반 어텐션 메커니즘
2. AttentionLSTMModel: 어텐션 메커니즘이 통합된 LSTM 모델 아키텍처

사용 예시:
    from model import AttentionLSTMModel
    
    # 모델 초기화
    model = AttentionLSTMModel(
        input_size=input_dim, 
        hidden_size=100,
        target_size=14
    )
    
    # 모델 추론 (forward pass)
    y_pred, attention_weights = model(x_batch)
"""

import torch
import torch.nn as nn

try:
    from entmax import Entmax15
except ImportError:
    print("⚠️ entmax package not found. Please install it with: pip install entmax")
    # Define a fallback for Entmax15 that uses softmax instead
    class Entmax15(nn.Module):
        def __init__(self, dim=None):
            super().__init__()
            self.dim = dim
        
        def forward(self, x):
            return nn.functional.softmax(x, dim=self.dim)


class EntmaxAttention(nn.Module):
    """
    Entmax15 기반의 어텐션 메커니즘
    
    기존 Softmax 어텐션보다 더 희소한(sparse) 어텐션 가중치를 산출하여
    더 선택적인 특성을 학습할 수 있도록 합니다.
    """
    
    def __init__(self, hidden_size, attn_dim=64):
        """
        EntmaxAttention 초기화
        
        Args:
            hidden_size (int): LSTM의 은닉 상태 크기
            attn_dim (int): 어텐션 내부 표현 차원
        """
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.entmax = Entmax15(dim=1)

    def forward(self, lstm_output):
        """
        순방향 계산
        
        Args:
            lstm_output (torch.Tensor): LSTM 출력, 형태 (batch_size, seq_len, hidden_size)
            
        Returns:
            tuple: (context, weights) 형태의 튜플
                - context: 어텐션 컨텍스트 벡터, 형태 (batch_size, hidden_size)
                - weights: 어텐션 가중치, 형태 (batch_size, seq_len)
        """
        # lstm_output: (B, T, H)
        scores = self.score_layer(lstm_output).squeeze(-1)  # (B, T)
        weights = self.entmax(scores)  # sparse attention weights
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)  # (B, H)
        return context, weights


class AttentionLSTMModel(nn.Module):
    """
    어텐션 메커니즘이 통합된 LSTM 모델
    
    특징:
    1. 다층(multi-layer) LSTM 
    2. EntmaxAttention을 통한 중요 시점 강조
    3. 게이팅 메커니즘을 통한 컨텍스트와 최종 은닉 상태 결합
    4. 비선형 예측 헤드
    """
    
    def __init__(self, input_size, hidden_size=100, num_layers=2, target_size=14, dropout=0.2):
        """
        AttentionLSTMModel 초기화
        
        Args:
            input_size (int): 입력 특성 차원
            hidden_size (int): LSTM 은닉 상태 차원
            num_layers (int): LSTM 층 수
            target_size (int): 예측할 미래 시점 수
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_size = target_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.attention = EntmaxAttention(hidden_size)

        # Gating layer to mix context and last_hidden
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

        # Nonlinear prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, target_size)
        )

    def forward(self, x, hidden_states=None):
        """
        순방향 계산
        
        Args:
            x (torch.Tensor): 입력 데이터, 형태 (batch_size, seq_len, input_size)
            hidden_states (tuple, optional): 초기 은닉 상태 튜플 (h0, c0)
            
        Returns:
            tuple: (predictions, attn_weights) 형태의 튜플
                - predictions: 모델의 예측값, 형태 (batch_size, target_size)
                - attn_weights: 어텐션 가중치, 형태 (batch_size, seq_len)
        """
        batch_size = x.size(0)

        if hidden_states is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden_states = (h0, c0)

        lstm_out, _ = self.lstm(x, hidden_states)  # (B, T, H)

        context, attn_weights = self.attention(lstm_out)  # (B, H)
        last_hidden = lstm_out[:, -1, :]  # (B, H)

        # Gated fusion of context and last hidden
        combined = torch.cat([context, last_hidden], dim=1)  # (B, 2H)
        alpha = self.gate(combined)  # (B, 1)
        fused = alpha * context + (1 - alpha) * last_hidden  # (B, H)

        out = self.fc(fused)  # (B, target_size)

        return out, attn_weights
