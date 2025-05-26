"""
모델 관련 클래스들
"""
import torch
import torch.nn as nn
from entmax import Entmax15


class EntmaxAttention(nn.Module):
    """
    Entmax 기반 어텐션 메커니즘을 구현한 클래스.
    """
    
    def __init__(self, hidden_size, attn_dim=64):
        """
        Args:
            hidden_size (int): LSTM 은닉 상태 크기
            attn_dim (int): 어텐션 차원 크기
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
        Args:
            lstm_output (torch.Tensor): LSTM 출력 텐서 [batch_size, seq_len, hidden_size]
            
        Returns:
            tuple: (context_vector, attention_weights)
        """
        scores = self.score_layer(lstm_output).squeeze(-1)
        weights = self.entmax(scores)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights


class AttentionLSTMModel(nn.Module):
    """
    Entmax 어텐션과 정적 피처를 결합한 LSTM 모델.
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, target_size=14, 
                 dropout=0.1, static_feat_dim=9):
        """
        Args:
            input_size (int): 입력 피처 차원
            hidden_size (int): LSTM 은닉 상태 크기
            num_layers (int): LSTM 레이어 수
            target_size (int): 예측 구간 길이
            dropout (float): 드롭아웃 비율
            static_feat_dim (int): 정적 피처 차원
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_size = target_size
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # 어텐션 메커니즘
        self.attention = EntmaxAttention(hidden_size)
        
        # 게이트 메커니즘 (컨텍스트와 마지막 은닉 상태 결합)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )
        
        # 최종 예측 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, target_size)
        )
    
    def forward(self, x_seq, x_static, hidden_states=None):
        """
        Args:
            x_seq (torch.Tensor): 시퀀스 입력 [batch_size, seq_len, input_size]
            x_static (torch.Tensor): 정적 피처 입력 [batch_size, static_feat_dim]
            hidden_states (tuple, optional): 초기 은닉 상태
            
        Returns:
            tuple: (predictions, attention_weights)
        """
        batch_size = x_seq.size(0)
        
        # 초기 은닉 상태 설정
        if hidden_states is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_seq.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_seq.device)
            hidden_states = (h0, c0)
        
        # LSTM 순전파
        lstm_out, _ = self.lstm(x_seq, hidden_states)
        
        # 어텐션 적용
        context, attn_weights = self.attention(lstm_out)
        last_hidden = lstm_out[:, -1, :]
        
        # 게이트를 통한 컨텍스트와 마지막 은닉 상태 결합
        combined = torch.cat([context, last_hidden], dim=1)
        alpha = self.gate(combined)
        fused = alpha * context + (1 - alpha) * last_hidden
        
        # 정적 피처와 결합하여 최종 예측
        fused_with_static = torch.cat([fused, x_static], dim=1)
        out = self.fc(fused_with_static)
        
        return out, attn_weights 