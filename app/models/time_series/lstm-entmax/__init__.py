"""
LSTM with Entmax Attention for Coffee Price Prediction

이 패키지는 Entmax 어텐션을 활용한 LSTM 기반 커피 가격 예측 모델을 제공합니다.
"""

from .utils import get_device, compute_attention_entropy
from .data_loader import load_eco_data, load_weather_data, save_result
from .preprocessor import preprocess_data, split_and_scale
from .dataset import MultiStepTimeSeriesDataset
from .models import EntmaxAttention, AttentionLSTMModel
from .losses import directional_loss, variance_loss
from .visualizer import plot_loss, plot_prediction
from .trainer import train_model, predict_and_inverse, predict_future, evaluate_and_save

__version__ = "1.0.0"
__author__ = "Coffee Price Prediction Team"

__all__ = [
    # Utils
    "get_device",
    "compute_attention_entropy",
    
    # Data
    "load_eco_data",
    "load_weather_data", 
    "save_result",
    "preprocess_data",
    "split_and_scale",
    
    # Dataset
    "MultiStepTimeSeriesDataset",
    
    # Models
    "EntmaxAttention",
    "AttentionLSTMModel",
    
    # Losses
    "directional_loss",
    "variance_loss",
    
    # Visualization
    "plot_loss",
    "plot_prediction",
    
    # Training
    "train_model",
    "predict_and_inverse", 
    "predict_future",
    "evaluate_and_save",
] 