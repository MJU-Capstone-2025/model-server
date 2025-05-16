"""
RF 모델의 진입점 파일
"""

import sys
import os

# 현재 파일의 디렉토리 경로를 구한 다음, rf_model 폴더의 위치를 확인합니다
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir, 'rf_model')

# 모듈 경로를 시스템 경로에 추가합니다
sys.path.append(module_dir)

from main import run_model

if __name__ == "__main__":
    run_model()
