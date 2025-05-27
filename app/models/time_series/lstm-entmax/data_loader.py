"""
데이터 로딩 및 저장 함수들
"""
import pandas as pd
import os


def load_eco_data(data_path=None):
    """
    커피, 원유, 환율 등 경제 데이터를 로드합니다.
    
    Args:
        data_path (str, optional): 데이터 파일 경로. None이면 기본 경로 사용.
        
    Returns:
        pd.DataFrame: 경제 데이터가 포함된 데이터프레임
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'input', 'coffee_oil_exchange_daily.csv')
        if not os.path.exists(data_path):
            root_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
            data_path = os.path.join(root_dir, 'app', 'data', 'input', 'coffee_oil_exchange_daily.csv')
    
    df = pd.read_csv(data_path)
    return df


def load_weather_data(data_path=None):
    """
    날씨 및 기후 관련 데이터를 로드합니다.
    
    Args:
        data_path (str, optional): 데이터 파일 경로. None이면 기본 경로 사용.
        
    Returns:
        pd.DataFrame: 날씨 데이터가 포함된 데이터프레임
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'input', '비수확기평균커피가격통합데이터.csv')
        if not os.path.exists(data_path):
            root_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
            data_path = os.path.join(root_dir, 'app', 'data', 'input', '비수확기평균커피가격통합데이터.csv')
    
    df = pd.read_csv(data_path)
    return df


def save_result(result_df, data_path=None):
    """
    예측 결과를 CSV 파일로 저장합니다.
    
    Args:
        result_df (pd.DataFrame): 저장할 예측 결과 데이터프레임
        data_path (str, optional): 저장할 파일 경로. None이면 기본 경로 사용.
    """
    if data_path is None:
        # 기본 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        output_dir = os.path.join(app_dir, 'data', 'output')
        data_path = os.path.join(output_dir, 'prediction_result.csv')
    else:
        # 상대 경로인 경우 절대 경로로 변환
        if not os.path.isabs(data_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if data_path.startswith('../'):
                # 상대 경로 처리
                app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
                data_path = os.path.join(app_dir, data_path.replace('../', ''))
            else:
                data_path = os.path.join(current_dir, data_path)
    
    # 출력 디렉토리가 없으면 생성
    output_dir = os.path.dirname(data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        result_df.to_csv(data_path, index=False)
        print(f"예측 결과가 {data_path}에 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")
        # 현재 디렉토리에 백업 저장 시도
        backup_filename = f"prediction_result_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        backup_path = os.path.join(os.getcwd(), backup_filename)
        try:
            result_df.to_csv(backup_path, index=False)
            print(f"백업 파일이 {backup_path}에 저장되었습니다.")
        except Exception as backup_error:
            print(f"백업 저장도 실패했습니다: {backup_error}")