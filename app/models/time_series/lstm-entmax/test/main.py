import numpy as np
import pandas as pd

def analyze_sliding_window_leakage():
    """슬라이딩 윈도우에서 데이터 누출 문제 분석"""
    
    # 샘플 데이터 생성 (날짜별)
    dates = pd.date_range('2024-01-01', periods=120, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'day': range(1, 121),  # 1일부터 120일까지
        'log_return': np.random.randn(120) * 0.02
    })
    
    print("=== 슬라이딩 윈도우 데이터 누출 분석 ===\n")
    
    # 현재 코드 방식: shift(-13)
    df['target_volatility_14d'] = df['log_return'].rolling(14).std().shift(-13)
    
    # 특정 시점에서 확인해보자
    check_day = 50  # 50일차 확인
    
    print(f"📅 {check_day}일차 분석:")
    print(f"날짜: {df.loc[check_day-1, 'date'].strftime('%Y-%m-%d')}")
    
    # 50일차의 target_volatility_14d 값이 실제로 어느 기간의 데이터인지 확인
    # shift(-13)이므로, 50일차의 값은 실제로는 (50+13)=63일차부터의 14일간 변동성
    actual_period_start = check_day + 13  # 63일
    actual_period_end = actual_period_start + 13  # 76일
    
    print(f"50일차의 target_volatility_14d 값:")
    print(f"  - 표면적: 37~50일의 변동성")
    print(f"  - 실제: {actual_period_start}~{actual_period_end}일의 변동성")
    print(f"  - 값: {df.loc[check_day-1, 'target_volatility_14d']:.6f}")
    
    # 실제 계산해서 확인
    manual_calc = df.loc[actual_period_start-1:actual_period_end-1, 'log_return'].std()
    print(f"  - 수동 계산: {manual_calc:.6f}")
    print(f"  - 일치여부: {abs(df.loc[check_day-1, 'target_volatility_14d'] - manual_calc) < 1e-10}")
    
    print(f"\n🚨 문제점:")
    print(f"50일차까지의 데이터로 학습할 때,")
    print(f"실제로는 {actual_period_start}~{actual_period_end}일의 미래 정보를 사용!")
    
    # 슬라이딩 윈도우 시나리오 시뮬레이션
    print(f"\n=== 슬라이딩 윈도우 시나리오 ===")
    
    window_size = 50
    prediction_target_day = 101  # 101일을 예측하려고 함
    
    # 학습 데이터: 1~50일
    train_end = 50
    print(f"학습 구간: 1~{train_end}일")
    print(f"예측 목표: {prediction_target_day}일")
    
    # 현재 방식으로 50일차의 피처 확인
    feature_at_day50 = df.loc[49, 'target_volatility_14d']  # 50일차 (0-indexed)
    
    # 이 값이 실제로는 언제의 데이터인지
    leaked_period_start = 50 + 13  # 63일
    leaked_period_end = leaked_period_start + 13  # 76일
    
    print(f"\n50일차 학습 시 사용되는 'target_volatility_14d' 피처:")
    print(f"  - 실제 데이터 기간: {leaked_period_start}~{leaked_period_end}일")
    print(f"  - 예측하려는 101일보다는 과거이지만...")
    print(f"  - 학습 시점(50일)보다는 미래 데이터!")
    
    return df

def show_correct_approach():
    """올바른 접근 방식"""
    
    print(f"\n=== 올바른 접근 방식 ===")
    
    dates = pd.date_range('2024-01-01', periods=120, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'day': range(1, 121),
        'log_return': np.random.randn(120) * 0.02
    })
    
    # 올바른 방식: 과거 데이터만 사용
    df['past_volatility_14d'] = df['log_return'].rolling(14).std()  # shift 없음!
    
    # 타겟: 미래 14일 후의 변동성
    df['target_volatility_14d'] = df['log_return'].rolling(14).std().shift(-14)  # 14일 후
    
    check_day = 50
    print(f"\n📅 {check_day}일차 (올바른 방식):")
    print(f"피처 'past_volatility_14d': {df.loc[check_day-1, 'past_volatility_14d']:.6f}")
    print(f"  - 기간: {check_day-13}~{check_day}일의 변동성")
    
    print(f"타겟 'target_volatility_14d': {df.loc[check_day-1, 'target_volatility_14d']:.6f}")
    print(f"  - 기간: {check_day+1}~{check_day+14}일의 변동성")
    
    print(f"\n✅ 이제 50일차까지 학습할 때:")
    print(f"  - 피처: 과거 데이터만 사용 (37~50일)")
    print(f"  - 타겟: 미래 데이터 사용 (51~64일) - 정상!")
    print(f"  - 예측: 101일의 변동성을 예측할 때, 미래 정보 없이 가능")

def demonstrate_realistic_scenario():
    """실제 운용 시나리오 시연"""
    
    print(f"\n=== 실제 운용 시나리오 ===")
    
    print("상황: 2024-12-01 (120일차)에 앉아서 2024-12-15 (134일차)를 예측")
    print("사용 가능한 데이터: 2024-08-03 ~ 2024-12-01 (71~120일)")
    
    print(f"\n현재 코드 방식의 문제:")
    print("  - 120일차의 'target_volatility_14d' 피처 = 133~146일의 변동성")
    print("  - 133일은 아직 오지도 않은 미래!")
    print("  - 실제 운용에서는 이 값을 알 수 없음")
    
    print(f"\n올바른 방식:")
    print("  - 120일차의 'past_volatility_14d' 피처 = 107~120일의 변동성")
    print("  - 이미 관측된 과거 데이터만 사용")
    print("  - 실제 운용에서 사용 가능")

if __name__ == "__main__":
    df = analyze_sliding_window_leakage()
    show_correct_approach()
    demonstrate_realistic_scenario()