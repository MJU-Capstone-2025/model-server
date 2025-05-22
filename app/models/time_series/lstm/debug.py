def debug_data_shape(train_data, test_data, train_loader=None, test_loader=None):
    if train_data is not None:
        print(f"[DEBUG] train_data 형태: {train_data.shape}")
    if test_data is not None:
        print(f"[DEBUG] test_data 형태: {test_data.shape}")
    
    # DataLoader에서 배치 하나 추출하여 형태 확인 (None 체크 추가)
    if train_loader is not None:
        for X_batch, y_batch in train_loader:
            print(f"[DEBUG] X_batch 형태: {X_batch.shape}")
            print(f"[DEBUG] y_batch 형태: {y_batch.shape}")
            break
    
    if test_loader is not None:
        for X_batch, y_batch in test_loader:
            print(f"[DEBUG] 테스트 X_batch 형태: {X_batch.shape}")
            print(f"[DEBUG] 테스트 y_batch 형태: {y_batch.shape}")
            break