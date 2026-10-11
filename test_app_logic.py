import os
import pandas as pd
import numpy as np
import joblib

def test_app_core_logic():
    print("Testing core logic...")
    
    # 1. Dataset loading
    csv_path = 'Dataset/student-mat.csv'
    assert os.path.exists(csv_path), "Dataset missing"
    df = pd.read_csv(csv_path, sep=';')
    assert len(df) == 649, f"Expected 649 rows, got {len(df)}"
    print("[PASS] Test 1: Default dataset loaded.")
    
    # 2. Model loading
    model_path = 'Model/best_model.pkl'
    assert os.path.exists(model_path), "Model file missing"
    model = joblib.load(model_path)
    print("[PASS] Test 2: Trained model loaded.")
    
    # 3. Model feature compatibility & Single prediction
    expected_cols = df.drop(columns=['G1', 'G2', 'G3']).columns.tolist()
    sample_single = df[expected_cols].iloc[0:1]
    pred = model.predict(sample_single)
    assert len(pred) == 1, "Prediction output shape mismatch"
    assert 0 <= pred[0] <= 20, f"Unusual predicted grade: {pred[0]}"
    print(f"[PASS] Test 3: Single prediction output: {pred[0]:.2f}")
    
    # 4. Batch prediction
    sample_batch = df[expected_cols].iloc[0:10].copy()
    batch_preds = model.predict(sample_batch)
    assert len(batch_preds) == 10
    sample_batch['predicted_final_grade'] = np.clip(batch_preds, 0, 20).round(2)
    print(f"[PASS] Test 4: Batch prediction for 10 records successful.")
    
    # 5. Missing columns validation
    invalid_batch = sample_batch.drop(columns=['age', 'studytime'])
    missing_cols = [c for c in expected_cols if c not in invalid_batch.columns]
    assert len(missing_cols) == 2
    print(f"[PASS] Test 5: Incompatible schema properly flagged: {missing_cols}")
    
    # 6. Model evaluation results
    eval_path = 'Output/model_results.csv'
    assert os.path.exists(eval_path), "model_results.csv missing"
    eval_df = pd.read_csv(eval_path)
    assert 'Model' in eval_df.columns and 'R2 Score' in eval_df.columns
    print(f"[PASS] Test 6: Model evaluation table verified with {len(eval_df)} models.")
    
    # 7. Test with user's 8-row sample CSV (without G3)
    sample_8row_path = 'Dataset/sample_student_test_8rows.csv'
    assert os.path.exists(sample_8row_path), "8-row sample CSV missing"
    df_8rows = pd.read_csv(sample_8row_path)
    assert len(df_8rows) == 8, f"Expected 8 rows, got {len(df_8rows)}"
    assert 'G3' not in df_8rows.columns, "G3 should not be in test CSV"
    
    # Check that required features match
    missing_in_8rows = [col for col in expected_cols if col not in df_8rows.columns]
    assert len(missing_in_8rows) == 0, f"Missing features in 8-row sample: {missing_in_8rows}"
    
    # Generate predictions on the 8-row dataset
    preds_8rows = model.predict(df_8rows[expected_cols])
    assert len(preds_8rows) == 8, f"Expected 8 predictions, got {len(preds_8rows)}"
    clipped_8rows = np.clip(preds_8rows, 0.0, 20.0).round(2)
    df_8rows_result = df_8rows.copy()
    df_8rows_result['predicted_final_grade'] = clipped_8rows
    print(f"[PASS] Test 7: 8-row sample predictions generated successfully:")
    print("      Predicted grades:", clipped_8rows.tolist())
    
    print("\nALL AUTOMATED TESTS PASSED!")

if __name__ == '__main__':
    test_app_core_logic()
