import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA

warnings.filterwarnings('ignore')

# ======================== CustomPCA Class (Required for loading models) ========================
class CustomPCA:
    def __init__(self, n_components=0.85, n_feat=10, method="pca"):
        self.n_components = n_components
        self.n_feat = n_feat
        self.method = method
        self.pca = None
        self.explained_variance_ratio_ = None
        self.loadings_ = None
        self.components_ = None
        self.X = None
        self.selected_features = None

    def fit_transform(self, X, row_labels=None, col_labels=None):
        pass

    def transform(self, X, row_labels=None, col_labels=None):
        pass

# ======================== Load Model Function ========================
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model. Ensure CustomPCA class is defined. Error: {e}")

# ======================== Predict Function ========================
def predict(model_package, input_data_path, output_path='prediction_result.csv'):
    print(f"Loading data from {input_data_path}...")
    df = pd.read_csv(input_data_path)
    print(f"Data loaded: {df.shape}")
    
    # Get required feature names from model
    feature_key = 'selected_features_final' if 'selected_features_final' in model_package else 'selected_features'
    required_features = model_package[feature_key]
    print(f"Model requires {len(required_features)} features after PCA selection")
    
    # Check if input data contains all required features
    missing_cols = set(required_features) - set(df.columns)
    if missing_cols:
        print(f"Warning: Input data missing {len(missing_cols)} features")
        print(f"   Missing: {list(missing_cols)[:5]}...")
        raise ValueError(f"Input data missing required features: {missing_cols}")
    
    # Align features (in the same order as model)
    X = df[required_features].values
    print(f"Features aligned: {X.shape}")
    
    # Get preprocessors
    scaler_X = model_package.get('scaler_X')
    scaler_y = model_package.get('scaler_y')
    if scaler_X is None or scaler_y is None:
        raise ValueError("Model package missing scalers")
    
    # Standardize
    X_scaled = scaler_X.transform(X)
    print(f"Data standardized: {X_scaled.shape}")
    
    # Check if feature indices exist for each base model
    has_feature_indices = all(key in model_package for key in ['rf_indices', 'svm_indices', 'xgb_indices'])
    
    if has_feature_indices:
        print("Using model-specific feature indices (from forward selection)")
        print(f"  RF uses {len(model_package['rf_indices'])} features")
        print(f"  SVM uses {len(model_package['svm_indices'])} features")
        print(f"  XGB uses {len(model_package['xgb_indices'])} features")
        
        # Each base model uses different feature subset
        X_rf = X_scaled[:, model_package['rf_indices']]
        X_svm = X_scaled[:, model_package['svm_indices']]
        X_xgb = X_scaled[:, model_package['xgb_indices']]
        
        print(f"  RF input shape: {X_rf.shape}")
        print(f"  SVM input shape: {X_svm.shape}")
        print(f"  XGB input shape: {X_xgb.shape}")
    else:
        print("Warning: No feature indices found in model file!")
        print("   Using all features for all base models (may cause errors)")
        X_rf = X_svm = X_xgb = X_scaled
    
    # Get base models
    rf_model = model_package.get('rf_model')
    svm_model = model_package.get('svm_model')
    xgb_model = model_package.get('xgb_model')
    
    if not all([rf_model, svm_model, xgb_model]):
        raise ValueError("Model package missing base models")
    
    # Base model predictions
    print("\nRunning predictions...")
    pred_rf = rf_model.predict(X_rf)
    pred_svm = svm_model.predict(X_svm)
    pred_xgb = xgb_model.predict(X_xgb)
    
    # Get ensemble weights
    weights_key = 'meta_model_weights' if 'meta_model_weights' in model_package else 'weights'
    weights = model_package[weights_key]
    
    # Ensemble weighted average
    ensemble_pred_scaled = (pred_rf * weights[0] + 
                            pred_svm * weights[1] + 
                            pred_xgb * weights[2])
    
    # Inverse standardization to get final LNC values
    final_pred = scaler_y.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).flatten()
    
    # Save results
    result_df = df.copy()
    result_df['Predicted_LNC'] = final_pred
    result_df.to_csv(output_path, index=False)
    print(f"\nPrediction completed. Results saved to {output_path}")
    print(f"  Predictions range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    return final_pred

# ======================== Main Program ========================
if __name__ == "__main__":
    # ================= Reviewers edit this section =================
    # Select the model file to test
    # Option 1: Baoding ensemble model
    # model_path = 'model_weights/baoding_ensemble_model.pkl'
    
    # Option 2: Transfer learning model (Wei County Variety A: JND36/SK126)
    model_path = 'model_weights/lnc_prediction_unified_transfer_Wei_county_JND36.pkl'
    
    # Input data path
    input_path = 'sample_data/test_input.csv'
    output_path = 'prediction_result.csv'
    try:
        print(f"Loading model from {model_path}...")
        model_package = load_model(model_path)
        print("✅ Model loaded successfully")
        
        # 执行预测
        final_predictions = predict(model_package, input_path, output_path)
        print("\n✅ Prediction finished!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    
