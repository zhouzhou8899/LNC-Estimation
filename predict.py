import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA

warnings.filterwarnings('ignore')

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
        self.X = X
        self.selected_features = col_labels
        self.pca = SKPCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_
        correlation_loadings = np.zeros((X.shape[1], self.pca.components_.shape[0]))
        for i in range(X.shape[1]):
            for j in range(self.pca.components_.shape[0]):
                correlation_loadings[i, j] = np.corrcoef(X[:, i], X_pca[:, j])[0, 1]
        self.loadings_ = pd.DataFrame(
            correlation_loadings,
            index=col_labels,
            columns=[f'PC{i+1}' for i in range(correlation_loadings.shape[1])]
        )
        return {'PC': X_pca, 'loadings': self.loadings_, 'explained_var': self.explained_variance_ratio_}

    def transform(self, X, row_labels=None, col_labels=None):
        return self.pca.transform(X)

def load_model(model_path):
    """加载模型包"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model. Ensure CustomPCA class is defined. Error: {e}")

def predict(model_package, input_data_path, output_path='prediction_result.csv'):
    print(f"Loading data from {input_data_path}...")
    df = pd.read_csv(input_data_path)
    print(f"Data loaded: {df.shape}")
    feature_key = 'selected_features_final' if 'selected_features_final' in model_package else 'selected_features'
    required_features = model_package[feature_key]
    print(f"Model requires {len(required_features)} features after PCA selection")
    missing_cols = set(required_features) - set(df.columns)
    if missing_cols:
        print(f"⚠️  Warning: Input data missing {len(missing_cols)} features")
        print(f"   Missing: {list(missing_cols)[:5]}...")
        print(f"   Available columns in input: {list(df.columns)[:10]}...")
        raise ValueError(f"Input data missing required features: {missing_cols}")
    X = df[required_features].values
    print(f"Features aligned: {X.shape}")
    
    scaler_X = model_package.get('scaler_X')
    scaler_y = model_package.get('scaler_y')
    if scaler_X is None or scaler_y is None:
        raise ValueError("Model package missing scalers")
    X_scaled = scaler_X.transform(X)
    
    has_feature_indices = all(key in model_package for key in ['rf_indices', 'svm_indices', 'xgb_indices'])
    
    if has_feature_indices:
        print("Using model-specific feature indices (from forward selection)")
        X_rf = X_scaled[:, model_package['rf_indices']]
        X_svm = X_scaled[:, model_package['svm_indices']]
        X_xgb = X_scaled[:, model_package['xgb_indices']]
    else:
        print("Using all features for all base models")
        X_rf = X_svm = X_xgb = X_scaled
    
    rf_model = model_package.get('rf_model')
    svm_model = model_package.get('svm_model')
    xgb_model = model_package.get('xgb_model')
    
    if not all([rf_model, svm_model, xgb_model]):
        raise ValueError("Model package missing base models")
    
    pred_rf = rf_model.predict(X_rf)
    pred_svm = svm_model.predict(X_svm)
    pred_xgb = xgb_model.predict(X_xgb)
    
    weights_key = 'meta_model_weights' if 'meta_model_weights' in model_package else 'weights'
    weights = model_package[weights_key]
    
    ensemble_pred_scaled = (pred_rf * weights[0] + 
                            pred_svm * weights[1] + 
                            pred_xgb * weights[2])
    
    final_pred = scaler_y.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).flatten()
    
    result_df = df.copy()
    result_df['Predicted_LNC'] = final_pred
    result_df.to_csv(output_path, index=False)
    print(f"✓ Prediction completed. Results saved to {output_path}")
    print(f"  Predictions range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    return final_pred

if __name__ == "__main__":
    # ================= Configuration Area (Reviewers may edit this section) =================
    # Select the model file to test
    model_path = 'model_weights/baoding_ensemble_model.pkl'
    # model_path = 'model_weights/lnc_prediction_unified_transfer_Wei_county_JND36.pkl'
    
    input_path = 'sample_data/test_input.csv'



