import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import shap
import joblib


def evaluate_model(model, X_test, y_test, model_name, feature_names=None, save_path=None):
    """
    Evaluate a classification model and plot confusion matrix and PR curve.
    Optionally save plots to disk.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f'{model_name} F1-Score: {f1:.4f}')
    print(f'{model_name} AUC-PR: {ap:.4f}')

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f'{model_name} Confusion Matrix')
    if save_path:
        plt.savefig(f"{save_path}/{model_name}_confusion_matrix.png")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label=f'{model_name} PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}/{model_name}_pr_curve.png")
    plt.show()

    return {'f1': f1, 'auc_pr': ap, 'confusion_matrix': cm}


def run_shap_analysis(model, X, feature_names, save_path=None, batch_size=1000):
    """
    Run SHAP analysis for a tree-based model with batching for large datasets.
    Generates and saves summary plot.
    """
    print(f"🔍 Starting SHAP analysis for {X.shape[0]} samples...")
    
    explainer = shap.TreeExplainer(model)
    n_samples = X.shape[0]
    
    if n_samples > batch_size:
        print(f"📊 Processing in batches of {batch_size}...")
        shap_values = []
        total_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(0, n_samples, batch_size):
            batch_num = i // batch_size + 1
            batch = X[i:i+batch_size]
            print(f"   Processing batch {batch_num}/{total_batches} ({batch.shape[0]} samples)...")
            
            # Get SHAP values for the batch
            batch_shap = explainer.shap_values(batch)
            
            # Handle different SHAP value formats
            if isinstance(batch_shap, list):
                # For tree-based models, SHAP returns [shap_values_class_0, shap_values_class_1]
                batch_shap_positive = batch_shap[1]  # Extract positive class
            elif len(batch_shap.shape) == 3:
                # If shape is (n_samples, n_features, n_classes)
                batch_shap_positive = batch_shap[:, :, 1]  # Extract positive class
            else:
                # Default case
                batch_shap_positive = batch_shap
            
            shap_values.append(batch_shap_positive)
            
            # Show progress percentage
            progress = (batch_num / total_batches) * 100
            print(f"   Progress: {progress:.1f}% complete")
        
        print("🔗 Combining all batches...")
        shap_values = np.vstack(shap_values)
    else:
        print(f"📊 Processing all {n_samples} samples at once...")
        shap_values = explainer.shap_values(X)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For tree-based models, SHAP returns [shap_values_class_0, shap_values_class_1]
            shap_values = shap_values[1]  # Extract positive class
        elif len(shap_values.shape) == 3:
            # If shape is (n_samples, n_features, n_classes)
            shap_values = shap_values[:, :, 1]  # Extract positive class
    
    # Ensure we have the correct shape
    print(f"📊 SHAP values shape: {shap_values.shape}")
    print(f"📊 X data shape: {X.shape}")
    
    # Verify shapes match
    if shap_values.shape[1] != X.shape[1]:
        print(f"⚠️  Shape mismatch! SHAP: {shap_values.shape[1]}, X: {X.shape[1]}")
        print("🔧 Attempting to fix shape...")
        
        # If SHAP values have wrong shape, try to reshape
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # Extract positive class
        elif shap_values.shape[1] == 2:
            # If we have (n_samples, 2), this is wrong - we need (n_samples, n_features)
            print("❌ SHAP values have wrong shape. Cannot proceed.")
            return None
    
    print("📈 Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    if save_path:
        plt.savefig(f"{save_path}/shap_summary.png")
        print(f"💾 Plot saved to {save_path}/shap_summary.png")
    
    plt.show()
    print("✅ SHAP analysis completed!")
    return shap_values


def hyperparameter_tuning_rf(X, y, param_grid=None, cv=3, n_jobs=-1, random_state=42):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.
    Returns the best estimator and parameters.
    """
    from sklearn.ensemble import RandomForestClassifier
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        }
    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=random_state),
        param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=n_jobs
    )
    grid_search.fit(X, y)
    print("Best Random Forest parameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.best_params_


def hyperparameter_tuning_logreg(X, y, param_grid=None, cv=3, n_jobs=-1, random_state=42):
    """
    Perform hyperparameter tuning for Logistic Regression using GridSearchCV.
    Returns the best estimator and parameters.
    """
    from sklearn.linear_model import LogisticRegression
    if param_grid is None:
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [1000]
        }
    grid_search = GridSearchCV(
        LogisticRegression(class_weight='balanced', random_state=random_state),
        param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=n_jobs
    )
    grid_search.fit(X, y)
    print("Best Logistic Regression parameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.best_params_


def save_model(model, path):
    """
    Save a trained model to disk using joblib.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def save_metrics(metrics_dict, path):
    """
    Save metrics dictionary to disk as a numpy file.
    """
    np.save(path, metrics_dict)
    print(f"Metrics saved to {path}") 