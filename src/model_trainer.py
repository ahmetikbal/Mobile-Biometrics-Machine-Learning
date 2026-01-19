"""
Model Trainer & Evaluator Module
Part 5 of Mobile Biometrics ML Project

Model eğitimi, hyperparameter tuning ve evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# XGBoost opsiyonel
HAS_XGBOOST = False
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except (ImportError, Exception):
    print("Note: XGBoost not available, using other classifiers.")
    pass

from preprocessor import Preprocessor


class ModelTrainer:
    """Model eğitimi ve evaluation sınıfı."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def get_models(self) -> Dict:
        """Kullanılacak modelleri döndürür."""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        
        return models
    
    def train_single_model(self, 
                           model, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           model_name: str) -> Dict:
        """
        Tek bir modeli eğitir ve sonuçları döndürür.
        """
        print(f"\n  Training {model_name}...")
        
        # Eğitim
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        return {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def train_all_models(self, 
                         X_train: pd.DataFrame, 
                         y_train: pd.Series) -> Dict:
        """
        Tüm modelleri eğitir.
        """
        models = self.get_models()
        results = {}
        
        for name, model in models.items():
            result = self.train_single_model(model, X_train, y_train, name)
            self.models[name] = result['model']
            results[name] = {
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
            print(f"    CV Accuracy: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        
        self.results = results
        return results
    
    def evaluate_model(self, 
                       model, 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series,
                       model_name: str) -> Dict:
        """
        Modeli değerlendirir.
        """
        y_pred = model.predict(X_test)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def evaluate_all_models(self, 
                            X_test: pd.DataFrame, 
                            y_test: pd.Series) -> Dict:
        """
        Tüm modelleri değerlendirir.
        """
        all_results = {}
        best_accuracy = 0
        
        print("\nEvaluation Results:")
        print("-" * 50)
        
        for name, model in self.models.items():
            result = self.evaluate_model(model, X_test, y_test, name)
            all_results[name] = result
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {result['accuracy']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  F1-Score:  {result['f1_score']:.4f}")
            
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return all_results
    
    def save_results(self, results: Dict, output_path: str):
        """
        Sonuçları kaydeder.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix'leri numpy'dan listeye çevir
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in result.items()
            }
        
        # JSON olarak kaydet
        with open(output_path / 'evaluation_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {output_path / 'evaluation_results.json'}")
    
    def save_model(self, output_path: str, model_name: Optional[str] = None):
        """
        Modeli kaydeder.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
        
        if model is None:
            print("No model to save!")
            return
        
        model_file = output_path / f'{model_name.lower()}_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to: {model_file}")


def main():
    """Ana fonksiyon - tam pipeline."""
    project_root = Path(__file__).parent.parent
    features_path = project_root / "output" / "features" / "session_features.csv"
    models_path = project_root / "output" / "models"
    reports_path = project_root / "output" / "reports"
    
    print("=" * 60)
    print("Mobile Biometrics ML - Model Training & Evaluation")
    print("=" * 60)
    
    # 1. Veri yükle
    print("\n1. Loading data...")
    df = pd.read_csv(features_path)
    print(f"   Loaded: {df.shape}")
    print(f"   Users: {df['user'].nunique()}")
    print(f"   Sessions per user: {df.groupby('user').size().mean():.0f}")
    
    # 2. Preprocessing
    print("\n2. Preprocessing...")
    preprocessor = Preprocessor(
        scaling_method='standard',
        variance_threshold=0.01,
        test_size=0.2,
        random_state=42
    )
    
    data = preprocessor.prepare_data(df)
    print(f"   Train: {data['X_train'].shape}")
    print(f"   Test: {data['X_test'].shape}")
    print(f"   Features: {len(preprocessor.feature_columns)}")
    print(f"   Removed (zero/near-zero variance): {len(preprocessor.removed_columns)}")
    
    # 3. Model Training
    print("\n3. Training models...")
    trainer = ModelTrainer(random_state=42)
    trainer.train_all_models(data['X_train'], data['y_train'])
    
    # 4. Evaluation
    print("\n4. Evaluating models...")
    results = trainer.evaluate_all_models(data['X_test'], data['y_test'])
    
    # 5. Sonuçları kaydet
    print("\n5. Saving results...")
    trainer.save_results(results, str(reports_path))
    trainer.save_model(str(models_path))
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return trainer, results


if __name__ == "__main__":
    main()
