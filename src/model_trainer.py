"""
Model Trainer & Evaluator Module (v2)
Part 5 of Mobile Biometrics ML Project

Model eğitimi, hyperparameter tuning ve evaluation (EER/FAR/FRR dahil).
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
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import LabelBinarizer

from preprocessor import Preprocessor


class BiometricEvaluator:
    """EER, FAR, FRR hesaplama sınıfı."""
    
    @staticmethod
    def calculate_far_frr(y_true: np.ndarray, y_scores: np.ndarray, 
                          threshold: float) -> Tuple[float, float]:
        """
        Belirli bir threshold için FAR ve FRR hesaplar.
        
        FAR (False Acceptance Rate): Yanlış kabul oranı
        FRR (False Rejection Rate): Yanlış red oranı
        """
        y_pred = (y_scores >= threshold).astype(int)
        
        # True positives, false positives, etc.
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # FAR = FP / (FP + TN) - impostor kabul edilme oranı
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # FRR = FN / (FN + TP) - genuine reddedilme oranı
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return far, frr
    
    @staticmethod
    def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """
        Equal Error Rate (EER) hesaplar.
        
        EER: FAR = FRR olduğu nokta
        
        Returns:
            (eer, threshold): EER değeri ve bu değerin elde edildiği threshold
        """
        thresholds = np.linspace(0, 1, 1000)
        
        min_diff = float('inf')
        eer = 0
        eer_threshold = 0.5
        
        for thresh in thresholds:
            far, frr = BiometricEvaluator.calculate_far_frr(y_true, y_scores, thresh)
            
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
                eer_threshold = thresh
        
        return eer, eer_threshold
    
    @staticmethod
    def calculate_multiclass_eer_far_frr(y_true: np.ndarray, 
                                          y_proba: np.ndarray,
                                          classes: np.ndarray) -> Dict:
        """
        Multi-class problem için EER, FAR, FRR hesaplar.
        
        Her sınıf için one-vs-rest yaklaşımı kullanılır.
        """
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        
        if len(classes) == 2:
            y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        
        class_metrics = {}
        eers = []
        fars = []
        frrs = []
        
        for i, cls in enumerate(classes):
            y_true_cls = y_true_bin[:, i]
            y_scores_cls = y_proba[:, i]
            
            eer, threshold = BiometricEvaluator.calculate_eer(y_true_cls, y_scores_cls)
            far, frr = BiometricEvaluator.calculate_far_frr(y_true_cls, y_scores_cls, threshold)
            
            class_metrics[str(cls)] = {
                'eer': eer,
                'far': far,
                'frr': frr,
                'threshold': threshold
            }
            
            eers.append(eer)
            fars.append(far)
            frrs.append(frr)
        
        return {
            'per_class': class_metrics,
            'mean_eer': np.mean(eers),
            'mean_far': np.mean(fars),
            'mean_frr': np.mean(frrs),
            'std_eer': np.std(eers)
        }


class ModelTrainer:
    """Model eğitimi ve evaluation sınıfı (hyperparameter tuning ile)."""
    
    def __init__(self, random_state: int = 42, use_hyperparameter_tuning: bool = True):
        self.random_state = random_state
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluator = BiometricEvaluator()
    
    def get_models_with_params(self) -> Dict:
        """
        Modeller ve hyperparameter grid'leri döndürür.
        """
        return {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=self.random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
    
    def get_default_models(self) -> Dict:
        """Default modeller (tuning olmadan)."""
        return {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=self.random_state, n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf', C=1.0, gamma='scale',
                probability=True, random_state=self.random_state
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5, weights='distance', n_jobs=-1
            )
        }
    
    def hyperparameter_tuning(self, 
                               model, 
                               params: Dict,
                               X_train: pd.DataFrame, 
                               y_train: pd.Series,
                               model_name: str) -> Tuple:
        """
        GridSearchCV ile hyperparameter tuning yapar.
        """
        print(f"\n  Tuning {model_name}...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            model, params, cv=cv, scoring='accuracy',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"    Best params: {grid_search.best_params_}")
        print(f"    Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def train_all_models(self, 
                         X_train: pd.DataFrame, 
                         y_train: pd.Series) -> Dict:
        """
        Tüm modelleri eğitir (opsiyonel hyperparameter tuning ile).
        """
        results = {}
        
        if self.use_hyperparameter_tuning:
            models_with_params = self.get_models_with_params()
            
            for name, config in models_with_params.items():
                best_model, best_params, best_score = self.hyperparameter_tuning(
                    config['model'], config['params'],
                    X_train, y_train, name
                )
                
                self.models[name] = best_model
                results[name] = {
                    'cv_score': best_score,
                    'best_params': best_params
                }
        else:
            models = self.get_default_models()
            
            for name, model in models.items():
                print(f"\n  Training {model_name}...")
                model.fit(X_train, y_train)
                
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
                
                self.models[name] = model
                results[name] = {
                    'cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                print(f"    CV Accuracy: {cv_scores.mean():.4f}")
        
        self.results = results
        return results
    
    def evaluate_model_with_biometrics(self, 
                                        model, 
                                        X_test: pd.DataFrame, 
                                        y_test: pd.Series,
                                        model_name: str) -> Dict:
        """
        Modeli değerlendirir (EER, FAR, FRR dahil).
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # EER, FAR, FRR hesapla
        classes = model.classes_
        biometric_metrics = BiometricEvaluator.calculate_multiclass_eer_far_frr(
            y_test.values, y_proba, classes
        )
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'eer': biometric_metrics['mean_eer'],
            'far': biometric_metrics['mean_far'],
            'frr': biometric_metrics['mean_frr'],
            'eer_std': biometric_metrics['std_eer'],
            'biometric_details': biometric_metrics
        }
    
    def evaluate_all_models(self, 
                            X_test: pd.DataFrame, 
                            y_test: pd.Series) -> Dict:
        """
        Tüm modelleri değerlendirir.
        """
        all_results = {}
        best_accuracy = 0
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        for name, model in self.models.items():
            result = self.evaluate_model_with_biometrics(model, X_test, y_test, name)
            all_results[name] = result
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {result['accuracy']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  F1-Score:  {result['f1_score']:.4f}")
            print(f"  EER:       {result['eer']:.4f} (+/- {result['eer_std']:.4f})")
            print(f"  FAR:       {result['far']:.4f}")
            print(f"  FRR:       {result['frr']:.4f}")
            
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n{'='*70}")
        print(f"Best Model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        print("=" * 70)
        
        return all_results
    
    def generate_results_table(self, results: Dict) -> str:
        """
        Sonuçları markdown tablosu olarak formatlar.
        """
        table = "| Model | Accuracy | Precision | Recall | F1-Score | EER | FAR | FRR |\n"
        table += "|-------|----------|-----------|--------|----------|-----|-----|-----|\n"
        
        for name, r in results.items():
            table += f"| {name} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            table += f"{r['recall']:.4f} | {r['f1_score']:.4f} | "
            table += f"{r['eer']:.4f} | {r['far']:.4f} | {r['frr']:.4f} |\n"
        
        return table
    
    def save_results(self, results: Dict, output_path: str):
        """Sonuçları kaydeder."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Serialize edilebilir hale getir
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    serializable_results[model_name][k] = v.tolist()
                elif isinstance(v, dict):
                    # Nested dict'leri de dönüştür
                    serializable_results[model_name][k] = {
                        str(kk): (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items()
                    }
                else:
                    serializable_results[model_name][k] = v
        
        # JSON kaydet
        with open(output_path / 'evaluation_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Markdown tablo kaydet
        table = self.generate_results_table(results)
        with open(output_path / 'results_table.md', 'w') as f:
            f.write("# Model Evaluation Results\n\n")
            f.write(f"**Date:** 2026-01-19\n\n")
            f.write(table)
        
        print(f"Results saved to: {output_path}")
    
    def save_model(self, output_path: str, model_name: Optional[str] = None):
        """Modeli kaydeder."""
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
    """Ana fonksiyon - tam pipeline (hyperparameter tuning + EER/FAR/FRR)."""
    project_root = Path(__file__).parent.parent
    features_path = project_root / "output" / "features" / "session_features.csv"
    models_path = project_root / "output" / "models"
    reports_path = project_root / "output" / "reports"
    
    print("=" * 70)
    print("Mobile Biometrics ML - Model Training & Evaluation")
    print("(with Hyperparameter Tuning and EER/FAR/FRR metrics)")
    print("=" * 70)
    
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
    
    # 3. Model Training with Hyperparameter Tuning
    print("\n3. Training models with hyperparameter tuning...")
    trainer = ModelTrainer(random_state=42, use_hyperparameter_tuning=True)
    trainer.train_all_models(data['X_train'], data['y_train'])
    
    # 4. Evaluation (EER, FAR, FRR dahil)
    print("\n4. Evaluating models (including EER/FAR/FRR)...")
    results = trainer.evaluate_all_models(data['X_test'], data['y_test'])
    
    # 5. Sonuçları kaydet
    print("\n5. Saving results...")
    trainer.save_results(results, str(reports_path))
    trainer.save_model(str(models_path))
    
    # 6. Sonuç tablosu göster
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(trainer.generate_results_table(results))
    
    print("Pipeline completed successfully!")
    
    return trainer, results


if __name__ == "__main__":
    main()
