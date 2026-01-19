# Makine Öğrenmesi Eğitim Dokümanı

Mobil Biyometri Projesi için ML kavramları ve uygulamaları.

---

## 1. Makine Öğrenmesi Temelleri

### 1.1 Supervised Learning (Denetimli Öğrenme)

**Tanım:** Etiketli verilerle eğitim yapılır. Model, girdi-çıktı ilişkisini öğrenir.

```python
# Eğitim verisi
X_train = [[1, 2], [3, 4], [5, 6]]  # Özellikler
y_train = ['A', 'B', 'A']           # Etiketler

# Model eğitimi
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)
```

**Classification vs Regression:**
- **Classification:** Kategorik çıktı (kullanıcı kimliği, spam/not spam)
- **Regression:** Sürekli çıktı (fiyat tahmini, sıcaklık)

---

## 2. Kullanılan Algoritmalar

### 2.1 Random Forest

**Çalışma Prensibi:**
- Birden fazla karar ağacı oluşturur
- Her ağaç farklı veri alt kümesiyle eğitilir
- Sonuçlar oylama ile birleştirilir (ensemble)

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,    # Ağaç sayısı
    max_depth=10,        # Maksimum derinlik
    random_state=42
)
model.fit(X_train, y_train)
```

**Avantajları:**
- Overfitting'e dirençli
- Feature importance verir
- Kategorik ve numeric verilerle çalışır

**Projemizdeki Sonuç:** %98.33 accuracy

---

### 2.2 SVM (Support Vector Machine)

**Çalışma Prensibi:**
- Sınıflar arasında maksimum margin bulur
- RBF kernel ile non-linear sınırlar çizer

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',        # Radial Basis Function
    C=10,                # Regularization parametresi
    gamma='scale',       # Kernel katsayısı
    probability=True
)
model.fit(X_train, y_train)
```

**Avantajları:**
- Yüksek boyutlu verilerde etkili
- Bellek verimli (sadece support vectors saklanır)

**Projemizdeki Sonuç:** %98.33 accuracy, EER: 0.02%

---

### 2.3 k-NN (k-Nearest Neighbors)

**Çalışma Prensibi:**
- Test noktasına en yakın k komşuyu bulur
- Çoğunluk oyu ile sınıf atar

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance',  # Yakın komşulara daha fazla ağırlık
    metric='manhattan'   # Mesafe metriği
)
model.fit(X_train, y_train)
```

**Avantajları:**
- Basit, sezgisel
- Eğitim gerektirmez (lazy learning)

**Dezavantajları:**
- Büyük verilerde yavaş
- Boyut lanetine duyarlı

**Projemizdeki Sonuç:** %97.00 accuracy

---

## 3. Veri Ön İşleme

### 3.1 Feature Scaling

**Neden Gerekli?**
- SVM ve k-NN mesafe bazlı çalışır
- Farklı ölçeklerdeki özellikler hatalı mesafe hesabına yol açar

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Test verisi için sadece transform (fit değil!)
X_test_scaled = scaler.transform(X_test)
```

**StandardScaler:** z = (x - μ) / σ  → Mean=0, Std=1

---

### 3.2 Zero Variance Feature Removal

**Problem:** Sabit değerli özellikler hiçbir bilgi taşımaz.

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
```

**Projemizdeki Örnek:** `pressure` sütunu tüm kayıtlarda 1.0 → çıkarıldı

---

### 3.3 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # %20 test
    random_state=42,    # Tekrarlanabilirlik
    stratify=y          # Sınıf dağılımını koru
)
```

---

## 4. Hyperparameter Tuning

### 4.1 GridSearchCV

Tüm parametre kombinasyonlarını dener:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,           # 5-fold cross-validation
    scoring='accuracy'
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")
```

**Projemizdeki Sonuçlar:**
| Model | Best Params |
|-------|-------------|
| RF | n_estimators=200, max_depth=None |
| SVM | C=10, kernel=rbf, gamma=scale |
| k-NN | n_neighbors=3, weights=distance |

---

## 5. Evaluation Metrikleri

### 5.1 Temel Metrikler

```
             Predicted
             Positive  Negative
Actual  Positive   TP       FN
        Negative   FP       TN
```

| Metrik | Formül | Anlamı |
|--------|--------|--------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Genel doğruluk |
| **Precision** | TP/(TP+FP) | Pozitif tahminlerin doğruluğu |
| **Recall** | TP/(TP+FN) | Gerçek pozitiflerin bulunma oranı |
| **F1-Score** | 2×(P×R)/(P+R) | Precision-Recall harmonik ortalaması |

---

### 5.2 Biyometrik Metrikler

#### FAR (False Acceptance Rate)
```
FAR = FP / (FP + TN)
```
**Anlamı:** Impostor'ın yanlışlıkla kabul edilme oranı.
**İdeal:** FAR → 0

#### FRR (False Rejection Rate)
```
FRR = FN / (FN + TP)
```
**Anlamı:** Genuine kullanıcının yanlışlıkla reddedilme oranı.
**İdeal:** FRR → 0

#### EER (Equal Error Rate)
```
EER = FAR = FRR (eşleştikleri noktada)
```
**Anlamı:** FAR ve FRR'nin eşit olduğu nokta.
**İdeal:** EER → 0 (mükemmel sistem)

**Projemizdeki Sonuçlar:**
| Model | EER | FAR | FRR |
|-------|-----|-----|-----|
| SVM | 0.02% | 0.05% | 0.00% |
| RF | 0.06% | 0.11% | 0.00% |
| k-NN | 0.37% | 0.08% | 0.67% |

---

## 6. Cross-Validation

### Neden Gerekli?
- Tek bir train/test split varyansa duyarlı
- Tüm verinin hem train hem test olarak kullanılması

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

```
Fold 1: Train [2,3,4,5] → Test [1]
Fold 2: Train [1,3,4,5] → Test [2]
Fold 3: Train [1,2,4,5] → Test [3]
...
```

---

## 7. Feature Engineering

### 7.1 Timing Features (Touch-based)

```python
# Dwell Time: Tuşa basılı kalma süresi
dwell_time = t_release - t_press

# Flight Time: Tuşlar arası bekleme
flight_time = t_press_next - t_release_prev

# Inter-key Latency: Basımlar arası süre
latency = t_press_next - t_press_current
```

### 7.2 Sensor Features

```python
# Sensör penceresinden istatistikler
features = {
    'mean': np.mean(sensor_window),
    'std': np.std(sensor_window),
    'min': np.min(sensor_window),
    'max': np.max(sensor_window),
    'magnitude': np.sqrt(x**2 + y**2 + z**2)
}
```

---

## 8. Pratik İpuçları

### Overfitting Belirtileri
- Train accuracy çok yüksek, test accuracy düşük
- Çözüm: Regularization, daha fazla veri, daha basit model

### Underfitting Belirtileri
- Hem train hem test accuracy düşük
- Çözüm: Daha karmaşık model, daha fazla özellik

### Model Seçimi
1. Baseline olarak Random Forest veya SVM ile başla
2. Hyperparameter tuning yap
3. EER/FAR/FRR kontrol et (biyometrik için kritik)

---

## 9. Kod Örneği: Tam Pipeline

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Veri yükle
df = pd.read_csv('session_features.csv')
X = df.drop(['user', 'session'], axis=1)
y = df['user']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = grid.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

---

## 10. Kaynaklar

1. **Scikit-learn Documentation:** https://scikit-learn.org
2. **Hands-On Machine Learning** - Aurélien Géron
3. **Pattern Recognition and ML** - Christopher Bishop

---

**Son Güncelleme:** 2026-01-19
