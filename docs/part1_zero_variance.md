# Part 1: Zero Variance Features Analysis Report

Bu rapor, dataset'teki tüm özelliklerin variance analizini içerir.

## Özet

- **Toplam özellik sayısı:** 32
- **Zero variance özellikler:** 5
- **Near-zero variance özellikler:** 12

## Zero Variance Özellikler

Bu özellikler hiçbir ayırt edici bilgi taşımaz ve çıkarılmalıdır.

| Dataset | Sütun | Unique Count | Variance |
|---------|-------|--------------|----------|
| touch_events | orientation | 1 | 0.000000 |
| touch_events | pressure | 1 | 0.000000 |
| touch_events | targetInput | 1 | 0.000000 |
| sensor_data | orientation | 1 | 0.000000 |
| device_info | orientation | 1 | 0.000000 |

## Near-Zero Variance Özellikler

Bu özellikler düşük varyansa sahip, dikkatli kullanılmalıdır.

| Dataset | Sütun | Unique Ratio | Freq Ratio | Variance |
|---------|-------|--------------|------------|----------|
| touch_events | sessionID | 0.0005 | 1.10 | 26.851093 |
| touch_events | repeatNo | 0.0005 | 1.10 | 26.851093 |
| touch_events | major | 0.0078 | 1.28 | 6.609955 |
| touch_events | minor | 0.0062 | 2.70 | 5.507823 |
| touch_events | size | 0.0043 | 1.11 | 0.000171 |
| touch_events | action | 0.0003 | 3.41 | 0.616672 |
| sensor_data | sessionID | 0.0000 | 1.27 | 27.575921 |
| sensor_data | headingAccuracy | 0.0000 | 1.45 | 0.225776 |
| sensor_data | angle | 0.0000 | 4.00 | 10897.971596 |
| device_info | widthPixels | 0.0400 | 9.00 | 11902.040816 |
| device_info | density | 0.0800 | 1.33 | 0.104791 |
| device_info | densityDpi | 0.0800 | 1.33 | 2682.653061 |

## Zero Variance Handling Yöntemleri

### 1. Variance Thresholding (Önerilen)

**Artıları:**
- Basit ve hızlı uygulama
- sklearn.feature_selection.VarianceThreshold ile kolay implementasyon
- Unsupervised - target değişkene ihtiyaç duymaz

**Eksileri:**
- Kategorik değişkenler için encoding gerektirir
- Threshold değeri manuel belirlenir

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0)  # Zero variance için
X_filtered = selector.fit_transform(X)
```

### 2. Doğrudan Silme

**Artıları:**
- En basit yöntem
- Model karmaşıklığını azaltır

**Eksileri:**
- Potansiyel bilgi kaybı
- Geri dönüşü yok

### 3. Tree-based Modellere Bırakma

**Artıları:**
- Random Forest, XGBoost gibi modeller otomatik ignore eder
- Ekstra preprocessing gerektirmez

**Eksileri:**
- Tüm modeller için geçerli değil (SVM, k-NN etkilenir)
- Gereksiz hesaplama maliyeti

### 4. Feature-engine DropConstantFeatures

**Artıları:**
- Hem constant hem quasi-constant features için
- Kategorik değişkenlerle çalışır

**Eksileri:**
- Ek kütüphane gerektirir

---

## Sonuç ve Öneriler

1. Yukarıda listelenen **zero variance** özellikler preprocessing aşamasında çıkarılmalıdır.
2. **Near-zero variance** özellikler için hedef değişkenle korelasyon kontrolü yapılmalıdır.
3. Bu projede **VarianceThreshold** kullanılması önerilir.
