# Part 2: Feature Derivation Literature Review

Mobil cihazlarda davranışsal biyometrik kimlik doğrulama için kullanılan derived feature'ların kapsamlı literatür taraması.

---

## 1. Touch-Based Derived Features

### 1.1 Dwell Time (Hold Time)

**Tanım:** Bir tuşa basılı kalma süresi - parmağın ekrana dokunmasından kalkmasına kadar geçen süre.

**Formül:** `Dwell Time = t_release - t_press`

**Dataset'te Hesaplama:**
```python
# touch_events.csv'den:
# action=0: touch starts, action=1: touch ends
dwell_time = timestamp_action1 - timestamp_action0
```

**Akademik Kullanım:**
- Killourhy & Maxion (2009) - Keystroke dynamics benchmark çalışmasında temel feature olarak kullanılmış [1]
- Teh et al. (2016) - Mobile touch dynamics survey'inde en önemli feature olarak belirtilmiş [2]

**Bizim Dataset İçin Uygunluk:** ✅ **Çok Uygun**

---

### 1.2 Flight Time

**Tanım:** Bir tuşun bırakılmasından sonraki tuşa basılmasına kadar geçen süre.

**Formül:** `Flight Time = t_press[i+1] - t_release[i]`

**Akademik Kullanım:**
- Frank et al. (2013) - Touchalytics çalışmasında yüksek ayırt edicilik göstermiş [3]
- Monaco et al. (2017) - PIN/pattern authentication'da kritik feature [4]

**Bizim Dataset İçin Uygunluk:** ✅ **Çok Uygun**

---

### 1.3 Inter-Key Latency

**Tanım:** Ardışık iki tuşa basma arası geçen toplam süre.

**Formül:** `Inter-Key Latency = t_press[i+1] - t_press[i]`

**Akademik Kullanım:**
- Antal et al. (2015) - Mobile keystroke dynamics'te yüksek EER performansı [5]

**Bizim Dataset İçin Uygunluk:** ✅ **Çok Uygun**

---

### 1.4 Touch Pressure

**Tanım:** Ekrana uygulanan basınç değeri (0-1 arası normalize).

**Dataset'teki Sütun:** `pressure`

**Akademik Kullanım:**
- De Luca et al. (2012) - Touch me once çalışmasında kullanılmış [6]

**Bizim Dataset İçin Uygunluk:** ⚠️ **Dikkatli Kullanılmalı**
- Bazı cihazlarda pressure değeri sabit dönebilir
- Zero variance analizi sonrası karar verilmeli

---

### 1.5 Touch Area (Major/Minor)

**Tanım:** Parmağın ekranla temas alanını temsil eden elips parametreleri.

**Android API:** `getTouchMajor()`, `getTouchMinor()`

**Akademik Kullanım:**
- Feng et al. (2014) - Continuous authentication'da kullanılmış [8]

**Bizim Dataset İçin Uygunluk:** ⚠️ **Dikkatli Kullanılmalı**
- Danışman notu: Bu verilerin sağlıklı olmadığı belirtilmiş

---

### 1.6 Touch Position Features

**Tanım:** Tuşa basılan x, y koordinatları ve bunlardan türetilen özellikler.

**Derived Features:**
- Position mean/variance per button
- Drift (aynı tuşun farklı basımları arası fark)
- Distance between consecutive touches

**Bizim Dataset İçin Uygunluk:** ✅ **Çok Uygun**

---

## 2. Sensor-Based Derived Features

### 2.1 Accelerometer Features

**Tanım:** Cihazın lineer ivme değerleri (x, y, z eksenleri).

| Feature | Formül | Açıklama |
|---------|--------|----------|
| Mean | μ = Σx/n | Ortalama ivme |
| Std | σ = √(Σ(x-μ)²/n) | Hareket tutarlılığı |
| Min/Max | min(x), max(x) | Extremum değerler |
| Range | max - min | Hareket genişliği |
| Magnitude | √(x²+y²+z²) | Toplam ivme |
| Energy | Σx² | Sinyal enerjisi |

**Akademik Kullanım:**
- Sitová et al. (2016) - HMOG dataset'inde temel feature'lar [12]
- Shen et al. (2016) - Motion-sensor behavior analysis [13]

**Bizim Dataset İçin Uygunluk:** ✅ **Çok Uygun**

---

### 2.2 Gyroscope Features

**Tanım:** Cihazın açısal hız değerleri (rotasyon).

**İstatistiksel Features:**
- Mean, Std, Min, Max, Range (her eksen için)
- Magnitude: √(ωx² + ωy² + ωz²)

**Akademik Kullanım:**
- Muaaz & Mayrhofer (2017) - Gait recognition'da accelerometer ile birlikte [14]

**Bizim Dataset İçin Uygunluk:** ✅ **Çok Uygun**

---

### 2.3 Magnetometer Features

**Tanım:** Manyetik alan ölçümleri (pusula).

**İstatistiksel Features:**
- Mean, Std, Min, Max (her eksen için)
- Magnetic field magnitude

**Akademik Kullanım:**
- Shen et al. (2018) - Multi-sensor fusion authentication [16]

**Bizim Dataset İçin Uygunluk:** ✅ **Uygun**

---

### 2.4 Sensor Magnitude Features

**Formül:**
```python
magnitude = np.sqrt(rawX**2 + rawY**2 + rawZ**2)
```

**Avantajları:**
- Cihaz oryantasyonundan bağımsız
- Feature boyutunu azaltır (3 → 1)

**Bizim Dataset İçin Uygunluk:** ✅ **Çok Uygun**

---

## 3. Dataset Applicability Summary

| Feature Category | Uygunluk | Notlar |
|-----------------|----------|--------|
| Dwell Time | ✅ | Temel feature |
| Flight Time | ✅ | Temel feature |
| Inter-Key Latency | ✅ | Temel feature |
| Pressure | ⚠️ | Zero variance kontrolü gerekli |
| Touch Area | ⚠️ | Danışman uyarısı var |
| Position (x, y) | ✅ | Kullanılabilir |
| Accelerometer | ✅ | Mükemmel veri kalitesi |
| Gyroscope | ✅ | Mükemmel veri kalitesi |
| Magnetometer | ✅ | Kullanılabilir |
| Magnitude | ✅ | Türetilebilir |

---

## 4. Recommended ML Algorithms

| Algoritma | Avantaj | Referans |
|-----------|---------|----------|
| **Random Forest** | Robust, feature importance | [12][13] |
| **SVM** | İyi generalization | [3][5] |
| **XGBoost** | SOTA performans | [19] |
| **k-NN** | Baseline, basit | [2][4] |

---

## Referanslar

[1] Killourhy, K. S., & Maxion, R. A. (2009). Comparing anomaly-detection algorithms for keystroke dynamics. *IEEE/IFIP DSN*.

[2] Teh, P. S., Zhang, N., Teoh, A. B. J., & Chen, K. (2016). A survey on touch dynamics authentication in mobile devices. *Computers & Security*, 59, 210-235.

[3] Frank, M., Biedert, R., Ma, E., Martinovic, I., & Song, D. (2013). Touchalytics: On the applicability of touchscreen input as a behavioral biometric. *IEEE TIFS*.

[4] Monaco, J. V., et al. (2017). Developing a keystroke biometric system for continual authentication. *IEEE EISIC*.

[5] Antal, M., Szabó, L. Z., & László, I. (2015). Keystroke dynamics on android platform. *Procedia Technology*, 19, 820-826.

[6] De Luca, A., et al. (2012). Touch me once and I know it's you! *ACM CHI*.

[8] Feng, T., et al. (2014). Continuous mobile authentication using touchscreen gestures. *IEEE HST*.

[12] Sitová, Z., et al. (2016). HMOG: New behavioral biometric features for continuous authentication. *IEEE TIFS*, 11(5), 877-892.

[13] Shen, C., Yu, T., Yuan, S., Li, Y., & Guan, X. (2016). Performance analysis of motion-sensor behavior for user authentication. *Sensors*, 16(3), 345.

[14] Muaaz, M., & Mayrhofer, R. (2017). Smartphone-based gait recognition. *IEEE TMC*.

[16] Shen, C., Chen, Y., & Guan, X. (2018). Performance evaluation of implicit smartphones authentication. *Information Sciences*, 430, 538-553.

[19] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM KDD*.

---

**Son Güncelleme:** 2026-01-19
