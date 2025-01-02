
# Türkçe SMS Spam Tespiti

Bu proje, Türkçe SMS mesajlarında spam tespiti yapan bir derin öğrenme modelini içermektedir. BERT tabanlı model, yüksek doğrulukla spam/normal SMS sınıflandırması yapabilmektedir.

## 1. Dosya Yapısı

- **train.py**: Model eğitimi ve değerlendirme işlemlerini gerçekleştiren ana script
- **requirements.txt**: Gerekli Python paketlerinin listesi
- **README.md**: Proje dokümantasyonu

## 2. Eğitim Süreci

### 2.1 Veri Hazırlama
- `TurkishSMSCollection.csv` dosyasından veri yüklenir
- Duplikasyon ve eksik değer temizliği yapılır
- Mesajlar lowercase'e dönüştürülür
- `LabelEncoder` ile etiketler sayısallaştırılır

### 2.2 Veri Seti Bölümleme
- Eğitim seti: %80
- Test seti: %20
- Doğrulama seti: Eğitim setinin %10'u

### 2.3 Tokenizasyon
- Model: `dbmdz/bert-base-turkish-cased`
- Maksimum token uzunluğu: 128
- Padding ve truncation uygulanır

### 2.4 Model Eğitimi
- BERT tabanlı sınıflandırma modeli (`AutoModelForSequenceClassification`)
- Hugging Face `Trainer` API kullanımı
- `EarlyStoppingCallback` implementasyonu
- Sınıf ağırlıkları dengesi

### 2.5 Değerlendirme
- Metrikler: Accuracy, F1-score, Precision, Recall
- Test seti üzerinde final değerlendirme
- Model kaydetme işlemi

## 3. Kurulum ve Çalıştırma

1. Projeyi klonlayın:
```bash
git clone https://github.com/bdereliuni/Turkish-SMS-Spam-Detection.git
cd Turkish-SMS-Spam-Detection
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Eğitimi başlatın:
```bash
python train.py
```

4. Eğitim sonunda model ve tokenizer `spam_model_final_YYYYMMDD_HHMMSS` klasöründe kaydedilir.

## 4. Model ve Arayüzler

- **Hugging Face Model**: [bdereliuni/turkish-sms-spam-detection-v2](https://huggingface.co/bdereliuni/turkish-sms-spam-detection-v2)
- **Gradio Demo**: [Türkçe SMS Spam Detection (v2space)](https://huggingface.co/spaces/bdereliuni/turkish-sms-spam-v2space)
- **Daha Detaylı Arayüz + Web API**: [Türkçe SMS Spam Detection](https://huggingface.co/spaces/bdereli/turkish-sms-spam-detection)

## 5. Sonuçlar ve Gelecek Çalışmalar

Model, test setinde yüksek başarı oranları elde etmiştir. İyileştirme için:
- Daha geniş veri seti toplanabilir
- Farklı model mimarileri denenebilir
- Çapraz doğrulama teknikleri uygulanabilir

## 6. Katkıda Bulunma

- Issue açarak geri bildirimde bulunabilirsiniz
- Pull request göndererek katkıda bulunabilirsiniz

## 7. Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

**Not**: Detaylı bilgi ve güncellemeler için lütfen GitHub repository'sini takip edin.
