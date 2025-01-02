# Türkçe SMS Spam Tespiti

Bu proje, **Türkçe SMS mesajlarını** yapay zeka (BERT tabanlı bir model) kullanarak **spam** veya **normal** olarak sınıflandırır. Model, [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) üzerinden eğitilmiş olup, final aşamada [Hugging Face](https://huggingface.co/) platformunda barındırılmaktadır.

---

## 1. Proje Hakkında

- **Amaç**: Türkçe SMS veri kümesini kullanarak, istenmeyen mesajları (spam) tespit eden bir sınıflandırma modeli geliştirmek.  
- **Veri Seti**: [TurkishSMSCollection.csv](https://github.com/onrkrsy/TurkishSMS-Collection/blob/main/TurkishSMSCollection.csv) dosyasını temel aldı.  
- **Model**: Fine-tuning işleminden sonra [**bdereliuni/turkish-sms-spam-detection-v2**](https://huggingface.co/bdereliuni/turkish-sms-spam-detection-v2) adıyla Hugging Face üzerinde yayınlandı.

---

## 2. Kurulum

### 2.1 Gerekli Paketler

Proje temelinde aşağıdaki teknolojileri kullanır:

- `transformers`
- `torch`
- `datasets`
- `scikit-learn`
- `evaluate`
- `wandb` (isteğe bağlı, eğitim takibi için)

Tüm paketleri tek komutla kurmak için:

```bash
pip install -r requirements.txt

## 2.2 Dosya Yapısı

- **train.py**  
  Model eğitimi ve değerlendirme işlemlerini gerçekleştiren temel Python scripti.

- **requirements.txt**  
  Gerekli Python paketlerinin listesini içerir.

- **README.md**  
  Projenin tanıtımı, kurulum ve kullanım rehberi.

---

## 3. Eğitim Adımları

1. **Veri Yükleme ve Temizlik**  
   - `TurkishSMSCollection.csv` dosyası okunur, duplikasyon ve eksik değer temizliği yapılır.  
   - Mesajlar lowercase’e dönüştürülür, `LabelEncoder` ile spam/normal etiketleri sayısallaştırılır.

2. **Veri Seti Bölme**  
   - Eğitim, doğrulama ve test kümeleri ayrılır (%80 / %20).  
   - Doğrulama için ayrıca eğitim setinden bir %10’luk kısım kullanılır.

3. **Tokenizer**  
   - `dbmdz/bert-base-turkish-cased` tokenizer kullanarak metinleri 128 tokena kadar keser/pad eder.

4. **Model Eğitimi**  
   - BERT tabanlı sınıflandırma modeli (`AutoModelForSequenceClassification`)  
   - `Trainer` API ile eğitim, `EarlyStoppingCallback`, class weight vb. stratejiler.

5. **Değerlendirme**  
   - Test seti üzerinde accuracy, F1, precision, recall vb. metrikler hesaplanır.  
   - Model final değerlendirme sonrası kaydedilir.

---

## 4. Nasıl Çalıştırılır?

1. **Bu projeyi klonlayın veya indirin**:
   ```bash
   git clone https://github.com/bdereliuni/Turkish-SMS-Spam-Detection.git
   cd Turkish-SMS-Spam-Detection
