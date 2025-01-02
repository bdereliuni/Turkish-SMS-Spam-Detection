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
