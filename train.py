import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import evaluate
import logging
from datetime import datetime
import os
import wandb  # Eğitim takibi için
from sklearn.metrics import classification_report
import shutil
from sklearn.utils.class_weight import compute_class_weight

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Weights & Biases başlatma
wandb.init(project="turkish-sms-spam-detection")

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Kullanılan cihaz: {device}")

def load_and_preprocess_data(file_path):
    """Veri setini yükle ve ön işleme yap"""
    logging.info("Veri seti yükleniyor...")
    df = pd.read_csv(file_path, sep=';')
    
    # Veri kalitesi kontrolleri
    df = df.dropna()  # Eksik değerleri temizle
    df = df.drop_duplicates()  # Tekrarlanan verileri temizle
    
    # Metin temizleme - sadece basic temizlik
    df['Message'] = df['Message'].str.strip()
    df['Message'] = df['Message'].str.lower()
    # URL temizleme kaldırıldı
    # Özel karakter temizleme kaldırıldı - çünkü "!!!" gibi işaretler de spam göstergesi olabilir
    
    # Label encoding
    le = LabelEncoder()
    df['labels'] = le.fit_transform(df['Group'])
    
    # Etiketleri kontrol et
    logging.info("Benzersiz etiketler:")
    logging.info(df['Group'].unique())
    
    logging.info(f"Toplam veri sayısı: {len(df)}")
    logging.info(f"Sınıf dağılımı:\n{df['Group'].value_counts()}")
    
    return df, le

def create_datasets(df, tokenizer, test_size=0.2, val_size=0.1):
    """Eğitim, doğrulama ve test setlerini oluştur"""
    # Önce test setini ayır
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        df['Message'].values,
        df['labels'].values,
        test_size=test_size,
        stratify=df['labels'].values,
        random_state=42
    )
    
    # Kalan veriyi eğitim ve doğrulama olarak ayır
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=42
    )
    
    # Dataset'leri oluştur
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'labels': train_labels.astype(int)
    })
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'labels': val_labels.astype(int)
    })
    test_dataset = Dataset.from_dict({
        'text': test_texts,
        'labels': test_labels.astype(int)
    })
    
    return train_dataset, val_dataset, test_dataset

def tokenize_function(examples, tokenizer):
    """Metinleri tokenize et"""
    tokenized = tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    # Etiketleri de ekleyelim
    tokenized['labels'] = examples['labels']
    return tokenized

def compute_metrics(eval_pred):
    """Model metriklerini hesapla"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Detaylı sınıflandırma raporu
    report = classification_report(
        labels,
        predictions,
        output_dict=True
    )
    
    # Her sınıf için ayrı metrikler
    metrics = {
        "accuracy": report['accuracy'],
        "macro_f1": report['macro avg']['f1-score'],
        "weighted_f1": report['weighted avg']['f1-score'],
        "spam_precision": report['1']['precision'],  # Spam sınıfı için precision
        "spam_recall": report['1']['recall'],        # Spam sınıfı için recall
        "spam_f1": report['1']['f1-score'],         # Spam sınıfı için f1
        "normal_precision": report['0']['precision'], # Normal sınıfı için precision
        "normal_recall": report['0']['recall'],      # Normal sınıfı için recall
        "normal_f1": report['0']['f1-score']         # Normal sınıfı için f1
    }
    
    return metrics

def compute_class_weights(df):
    """Sınıf ağırlıklarını hesapla"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['labels']),
        y=df['labels']
    )
    return dict(enumerate(class_weights))

def main():
    # Model ve tokenizer yükleme
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Veri yükleme ve ön işleme
    df, le = load_and_preprocess_data('TurkishSMSCollection.csv')
    
    # Sınıf ağırlıklarını hesapla
    class_weights = compute_class_weights(df)
    logging.info(f"Sınıf ağırlıkları: {class_weights}")
    
    # Dataset'leri oluştur
    train_dataset, val_dataset, test_dataset = create_datasets(df, tokenizer)
    
    # Tokenization
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    ).to(device)
    
    # Training arguments
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./spam_model_{current_time}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=f"turkish-sms-spam-detection-run-{current_time}",
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.1,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        logging_dir="./logs",
        logging_steps=50,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        warmup_steps=500,
        fp16=True,
        report_to="wandb",
        warmup_ratio=0.1
    )
    
    # Sınıf ağırlıklarını loss function'a ekle
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=torch.tensor(list(class_weights.values())).float(),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )]
    )
    
    # Eğitim
    logging.info("Model eğitimi başlıyor...")
    trainer.train()
    
    # Test seti üzerinde değerlendirme
    logging.info("Test seti değerlendirmesi yapılıyor...")
    test_results = trainer.evaluate(test_dataset)
    logging.info(f"Test sonuçları: {test_results}")
    
    # Model ve tokenizer'ı kaydet
    output_dir = f"./spam_model_final_{current_time}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Label encoder'ı kaydet
    import pickle
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    
    logging.info("Eğitim tamamlandı ve model kaydedildi.")
    wandb.finish()

if __name__ == "__main__":
    if os.path.exists("./spam_model"):
        shutil.rmtree("./spam_model")
    if os.path.exists("./spam_model_final"):
        shutil.rmtree("./spam_model_final")
    main() 