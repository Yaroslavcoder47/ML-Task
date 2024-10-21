import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import ParameterGrid

# load model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

# make labels of entities
# label_map = {
#     'B-PRODUCT': 0,  # start entity
#     'I-PRODUCT': 1,  # inside entity
#     'O': 2           # outside entity
# }

label_map = {
    'PRODUCT' : 0,
    'O': 1
}

def load_data(file_path):
    sentences, labels = [], []
    with open(file_path, 'r') as f:
        for index, line in enumerate(f):
            line = line.strip()
            if line:
                parts = line.split()
                words, tags = [], []

                for i in range(0, len(parts), 2):
                    try:
                        words.append(parts[i])
                        tags.append(parts[i+1])
                    except IndexError:
                        print(parts)
                        print(index)
            sentences.append(words)
            labels.append(tags)

    return sentences, labels

train_sentences, train_labels = load_data('data/data.txt')
eval_sentences, eval_labels = load_data('data/data_eval.txt')


train_data = {'tokens': train_sentences, 'ner_tags': train_labels}
eval_data = {'tokens': eval_sentences, 'ner_tags': eval_labels}

# Создайте Dataset из словарей
train_dataset = Dataset.from_dict(train_data)
eval_dataset = Dataset.from_dict(eval_data)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, padding=True, truncation=True)

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Получаем индексы слов
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])  # Заполняем метками по умолчанию
        for j in range(len(label_ids)):
            if word_ids[j] is not None:  # Если это токен из оригинального текста
                current_label = label[word_ids[j]]
                if current_label in label_map:  # Проверяем, есть ли метка в label_map
                    label_ids[j] = label_map[current_label]
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Применение токенизации
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# work with Grid Search
param_grid = {
    'learning_rate': [1e-5, 2e-5],
    'per_device_train_batch_size': [16, 18],
    'num_train_epochs': [5],
    'weight_decay': [0.01, 0.02]
}

# Генерация всех возможных комбинаций гиперпараметров
grid = list(ParameterGrid(param_grid))

# Перебираем все комбинации гиперпараметров
best_eval_loss = float('inf')
best_params = None

for params in grid:
    print(f"Training with params: {params}")
    
    # Определяем аргументы для тренировки с текущими гиперпараметрами
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=params['learning_rate'],
        per_device_train_batch_size=params['per_device_train_batch_size'],
        num_train_epochs=params['num_train_epochs'],
        weight_decay=params['weight_decay'],
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",  # чтобы не сохранять промежуточные модели
        load_best_model_at_end=True,  # загружаем лучшую модель в конце обучения
        save_total_limit=1, # ограничиваем количество сохраненных моделей
        report_to="none"  # отключаем WandB или другие логгеры, если они настроены
    )

    # Trainer для обучения модели
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    # Тренируем модель
    trainer.train()

    # Оцениваем модель на валидационном наборе
    eval_results = trainer.evaluate()
    eval_loss = eval_results['eval_loss']

    print(f"Eval loss for params {params}: {eval_loss}")

    # Сохраняем лучшую комбинацию гиперпараметров
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_params = params
    
    save_directory = 'model/saved_model_2'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if eval_loss == best_eval_loss:
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print("Model saved")

print(f"Best params: {best_params}")
print(f"Best eval loss: {best_eval_loss}")

