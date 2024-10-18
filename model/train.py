import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import ParameterGrid

# load model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)

# make labels of entities
label_map = {
    'B-PRODUCT': 0,  # start entity
    'I-PRODUCT': 1,  # inside entity
    'O': 2           # outside entity
}

def load_data(file_path):
    sentences, labels = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip() # delete spaces at the beginning and end of the line
            if line:
                parts = line.split()  # divide the line into parts
                words, tags = [], []
                for part in parts:
                    if part.startswith('B-') or part.startswith('I-'):  # check if the part is a label
                        if len(words) > 0:  # Если уже есть слово
                            tags.append('O')  # Добавляем метку O для слова перед меткой продукта
                        word = part[2:]  # Извлекаем само слово без префикса
                        words.append(word)
                        tags.append(part)  # Добавляем метку
                    else:
                        words.append(part)  # Добавляем слово
                        tags.append('O')  # Добавляем метку O

                # Сохраняем предложение и его метки
                sentences.append(words)
                labels.append(tags)
    return sentences, labels

train_sentences, train_labels = load_data('data/furniture_products_dataset_additional.txt')
eval_sentences, eval_labels = load_data('data/furniture_products_eval.txt')

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
                label_ids[j] = label_map[label[word_ids[j]]]  # Указываем метку для токена
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Применение токенизации
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# work with Grid Search
param_grid = {
    'learning_rate': [1e-5, 2e-5],
    'per_device_train_batch_size': [14],
    'num_train_epochs': [4, 6],
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
        save_strategy="no",  # чтобы не сохранять промежуточные модели
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
    
    save_directory = 'model/save_new_model'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if eval_loss == best_eval_loss:
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print("Model saved")

print(f"Best params: {best_params}")
print(f"Best eval loss: {best_eval_loss}")

