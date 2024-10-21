from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.preprocess_data import get_data_from_file

model = AutoModelForTokenClassification.from_pretrained("./model/saved_model_2")
tokenizer = AutoTokenizer.from_pretrained("./model/saved_model_2")



ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
results = ner_pipeline(get_data_from_file("web9.txt"))
for it in results:
    print(it)