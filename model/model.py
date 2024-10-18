from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained("./save_night_model")
tokenizer = AutoTokenizer.from_pretrained("./save_night_model")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
results = ner_pipeline("I just bought a stylish sofa and a wooden table from the store.")
for it in results:
    print(it)