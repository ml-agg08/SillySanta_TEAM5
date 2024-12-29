import datasets
import numpy as np
from transformers import (
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
import torch
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize


label_list = [
    "O",    
    "B-HOBBY",   
    "I-HOBBY",     
    "B-TOY",       
    "I-TOY",       
    "B-SPORT",     
    "I-SPORT",     
    "B-SUBJECT",   
    "I-SUBJECT"   
]


label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

def create_sample_dataset():
    """
    Create a larger synthetic dataset with annotated children's interests.
    """
    texts = [
        ["I", "love", "playing", "with", "lego", "blocks", "and", "reading", "science", "books"],
        ["My", "favorite", "sport", "is", "soccer", "and", "I", "enjoy", "painting", "pictures"],
        ["I", "want", "a", "playstation", "for", "christmas", "and", "love", "math", "class"],
        ["Playing", "minecraft", "is", "my", "favorite", "hobby", "besides", "basketball"],
        ["I", "enjoy", "collecting", "pokemon", "cards", "and", "playing", "chess"],
        ["Reading", "harry", "potter", "books", "is", "my", "favorite", "activity"],
        ["I", "love", "doing", "science", "experiments", "and", "solving", "puzzles"],
        ["My", "hobbies", "are", "drawing", "and", "playing", "piano"],
        ["I", "want", "to", "learn", "coding", "and", "play", "football"],
        ["Building", "with", "legos", "and", "reading", "adventure", "books", "are", "fun"],
        ["I", "really", "love", "playing", "basketball", "after", "school"],
        ["Chemistry", "is", "my", "favorite", "subject", "in", "school"],
        ["I", "spend", "hours", "playing", "with", "my", "nintendo", "switch"],
        ["Reading", "comic", "books", "and", "drawing", "manga", "are", "my", "passions"]
    ]
    
    labels = [
        [0, 0, 0, 0, 3, 4, 0, 1, 7, 8],
        [0, 0, 5, 0, 5, 0, 0, 0, 1, 2],
        [0, 0, 0, 3, 0, 0, 0, 0, 7, 8],
        [0, 3, 0, 0, 0, 0, 0, 5],
        [0, 0, 1, 3, 4, 0, 0, 1],
        [1, 3, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 7, 8, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 5],
        [0, 0, 3, 0, 1, 2, 2, 0, 0],
        [0, 0, 0, 0, 5, 0, 0],
        [7, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 4],
        [1, 3, 4, 0, 1, 2, 0, 0, 0]
    ]
    
    return datasets.Dataset.from_dict({
        "tokens": texts,
        "ner_tags": labels
    })

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """
    Tokenize inputs and align labels with tokens, handling subwords.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = []
    
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



def predict_interests_for_text(text, model, tokenizer):
    """
    Predict interests in a given text using the trained model.
    """
    words = word_tokenize(text.lower())
    
  
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
  
    word_ids = encoded.word_ids(0) 
 
    inputs = {k: v.to(model.device) for k, v in encoded.items()}
  
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
 
    predicted_labels = [id2label[pred.item()] for pred in predictions[0]]
    
 
    interests = []
    current_interest = []
    current_type = None
    
    for idx, (word_id, label) in enumerate(zip(word_ids, predicted_labels)):
        if word_id is not None:  
            if label.startswith('B-'):
                if current_interest:
                    interests.append({
                        'text': ' '.join(current_interest),
                        'type': current_type
                    })
                current_interest = [words[word_id]]
                current_type = label[2:]
            elif label.startswith('I-') and current_interest:
                current_interest.append(words[word_id])
            elif label == 'O' and current_interest:
                interests.append({
                    'text': ' '.join(current_interest),
                    'type': current_type
                })
                current_interest = []
                current_type = None
    

    if current_interest:
        interests.append({
            'text': ' '.join(current_interest),
            'type': current_type
        })
    
    return interests



def train_interest_ner():
    try:
   
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        
    
        dataset = create_sample_dataset()
        

        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        

        tokenized_datasets = dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer),
            batched=True
        )
        
  
        model = AutoModelForTokenClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )
        

        training_args = TrainingArguments(
            output_dir="interest-ner-model",
            learning_rate=1e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=10,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=10
        )
        
     
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=DataCollatorForTokenClassification(tokenizer)
        )
   
        trainer.train()
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    try:
      
        model, tokenizer = train_interest_ner()
        
  
        sample_letter = "I love playing minecraft and reading harry potter books. I also enjoy soccer and math class."
        interests = predict_interests_for_text(sample_letter, model, tokenizer)
        
        print("\nDetected interests:")
        for interest in interests:
            print(f"- {interest['text']} (Type: {interest['type']})")
        
        
        model.save_pretrained("saved_model")
        tokenizer.save_pretrained("saved_tokenizer")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")