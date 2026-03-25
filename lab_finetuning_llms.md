# 🧪 LAB GUIDE: Fine-Tuning LLMs for Domain-Specific Projects

> **Level:** Intermediate → Advanced  
> **Duration:** ~4–6 hours per lab  
> **Prerequisites:** Python basics, familiarity with Hugging Face Transformers, GPU (or Google Colab)

---

## 📚 TABLE OF CONTENTS

1. [Introduction to Fine-Tuning](#1-introduction)
2. [Lab 1 — Medical Q&A Bot (Healthcare Domain)](#lab-1)
3. [Lab 2 — Legal Document Summarizer (Legal Domain)](#lab-2)
4. [Lab 3 — Financial Sentiment Analyzer (Finance Domain)](#lab-3)
5. [Lab 4 — Customer Support Chatbot (E-Commerce Domain)](#lab-4)
6. [Lab 5 — Code Generation Assistant (Software Domain)](#lab-5)
7. [Evaluation & Benchmarking](#evaluation)
8. [Deployment Guide](#deployment)

---

## 1. Introduction to Fine-Tuning <a name="1-introduction"></a>

### What is Fine-Tuning?
Fine-tuning is adapting a pre-trained LLM (like GPT-2, Mistral, LLaMA) on a **domain-specific dataset** so it learns specialized vocabulary, tone, and knowledge.

### Why Fine-Tune Instead of Prompting?
| | Prompting | Fine-Tuning |
|---|---|---|
| Cost | Low setup | Higher setup, lower inference cost |
| Performance | Good for general | Excellent for specialized domains |
| Privacy | Data sent to API | Runs locally |
| Customization | Limited | Full control |

### Techniques Covered in This Lab

| Technique | When to Use |
|---|---|
| **Full Fine-Tuning** | Small models, lots of data |
| **LoRA** | Limited GPU, large models |
| **QLoRA** | Very limited GPU (4-bit quantization) |
| **Instruction Tuning** | Chat/instruction-following models |

### Core Setup (Install Once)

```bash
pip install transformers datasets peft trl accelerate bitsandbytes \
            sentencepiece huggingface_hub wandb
```

---

## 🔬 LAB 1 — Medical Q&A Bot (Healthcare Domain) <a name="lab-1"></a>

### 🎯 Objective
Fine-tune a small LLM on medical Q&A pairs so it can answer clinical questions accurately.

### 📁 Dataset
- **Source:** [MedQuAD](https://huggingface.co/datasets/medquad) on Hugging Face
- **Format:** Question → Answer pairs on medical topics

### Step 1: Load the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
print(dataset[0])
# {'qtype': 'symptoms', 'Question': 'What are symptoms of...', 'Answer': '...'}
```

### Step 2: Format into Instruction Template

```python
def format_prompt(example):
    return {
        "text": f"""### Instruction:
You are a medical assistant. Answer the following question accurately.

### Question:
{example['Question']}

### Answer:
{example['Answer']}"""
    }

dataset = dataset.map(format_prompt)
```

### Step 3: Load Base Model with QLoRA

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # or "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for low GPU

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
```

### Step 4: Apply LoRA

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~2% of total — very efficient!
```

### Step 5: Train with TRL's SFTTrainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./medical-llm",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    report_to="wandb"   # optional — tracks training metrics
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
```

### Step 6: Save & Inference

```python
model.save_pretrained("./medical-llm-adapter")
tokenizer.save_pretrained("./medical-llm-adapter")

# Inference
def ask_medical(question):
    prompt = f"### Instruction:\nAnswer this medical question.\n\n### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(ask_medical("What are the symptoms of diabetes?"))
```

### ✅ Lab 1 Checklist
- [ ] Dataset loaded and formatted
- [ ] QLoRA config applied
- [ ] Model trained for 3 epochs
- [ ] Model saved as adapter
- [ ] Inference tested with 5 medical questions

---

## 🔬 LAB 2 — Legal Document Summarizer (Legal Domain) <a name="lab-2"></a>

### 🎯 Objective
Fine-tune a model to summarize complex legal documents into plain English.

### 📁 Dataset
- **Source:** [legal_summarization](https://huggingface.co/datasets/joelniklaus/legal_summarization)
- **Format:** Legal document → plain-language summary

### Step 1: Load and Prepare

```python
from datasets import load_dataset

dataset = load_dataset("joelniklaus/legal_summarization", split="train")

def format_legal(example):
    return {
        "text": f"""### Instruction:
Summarize the following legal document in plain English.

### Legal Document:
{example['document'][:1500]}

### Summary:
{example['summary']}"""
    }

dataset = dataset.map(format_legal)
```

### Step 2: Use a Summarization-Friendly Base Model

```python
# Use FLAN-T5 for seq2seq summarization (lighter weight)
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
```

### Step 3: Fine-Tune with Seq2Seq Trainer

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

def preprocess(example):
    inputs = tokenizer(example["document"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["summary"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./legal-summarizer",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    predict_with_generate=True,
    fp16=True,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)
trainer.train()
```

### Step 4: Evaluate with ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(predicted_summary, reference_summary)
print(scores)
```

### ✅ Lab 2 Checklist
- [ ] Dataset loaded with legal docs
- [ ] FLAN-T5 fine-tuned
- [ ] ROUGE score calculated
- [ ] 3 legal docs summarized and reviewed

---

## 🔬 LAB 3 — Financial Sentiment Analyzer (Finance Domain) <a name="lab-3"></a>

### 🎯 Objective
Fine-tune a model to classify financial news/tweets as: **Positive / Neutral / Negative**

### 📁 Dataset
- **Source:** [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank)

### Step 1: Load Dataset

```python
dataset = load_dataset("financial_phrasebank", "sentences_75agree", split="train")
# Labels: 0=negative, 1=neutral, 2=positive
```

### Step 2: Fine-Tune FinBERT (or BERT)

```python
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)

def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")
```

### Step 3: Train

```python
training_args = TrainingArguments(
    output_dir="./financial-sentiment",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.train()
```

### Step 4: Streamlit App

```python
# app.py
import streamlit as st
from transformers import pipeline

pipe = pipeline("text-classification", model="./financial-sentiment")

st.title("📈 Financial Sentiment Analyzer")
text = st.text_area("Enter financial news headline:")
if st.button("Analyze"):
    result = pipe(text)[0]
    st.write(f"**Sentiment:** {result['label']} ({result['score']:.2f})")
```

### ✅ Lab 3 Checklist
- [ ] FinBERT loaded and fine-tuned
- [ ] Accuracy > 85% on validation set
- [ ] Streamlit app built and tested

---

## 🔬 LAB 4 — Customer Support Chatbot (E-Commerce Domain) <a name="lab-4"></a>

### 🎯 Objective
Fine-tune a chat model to handle customer queries for an e-commerce store.

### 📁 Dataset
- **Source:** [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

### Step 1: Format as Chat

```python
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

def format_chat(example):
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST] {example['response']} </s>"
    }

dataset = dataset.map(format_chat)
```

### Step 2: Fine-Tune Mistral / TinyLlama

```python
# Use TinyLlama for low GPU requirements
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Then follow the same QLoRA + SFTTrainer pattern from Lab 1
# Replace dataset and model name accordingly
```

### Step 3: Deploy as FastAPI Service

```python
# api.py
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
pipe = pipeline("text-generation", model="./support-chatbot-adapter", max_new_tokens=200)

@app.post("/chat")
def chat(query: str):
    prompt = f"<s>[INST] {query} [/INST]"
    return {"response": pipe(prompt)[0]["generated_text"]}

# Run: uvicorn api:app --reload
```

### ✅ Lab 4 Checklist
- [ ] Dataset formatted in chat template
- [ ] TinyLlama fine-tuned on support data
- [ ] FastAPI endpoint working
- [ ] 10 test queries answered correctly

---

## 🔬 LAB 5 — Code Generation Assistant (Software Domain) <a name="lab-5"></a>

### 🎯 Objective
Fine-tune a model to generate Python code from natural language descriptions.

### 📁 Dataset
- **Source:** [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)

### Step 1: Load and Format

```python
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

def format_code(example):
    return {
        "text": f"""### Instruction:
Write Python code to solve the following task.

### Task:
{example['instruction']}

### Python Code:
{example['output']}"""
    }

dataset = dataset.map(format_code)
```

### Step 2: Fine-Tune CodeLlama or StarCoder

```python
MODEL_NAME = "codellama/CodeLlama-7b-hf"
# Or lighter: "bigcode/starcoderbase-1b"

# Then apply same QLoRA pattern from Lab 1
```

### Step 3: VS Code Extension (Optional Advanced Step)
Create a simple extension that calls your FastAPI endpoint to suggest code completions in VS Code.

### ✅ Lab 5 Checklist
- [ ] Code dataset loaded and formatted
- [ ] CodeLlama/StarCoder fine-tuned
- [ ] Generate 5 Python functions from descriptions
- [ ] Compare output quality vs base model

---

## 📊 Evaluation & Benchmarking <a name="evaluation"></a>

### Metrics by Task Type

| Task | Metric | Library |
|---|---|---|
| Text Classification | Accuracy, F1 | `sklearn` |
| Summarization | ROUGE-1, ROUGE-L | `rouge_score` |
| Generation (Open-ended) | Perplexity, BERTScore | `bert_score` |
| Code Generation | Pass@k, Syntax Check | `code_eval` |

### Universal Evaluation Code

```python
from bert_score import score as bert_score

predictions = ["model output here"]
references  = ["ground truth here"]

P, R, F1 = bert_score(predictions, references, lang="en")
print(f"BERTScore F1: {F1.mean():.4f}")
```

---

## 🚀 Deployment Guide <a name="deployment"></a>

### Option A: Hugging Face Hub (Easiest)

```bash
huggingface-cli login
python -c "model.push_to_hub('your-username/medical-llm-v1')"
```

### Option B: FastAPI + Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t domain-llm .
docker run -p 8000:8000 domain-llm
```

### Option C: Gradio Demo (Quickest Share)

```python
import gradio as gr

def respond(question):
    return ask_medical(question)   # your inference function

gr.Interface(fn=respond, inputs="text", outputs="text",
             title="Medical AI Assistant").launch(share=True)
```

---

## 📌 QUICK REFERENCE — LoRA Hyperparameter Guide

| Parameter | Recommended Value | Effect |
|---|---|---|
| `r` (rank) | 8–64 | Higher = more params = better quality |
| `lora_alpha` | 2× r | Scaling factor |
| `lora_dropout` | 0.05–0.1 | Regularization |
| `learning_rate` | 1e-4 to 3e-4 | Lower = more stable |
| `epochs` | 2–5 | Avoid overfitting |
| `batch_size` | 4–16 | Larger = faster + more stable |

---

## 🎓 LEARNING RESOURCES

| Resource | Link |
|---|---|
| Hugging Face PEFT Docs | https://huggingface.co/docs/peft |
| TRL (SFT Trainer) Docs | https://huggingface.co/docs/trl |
| QLoRA Paper | https://arxiv.org/abs/2305.14314 |
| LoRA Paper | https://arxiv.org/abs/2106.09685 |
| DeepLearning.AI Fine-Tuning Course | https://www.deeplearning.ai/short-courses/finetuning-large-language-models/ |

---

> 💡 **Pro Tip:** Always compare your fine-tuned model vs the base model on the same 20 test examples before shipping. If fine-tuned is not clearly better, train for more epochs or increase LoRA rank.
