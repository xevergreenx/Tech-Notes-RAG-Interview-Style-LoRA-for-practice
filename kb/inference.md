# Inference — конспект

## 1. Что такое Inference

**Inference** — это процесс использования обученной LLM или RAG для генерации ответов на новые запросы.
Цель: получить результат модели на практике.

---

## 2. Inference в LLM

### 2.1 Общий процесс

1. Входной текст (prompt / instruction)
2. Токенизация
3. Передача токенов в модель
4. Генерация следующего токена
5. Повтор до конца или max_tokens

### 2.2 Параметры генерации

* **max_tokens** — максимальная длина ответа
* **temperature** — случайность / креативность
* **top-p (nucleus sampling)** — разнообразие ответа
* **top-k** — ограничение по наиболее вероятным токенам
* **repetition_penalty** — штраф за повтор

### 2.3 Пример кода (PyTorch + Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Объясни статью 158 УК РФ простыми словами."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 3. Inference в RAG

### 3.1 Общий процесс

1. Входной вопрос → эмбеддинг
2. Поиск top-k релевантных документов в vector DB
3. Сбор retrieved context
4. Формирование prompt: вопрос + контекст
5. Генерация ответа LLM

### 3.2 Пример кода (RAG MVP)

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Документы
docs = ["Статья 158 УК РФ — кража это тайное хищение имущества",
        "Статья 105 УК РФ — убийство это умышленное причинение смерти"]

# Embeddings
embedder = SentenceTransformer("intfloat/multilingual-e5-base")
embeddings = embedder.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Вопрос
query = "Что будет за кражу?"
q_emb = embedder.encode([query])
D, I = index.search(np.array(q_emb), k=2)
context = "\n".join([docs[i] for i in I[0]])

# Формируем prompt для LLM
prompt = f"Контекст:\n{context}\nВопрос: {query}\nОтветь с опорой на контекст."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 4. Особенности Inference

* **Контекстное окно** LLM ограничено → длинные тексты нужно чанковать
* **RAG помогает обойти ограничение**: LLM получает только релевантный контекст
* **Latency** = время на поиск + генерацию

---

## 5. Практические советы

* В RAG: топ-k = 3–5 для быстрых и точных ответов
* Температуру лучше ставить 0–0.7 для factual tasks
* Держи chunk size ≤ max_tokens модели
* Можно кэшировать embeddings для ускорения поиска

---

## 6. Интуитивно

* LLM alone = «угадывает следующий токен»
* RAG = «сначала ищет полезные куски, потом пишет ответ»
* Inference = момент истины, когда модель реально отвечает

---

## 7. Краткое резюме

Inference — это применение модели к реальным запросам:

* LLM: генерация текста на основе prompt
* RAG: генерация с опорой на external knowledge
* Параметры генерации и правильная подготовка контекста = ключ к качественным ответам
