# Evaluation — конспект

## 1. Что такое Evaluation

**Evaluation** — это процесс оценки качества LLM или RAG-системы.
Цель: понять, насколько ответы модели правильные, полезные и соответствуют требованиям задачи.

---

## 2. Виды оценки

### 2.1 Автоматические метрики

#### 2.1.1 Perplexity

* Показывает, насколько модель «уверена» в тексте
* Формула:

```
PPL = exp(loss)
```

* Меньше perplexity → модель лучше предсказывает токены

#### 2.1.2 BLEU / ROUGE / METEOR

* Оценивают совпадение с эталонным текстом
* BLEU — n‑граммы
* ROUGE — совпадение по длине фраз/предложений
* Используются для summarization, translation, QA

#### 2.1.3 Cosine similarity (для RAG)

* Сравнивает embedding ответа с эталонным
* Показывает семантическую близость

---

### 2.2 Human Evaluation

* Проверка людьми по качеству:

  * factual correctness (фактическая точность)
  * fluency (читаемость)
  * helpfulness (полезность)
  * groundedness (опора на контекст, особенно в RAG)

#### Формат:

* Рейтинг 1–5
* Анкета с вопросами к модели
* Сравнение с baseline (другой моделью)

---

### 2.3 Task-specific Metrics

* Для QA → Exact Match, F1
* Для RAG → Recall@k, MRR, nDCG (из retrieval-metrics.md)
* Для summarization → ROUGE-L, coverage, conciseness

---

## 3. Evaluation в RAG

1. **Retrieval evaluation**

   * Проверяем, насколько релевантные документы найдены
   * Метрики: Recall@k, MRR, nDCG

2. **Generation evaluation**

   * Модель генерирует ответ на основе retrieved context
   * Метрики: factual accuracy, groundedness, human evaluation

3. **End-to-end**

   * Считаем success rate: ответ корректный и найден релевантный документ

---

## 4. Практические советы

* Всегда отделяй **train / validation / test**
* Для RAG — оцени отдельно retrieval и generation
* Используй mix автоматических и human метрик
* Не забывай latency / speed, особенно в production

---

## 5. Интуитивно

* Evaluation = зеркало модели
* Без оценки сложно понять:

  * правильно ли она отвечает
  * стоит ли менять embeddings, vector DB или fine-tune
* Хорошая практика = несколько метрик + ручная проверка

---

## 6. Краткое резюме

Evaluation в LLM / RAG включает:

* **Автоматические метрики** — скорость, точность, perplexity
* **Human evaluation** — полезность, стиль, фактическая корректность
* **Task-specific metrics** — QA, summarization, retrieval

Правильная оценка помогает:

* улучшать качество
* выбирать лучшие модели и embeddings
* строить надёжные RAG-системы