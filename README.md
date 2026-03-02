# Tech Notes RAG — Interview Assistant

Простой RAG-ассистент для работы с техническими заметками.
Индексирует Markdown-файлы и отвечает на вопросы в стиле собеседования с помощью локальной LLM.

--- 

## Что умеет

- Индексация заметок в FAISS

- Поиск релевантного контекста через embeddings

- Генерация ответов локальной моделью

- Структурированные ответы (кратко → детали → практика)

---

## Стек
```
LangChain
FAISS
Sentence-Transformers (multilingual-e5)
Transformers + Torch
Qwen2.5-3B-Instruct
```

P.s. (Практика для понимания)
