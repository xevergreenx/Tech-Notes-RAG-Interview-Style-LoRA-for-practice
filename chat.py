import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EMBED_MODEL = "intfloat/multilingual-e5-base"
INDEX_DIR = ".faiss"


def load_llm_pipeline_4bit():
    assert torch.cuda.is_available(), "for Colab"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.05,
        return_full_text=False,  # важно: возвращает только сгенерированное, без prompt
    )
    return gen


def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    return f"""Ты помощник по теме LLM/RAG.
Отвечай по-русски. Используй КОНТЕКСТ, если он релевантен.
Если в КОНТЕКСТЕ нет ответа, дай общий ответ и явно скажи: "В базе заметок нет прямого ответа".

КОНТЕКСТ:
{context_block}

ВОПРОС:
{question}

ОТВЕТ (формат):
1) Коротко (1-2 предложения)
2) Детали (3-6 bullet points)
3) Практика/подводные камни (2-4 bullets)
"""


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    db = load_db()
    gen = load_llm_pipeline_4bit()

    print("эйац-chat ready. Type your question.")
    print("Commands: /bye, /exit, /quit. Stop anytime with Ctrl+C.\n")

    try:
        while True:
            q = input("> ").strip()

            if not q:
                continue
            if q.lower() in {"/bye", "/exit", "/quit"}:
                print("Bye.")
                break
            
            retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.6})
            docs = retriever.invoke(q)
            contexts = [d.page_content[:900] for d in docs]
            prompt = build_prompt(q, contexts)

            generated = gen(prompt)[0]["generated_text"].strip()
            print("\n" + generated + "\n")

            print("SOURCES:")
            for i, d in enumerate(docs, 1):
                print(f"- [{i}] {d.metadata.get('source')}")
            print()

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")


if __name__ == "__main__":
    main()