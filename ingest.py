import os 
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

KB_DIR = "kb"
INDEX_DIR = ".faiss"
EMBED_MODEL = "intfloat/multilingual-e5-base"

def main():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(
        KB_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded documents: {len(docs)}")
    if not docs:
        raise SystemExit("No documents found in kb/. Add .md files and retry.")


    # 2) Режем на чанки (чтобы retrieval работал лучше)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Chunks: {len(chunks)}")


    # 3) Embeddings 
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)

    print(f"Done. Saved FAISS index to {INDEX_DIR}/")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()