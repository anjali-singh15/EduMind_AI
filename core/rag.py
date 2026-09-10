from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"


def embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def splitter():
    return RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)


def pdf_retriever(pdf_path: str, k: int = 5):
    docs = PyPDFLoader(pdf_path).load()
    splits = splitter().split_documents(docs)
    store = FAISS.from_documents(splits, embeddings())
    return store.as_retriever(search_kwargs={"k": k})


def career_kb_retriever(k: int = 4):
    docs = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        docs.append(
            Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
        )
    splits = splitter().split_documents(docs)
    store = FAISS.from_documents(splits, embeddings())
    return store.as_retriever(search_kwargs={"k": k})
