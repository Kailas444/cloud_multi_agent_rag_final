
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from config import PDF_PATH, FAISS_DIR, EMBEDDING_MODEL, LLM_MODEL, MAX_NEW_TOKENS

def build_vector_store():
    os.makedirs(FAISS_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(os.path.join(FAISS_DIR,"index.faiss")):
        return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    reader = PdfReader(PDF_PATH)
    docs=[]
    for i,p in enumerate(reader.pages,1):
        t=(p.extract_text() or "").strip()
        if t:
            docs.append(Document(page_content=t,metadata={"source":os.path.basename(PDF_PATH),"page":i}))

    splitter = RecursiveCharacterTextSplitter(500,100)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks,embeddings)
    vs.save_local(FAISS_DIR)
    return vs

def build_llm():
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    gen = pipeline("text2text-generation",model=mdl,tokenizer=tok,max_new_tokens=MAX_NEW_TOKENS)
    return HuggingFacePipeline(pipeline=gen)

VECTOR_STORE = build_vector_store()
LLM = build_llm()

def retrieve_and_answer(query:str):
    docs = VECTOR_STORE.similarity_search(query,k=3)
    context="\n\n".join(f"(source={d.metadata['source']}, page={d.metadata['page']})\n{d.page_content}" for d in docs)
    prompt=f"Answer strictly from context:\n{context}\n\nQ:{query}\nA:"
    return LLM.invoke(prompt), [{"source":d.metadata["source"],"page":d.metadata["page"]} for d in docs]
