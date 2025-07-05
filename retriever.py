# retriever.py
import uuid, os, json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever


CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "esg_rag"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=CHROMA_DB_DIR
)
store = InMemoryStore()
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")

# Load original documents into the retriever
def load_original_documents(summary_dir="./summarized"):
    for file in os.listdir(summary_dir):
        if file.endswith(".jsonl"):
            with open(os.path.join(summary_dir, file), "r", encoding="utf-8") as f:
                entry = json.loads(f.readline())

                
                summaries = entry["text_summaries"]
                chunks = entry["text_chunks"]

                filtered = [(s, c) for s, c in zip(summaries, chunks) if s.strip()]
                if not filtered:
                    print(f"Skipping {file}, all summaries are empty.")
                    continue

                doc_ids = [str(uuid.uuid4()) for _ in filtered]
                docs = [Document(page_content=s, metadata={"doc_id": doc_ids[i]}) for i, (s, _) in enumerate(filtered)]
                original_docs = [Document(page_content=c) for _, c in filtered]

                retriever.vectorstore.add_documents(docs)
                retriever.docstore.mset(list(zip(doc_ids, original_docs)))
                print(f"Loaded {len(filtered)} summaries from {file}")
load_original_documents()
