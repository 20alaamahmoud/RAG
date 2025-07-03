# rag_pipeline.py
import os
import json
import uuid
from unstructured.partition.pdf import partition_pdf
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

# === Settings ===
PDF_DIR = "./content"
SUMMARY_DIR = "./summarized"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "esg_rag"

os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# === Load local Gemma model ===
model_id = "google/gemma-7b-it"
cache_dir = "/mnt/models/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype="auto",
    device_map="auto"
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === PDF extraction ===
def extract_pdf_elements(file_path):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    texts, tables = [], []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        elif "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    return texts, tables

# === Summarizer ===
def summarize_text(text, max_tokens=300):
    prompt = (
        "You are an assistant tasked with summarizing tables and text.\n"
        "Give a concise summary of the table or text.\n\n"
        "Respond only with the summary, no additional comment.\n"
        "Do not start your message by saying 'Here is a summary'.\n\n"
        "Table or text chunk: " + text
    )
    try:
        output = generator(prompt, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"]
        summary = output.split("Table or text chunk:")[-1].strip()
        return summary
    except Exception as e:
        print("Error summarizing:", e)
        return ""

# === Initialize Chroma Vectorstore ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_model, persist_directory=CHROMA_DB_DIR)
store = InMemoryStore()
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")

# === Main ===
if __name__ == "__main__":
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            summary_path = os.path.join(SUMMARY_DIR, filename.replace(".pdf", ".jsonl"))
            if os.path.exists(summary_path):
                print(f"Skipping {filename}, already summarized.")
                continue

            file_path = os.path.join(PDF_DIR, filename)
            print(f"\nProcessing: {filename}")

            texts, tables = extract_pdf_elements(file_path)
            text_summaries, table_summaries = [], []

            for i, t in enumerate(texts):
                print(f"Summarizing text chunk {i+1}/{len(texts)}...")
                text_summaries.append(summarize_text(str(t)))

            for i, t in enumerate(tables):
                print(f"Summarizing table {i+1}/{len(tables)}...")
                table_summaries.append(summarize_text(t.metadata.text_as_html))

            # Save summaries with original data
            data = {
                "file": filename,
                "text_chunks": [str(t) for t in texts],
                "text_summaries": text_summaries,
                "table_chunks": [t.metadata.text_as_html for t in tables],
                "table_summaries": table_summaries
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

            # Add to vectorstore
            doc_ids_text = [str(uuid.uuid4()) for _ in text_summaries]
            docs_text = [Document(page_content=s, metadata={"doc_id": doc_ids_text[i]}) for i, s in enumerate(text_summaries)]
            retriever.vectorstore.add_documents(docs_text)
            retriever.docstore.mset(list(zip(doc_ids_text, [Document(page_content=c) for c in data["text_chunks"]])))

            doc_ids_tables = [str(uuid.uuid4()) for _ in table_summaries]
            docs_tables = [Document(page_content=s, metadata={"doc_id": doc_ids_tables[i]}) for i, s in enumerate(table_summaries)]
            retriever.vectorstore.add_documents(docs_tables)
            retriever.docstore.mset(list(zip(doc_ids_tables, [Document(page_content=c) for c in data["table_chunks"]])))

            print(f"Extracted {len(texts)} texts | {len(tables)} tables")
            print(f"Summaries saved: {summary_path}")

    # Persist DB
    print("Saving Chroma DB...")
    vectorstore.persist()
    print("All done.")
