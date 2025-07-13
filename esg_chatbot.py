# esg_chatbot.py
# === Combined RAG Pipeline with Loader ===
import os, json, uuid, torch, logging, subprocess
from unstructured.partition.pdf import partition_pdf
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

# === Config ===
PDF_DIR = "./content"
SUMMARY_DIR = "./summarized"
TRAINING_DIR = "./training_data"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "esg_rag"

os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# === Load Gemma ===
model_id = "google/gemma-7b-it"
cache_dir = "/mnt/models/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Embeddings & Vectorstore ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_model, persist_directory=CHROMA_DB_DIR)
store = InMemoryStore()
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")

# === Summarization ===
def summarize_batch(text_chunks, batch_size=8):
    summaries = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        prompts = [
            "You are an assistant tasked with summarizing ESG-related PDF text.\n"
            "Respond concisely with only the summary.\n"
            "Text chunk: " + chunk for chunk in batch
        ]
        logging.info(f"Summarizing batch {i+1} to {i+len(batch)} of {len(text_chunks)}")
        try:
            outputs = generator(prompts, max_new_tokens=300, do_sample=False)
            summaries.extend([out[0]["generated_text"].split("Text chunk:")[-1].strip() for out in outputs])
        except Exception as e:
            logging.error(f"Error during batch summarization: {e}")
            summaries.extend([""] * len(batch))
    return summaries

# === Save to training data format ===
def save_training_format(chunks, summaries, output_path):
    training_data = []
    for i in range(len(chunks)):
        if summaries[i].strip():
            training_data.append({
                "instruction": chunks[i],
                "input": "",
                "output": summaries[i]
            })
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# === PDF Processing ===
def process_pdf(file_path, filename):
    logging.info(f"\nProcessing: {filename}")

    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=False,
        strategy="fast",
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
    logging.info(f"Extracted {len(texts)} text chunks")
    if not texts:
        logging.warning(f"Skipping {filename} (no extractable text)")
        return False

    text_chunks = [str(t) for t in texts]
    text_summaries = summarize_batch(text_chunks)

    summary_data = {
        "file": filename,
        "text_chunks": text_chunks,
        "text_summaries": text_summaries,
    }
    summary_path = os.path.join(SUMMARY_DIR, filename.replace(".pdf", ".jsonl"))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary_data, ensure_ascii=False) + "\n")
    logging.info(f"Saved summaries to: {summary_path}")

    training_path = os.path.join(TRAINING_DIR, filename.replace(".pdf", ".jsonl"))
    save_training_format(text_chunks, text_summaries, training_path)
    logging.info(f"Saved training data to: {training_path}")

    doc_ids = [str(uuid.uuid4()) for _ in text_summaries]
    docs = [Document(page_content=s, metadata={"doc_id": doc_ids[i], "source": filename}) for i, s in enumerate(text_summaries)]
    retriever.vectorstore.add_documents(docs)
    retriever.docstore.mset(list(zip(doc_ids, [Document(page_content=c, metadata={"source": filename}) for c in text_chunks])))
    logging.info(f"Indexed {len(texts)} text chunks in vectorstore")
    return True

# === Load Summarized Docs into Memory ===
def load_original_documents(summary_dir=SUMMARY_DIR):
    for file in os.listdir(summary_dir):
        if file.endswith(".jsonl"):
            file_path = os.path.join(summary_dir, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.loads(f.readline())
            except Exception as e:
                logging.error(f"Failed to load {file}: {e}")
                continue

            summaries = content.get("text_summaries", [])
            chunks = content.get("text_chunks", [])
            if not summaries or not chunks or all(not s.strip() for s in summaries):
                logging.warning(f"{file} has empty or missing summaries. Re-summarizing...")
                os.remove(file_path)
                subprocess.run(["python", __file__])
                continue

            filtered = [(s, c) for s, c in zip(summaries, chunks) if s.strip()]
            if not filtered:
                logging.warning(f"Skipping {file}: all summaries are empty.")
                continue

            doc_ids = [str(uuid.uuid4()) for _ in filtered]
            docs = [Document(page_content=s, metadata={"doc_id": doc_ids[i], "source": file}) for i, (s, _) in enumerate(filtered)]
            original_docs = [Document(page_content=c) for _, c in filtered]
            retriever.vectorstore.add_documents(docs)
            retriever.docstore.mset(list(zip(doc_ids, original_docs)))
            logging.info(f"Loaded {len(filtered)} summaries from {file}")

# === Main Execution ===
if __name__ == "__main__":
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            summary_path = os.path.join(SUMMARY_DIR, filename.replace(".pdf", ".jsonl"))
            should_process = True
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if not content.get("text_chunks"):
                        logging.info(f"Reprocessing {filename} (was empty)...")
                    else:
                        logging.info(f"Skipping {filename} (already summarized)")
                        should_process = False
            if should_process:
                file_path = os.path.join(PDF_DIR, filename)
                process_pdf(file_path, filename)

    load_original_documents()
    logging.info("\nPersisting vectorstore...")
    vectorstore.persist()
    logging.info("All done.")
