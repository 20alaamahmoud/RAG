# extract_pdf_elements.py
from unstructured.partition.pdf import partition_pdf
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json

# === Load local Gemma model ===
model_id = "google/gemma-7b-it"
cache_dir = "/mnt/models/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Extract elements ===
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

# === Summarize text or table ===
def summarize_text(text, max_tokens=300):
    prompt = (
        "You are an assistant tasked with summarizing tables and text.\n"
        "Give a concise summary of the table or text.\n\n"
        "Respond only with the summary, no additional comment.\n"
        "Do not start your message by saying 'Here is a summary' or anything like that.\n"
        "Just give the summary as it is.\n\n"
        "Table or text chunk: " + text
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.replace(prompt, "").strip()
    except Exception as e:
        print("Error summarizing:", e)
        return ""

# === Main logic ===
if __name__ == "__main__":
    folder_path = "./content"
    output_root = "./summarized"
    os.makedirs(output_root, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            base_name = filename.replace(".pdf", "")
            output_folder = os.path.join(output_root, base_name)
            os.makedirs(output_folder, exist_ok=True)

            print("\nProcessing:", filename)

            texts, tables = extract_pdf_elements(file_path)

            text_summaries, table_summaries = [], []

            for idx, t in enumerate(texts):
                print(f"Summarizing text chunk {idx+1}/{len(texts)}...")
                summary = summarize_text(str(t))
                text_summaries.append(summary)
                with open(os.path.join(output_folder, f"text_chunk_{idx+1}.txt"), "w", encoding="utf-8") as f:
                    f.write(summary)

            for idx, t in enumerate(tables):
                print(f"Summarizing table {idx+1}/{len(tables)}...")
                summary = summarize_text(t.metadata.text_as_html)
                table_summaries.append(summary)
                with open(os.path.join(output_folder, f"table_{idx+1}.txt"), "w", encoding="utf-8") as f:
                    f.write(summary)

            # Save final JSON
            final_summary = {
                "file": filename,
                "text_summaries": text_summaries,
                "table_summaries": table_summaries,
            }

            with open(os.path.join(output_folder, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(final_summary, f, ensure_ascii=False, indent=2)

            print("Extracted", len(texts), "text chunks")
            print("Extracted", len(tables), "tables")
            print("Saved summaries to folder:", output_folder)
