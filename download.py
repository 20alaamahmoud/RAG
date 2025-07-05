from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
 
model_id = "google/gemma-7b-it"
cache_dir = "CHOOSE YOUR PATH"
 
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained( model_id,
                                             cache_dir=cache_dir,
                                             torch_dtype="auto",    #check first you're using CPU or GPU
                                             device_map="auto"
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

 
