import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from huggingface_hub import HfApi, model_info
from threading import Thread

# Global variables
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = None
model = None
model_list = []

# Constraints for matching
MIN_WORD_LEN = 2
JACCARD_THRESHOLD = 0.05

def get_initial_data():
    """Fetch initial model list from HF."""
    global model_list
    api = HfApi()
    models = api.list_models(limit=500)
    model_list = [model.modelId for model in models]
    return len(model_list)

def load_local_model():
    """Initialize the Qwen model and tokenizer."""
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

def find_best_model(prompt: str):
    """Find the best matching model using Jaccard Similarity and minimum word length."""
    # Filter prompt words: keep only alphanumeric words with len >= MIN_WORD_LEN
    prompt_words = {
        word.lower() for word in prompt.replace("/", " ").replace("-", " ").replace("_", " ").split()
        if len(word) >= MIN_WORD_LEN and word.isalnum()
    }
    
    if not prompt_words:
        return None

    best_model = None
    max_similarity = 0.0

    for model_id in model_list:
        # Normalize model parts
        model_parts = {
            part.lower() for part in model_id.replace("/", " ").replace("-", " ").replace("_", " ").split()
            if len(part) >= MIN_WORD_LEN and part.isalnum()
        }
        
        if not model_parts:
            continue
            
        # Jaccard Similarity: intersection over union
        intersection = prompt_words.intersection(model_parts)
        union = prompt_words.union(model_parts)
        similarity = len(intersection) / len(union) if union else 0.0
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_model = model_id

    if max_similarity >= JACCARD_THRESHOLD:
        return best_model
    return None

def get_model_metadata(model_id: str):
    """Fetch specific metadata for a given model ID."""
    info = model_info(model_id, files_metadata=True)
    
    safetensors_size = sum(
        file.size for file in info.siblings 
        if file.rfilename.endswith(".safetensors") and file.size is not None
    )
    
    size_gb = safetensors_size / (1024**3) if safetensors_size > 0 else 0
    
    base_model = "Unknown"
    if info.card_data and hasattr(info.card_data, 'base_model'):
        base_model = info.card_data.base_model

    return {
        "full_name": info.modelId,
        "purpose": info.pipeline_tag if info.pipeline_tag else "Unknown",
        "library": info.library_name if info.library_name else "Unknown",
        "base_model": base_model,
        "safetensors_size_gb": round(size_gb, 2)
    }

def generate_answer_stream(user_prompt: str, metadata: dict = None):
    """Generate a response using augmented prompt with specific instructions."""
    
    if metadata:
        metadata_text = (
            f"- Full Name: {metadata['full_name']}\n"
            f"- Primary Purpose: {metadata['purpose']}\n"
            f"- Required Library: {metadata['library']}\n"
            f"- Base Model: {metadata['base_model']}\n"
            f"- Total Weight Size: {metadata['safetensors_size_gb']} GB"
        )
    else:
        metadata_text = "No data in the database."
    
    full_prompt = (
        f"Answer shortly to the prompt query and then proceed to more detailed, "
        f"but concise discussion of the following database items (couple of sentences each):\n\n"
        f"{metadata_text}\n\n"
        f"Prompt Query: {user_prompt}"
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant providing information about AI models."},
        {"role": "user", "content": full_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for new_text in streamer:
        yield new_text