import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from huggingface_hub import HfApi, model_info
from threading import Thread

# Global variables
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = None
model = None
model_list = []

def get_initial_data():
    """Fetch initial model list from HF."""
    global model_list
    api = HfApi()
    models = api.list_models(limit=10000)
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
    """Find the model that matches the most prompt words from pre-fetched list."""
    prompt_words = set(prompt.lower().split())
    best_model = None
    max_matches = 0

    for model_id in model_list:
        model_name_parts = set(model_id.lower().replace("/", " ").replace("-", " ").split())
        matches = len(prompt_words.intersection(model_name_parts))
        
        if matches > max_matches:
            max_matches = matches
            best_model = model_id
            
    return best_model

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

def generate_answer_stream(user_prompt: str, metadata: dict):
    """Generate a response using augmented prompt with specific instructions."""
    
    metadata_text = (
        f"- Full Name: {metadata['full_name']}\n"
        f"- Primary Purpose: {metadata['purpose']}\n"
        f"- Required Library: {metadata['library']}\n"
        f"- Base Model: {metadata['base_model']}\n"
        f"- Total Weight Size: {metadata['safetensors_size_gb']} GB"
    )
    
    full_prompt = (
        f"Answer shortly to the prompt query and then proceed to more detailed, "
        f"but concise discussion of the following database items (explain each of those items in two-three sentences):\n\n"
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