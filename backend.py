import re
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
MATCH_THRESHOLD = 0.25  # At least 25% of filtered prompt words must match the model ID

# Set of common English conversational filler words to ignore (keeps size modifiers active)
STOPWORDS = {
    "what", "do", "you", "know", "about", "tell", "me", "show", "find", "get", 
    "search", "for", "please", "give", "information", "on", "is", "are", "the", 
    "a", "an", "and", "or", "of", "in", "to", "with", "how", "can", "could", 
    "would", "any", "some", "who", "which", "there", "here", "this", "that"
}

def get_initial_data():
    """Fetch initial model list from HF, sorted by downloads, then sorted alphabetically locally."""
    global model_list
    api = HfApi()
    # Fetch top 1000 models by downloads (without the deprecated direction parameter)
    models = api.list_models(sort="downloads", limit=1000)
    
    # Sort the retrieved list alphabetically for a better UI user experience
    model_list = sorted([model.modelId for model in models], key=lambda x: x.lower())
    return len(model_list)

def get_loaded_models():
    """Return the cached list of fetched models."""
    return model_list

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
    """Find the best matching model using substring overlap coefficient, ignoring conversational stopwords."""
    # Strip punctuation but keep alphanumeric characters, hyphens, underscores and slashes
    cleaned_prompt = re.sub(r'[^\w\s\-\/_]', ' ', prompt.lower())
    
    # Split by common separators used in model naming
    raw_words = cleaned_prompt.replace("/", " ").replace("-", " ").replace("_", " ").split()
    
    # Filter out conversational stopwords and words shorter than MIN_WORD_LEN
    prompt_words = {
        word for word in raw_words
        if len(word) >= MIN_WORD_LEN and word.isalnum() and word not in STOPWORDS
    }
    
    if not prompt_words:
        return None

    best_model = None
    max_similarity = 0.0

    for model_id in model_list:
        model_id_lower = model_id.lower()
        
        # Calculate how many of the prompt words exist as substrings inside the model ID
        matches = sum(1 for word in prompt_words if word in model_id_lower)
        similarity = matches / len(prompt_words) if prompt_words else 0.0
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_model = model_id
        elif similarity == max_similarity and similarity > 0:
            # Tie-breaker: prefer the more specific/shorter model ID if similarities are equal
            if best_model and len(model_id) < len(best_model):
                best_model = model_id

    if max_similarity >= MATCH_THRESHOLD:
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
    """Generate a response using augmented prompt with strict instructions to avoid hallucinations."""
    
    if metadata:
        context_knowledge = (
            f"Context: The model '{metadata['full_name']}' is available on Hugging Face.\n"
            f"Here are the official technical specifications retrieved from the Hugging Face database:\n"
            f"- Full Name: {metadata['full_name']}\n"
            f"- Primary Purpose: {metadata['purpose']}\n"
            f"- Required Library: {metadata['library']}\n"
            f"- Base Model: {metadata['base_model']}\n"
            f"- Total Weight Size: {metadata['safetensors_size_gb']} GB"
        )
        
        full_prompt = (
            f"{context_knowledge}\n\n"
            f"Instructions:\n"
            f"1. Answer the user prompt shortly based ONLY on the context provided above.\n"
            f"2. Explicitly state in your answer that this model is available on Hugging Face.\n"
            f"3. Briefly discuss the provided technical specs (name, purpose, library, base model, size) in a couple of concise sentences.\n"
            f"4. Strictly avoid hallucinating, guessing, or making up any biographical, academic, or other external facts about the author or the model that are not listed in the context above.\n\n"
            f"Prompt Query: {user_prompt}"
        )
    else:
        full_prompt = (
            f"Context: No data in the database.\n\n"
            f"Instructions:\n"
            f"1. Answer shortly to the prompt query.\n"
            f"2. State that no matching database entry was found.\n\n"
            f"Prompt Query: {user_prompt}"
        )
    
    messages = [
        {"role": "system", "content": "You are a precise assistant providing information about AI models using ONLY provided facts. You do not hallucinate details."},
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