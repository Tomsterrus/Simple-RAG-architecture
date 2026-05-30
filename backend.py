from huggingface_hub import HfApi, model_info

def get_model_list(limit: int = 500):
    """Fetch a list of models from Hugging Face."""
    api = HfApi()
    models = api.list_models(limit=limit)
    return [model.modelId for model in models]

def find_best_model(prompt: str, model_list: list):
    """Find the model that matches the most prompt words."""
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
    
    # Calculate total size of safetensors files
    safetensors_size = sum(
        file.size for file in info.siblings 
        if file.rfilename.endswith(".safetensors") and file.size is not None
    )
    
    # Convert size to GB for readability
    size_gb = safetensors_size / (1024**3) if safetensors_size > 0 else 0
    
    # Extract base model from card data if available
    base_model = "Unknown"
    if info.card_data and hasattr(info.card_data, 'base_model'):
        base_model = info.card_data.base_model

    metadata = {
        "full_name": info.modelId,
        "purpose": info.pipeline_tag if info.pipeline_tag else "Unknown",
        "library": info.library_name if info.library_name else "Unknown",
        "base_model": base_model,
        "safetensors_size_gb": round(size_gb, 2)
    }
    return metadata