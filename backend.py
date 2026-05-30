from huggingface_hub import HfApi

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