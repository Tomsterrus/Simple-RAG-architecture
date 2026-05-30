from huggingface_hub import HfApi

def get_model_list(filter_term: str):
    """Fetch models from Hugging Face based on a filter term."""
    api = HfApi()
    models = api.list_models(search=filter_term, limit=100)
    return [model.modelId for model in models]