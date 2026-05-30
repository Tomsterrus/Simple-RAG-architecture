# Simple RAG Architecture

A lightweight desktop GUI demonstrating a Retrieval-Augmented Generation (RAG) workflow. The system identifies AI models mentioned in user prompts, retrieves their technical metadata directly from the Hugging Face Hub, and injects this context into a local LLM to provide enriched, data-driven responses.

## Key Features

- Smart Metadata-Driven RAG: Automatically matches user queries against a live list of Hugging Face models to provide context-aware discussions based on real-time repository data.
- Dynamic Context Injection: Extracts and injects specific parameters into the prompt, including:
  - Primary Model Purpose (Pipeline Tag)
  - Required Integration Libraries
  - Base Model Lineage
  - Total Weight Size (Safetensors)
- Local Inference: Powered by the Qwen2.5-1.5B-Instruct model, allowing for private, local text generation without external LLM API costs.
- Asynchronous Initialization: Features a background loading system with a progress bar to initialize the 500-model database and the local LLM without freezing the UI.
- Real-time Streaming: Implements TextIteratorStreamer for token-by-token response rendering.
- Decoupled Architecture: Clean separation between the customtkinter frontend and the transformers/huggingface_hub backend.

## Tech Stack

- Frontend: customtkinter (Modern Desktop UI)
- Backend: PyTorch, Hugging Face Transformers, Hugging Face Hub API
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Concurrency: threading for non-blocking model loading and streaming inference.

## Prerequisites

- Python: 3.12.x
- CUDA Toolkit: Compatible version for NVIDIA GPU acceleration (optional, defaults to CPU if unavailable).
- Disk Space: Approximately 3GB for the local Qwen model weights.

## Installation

1. Clone the repository:
git clone https://github.com/YourUsername/simple-rag-architecture
cd simple-rag-architecture

2. Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

Note on GPU: For GPU support, ensure you install the PyTorch version corresponding to your CUDA version via pytorch.org.

## Usage

Run the application:
python frontend.py

1. Initialization: Wait for the progress bar to complete while the app fetches the model list and loads the LLM into memory.
2. Input: Enter a query involving a model name (e.g., "Tell me about bert-base-uncased").
3. Generation: Click "Generate RAG Answer" to see the retrieved metadata and the augmented response.

## Project Structure

- backend.py: Handles Hugging Face API communication, model matching logic, and LLM inference/streaming.
- frontend.py: Manages the customtkinter lifecycle, UI states, and threaded execution of backend tasks.
- requirements.txt: List of necessary libraries including huggingface_hub and transformers.

Please Note: prompt & model matching system is still in development, so try to use in your prompt model names close to those actually available at HF. 
