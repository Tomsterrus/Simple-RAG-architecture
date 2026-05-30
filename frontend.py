import customtkinter as ctk
import backend

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple RAG Architecture")
        self.geometry("600x600")

        self.models = []

        # UI elements
        self.btn_fetch = ctk.CTkButton(self, text="Fetch Models List", command=self.fetch_models)
        self.btn_fetch.pack(pady=10)

        self.label_status = ctk.CTkLabel(self, text="Models not fetched")
        self.label_status.pack(pady=5)

        self.entry_prompt = ctk.CTkEntry(self, placeholder_text="Enter prompt to find a model", width=400)
        self.entry_prompt.pack(pady=10)

        self.btn_analyze = ctk.CTkButton(self, text="Find and Analyze Model", command=self.analyze_model)
        self.btn_analyze.pack(pady=10)

        self.textbox_info = ctk.CTkTextbox(self, width=500, height=250)
        self.textbox_info.pack(pady=20, padx=20)

    def fetch_models(self):
        """Fetch models and update the status label."""
        self.models = backend.get_model_list(limit=500)
        self.label_status.configure(text=f"Fetched {len(self.models)} models")

    def analyze_model(self):
        """Find the best match and display its metadata."""
        if not self.models:
            self.update_info_display("Please fetch models first!")
            return

        prompt = self.entry_prompt.get()
        best_match = backend.find_best_model(prompt, self.models)
        
        if not best_match:
            self.update_info_display("No matching model found")
            return

        metadata = backend.get_model_metadata(best_match)
        
        info_text = (
            f"Full Name: {metadata['full_name']}\n"
            f"Purpose: {metadata['purpose']}\n"
            f"Library: {metadata['library']}\n"
            f"Base Model: {metadata['base_model']}\n"
            f"Total Safetensors Size: {metadata['safetensors_size_gb']} GB"
        )
        self.update_info_display(info_text)

    def update_info_display(self, text):
        """Helper to update the textbox content."""
        self.textbox_info.delete("1.0", "end")
        self.textbox_info.insert("1.0", text)

if __name__ == "__main__":
    app = App()
    app.mainloop()