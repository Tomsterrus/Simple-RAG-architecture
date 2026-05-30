import customtkinter as ctk
import backend

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple RAG Architecture")
        self.geometry("500x400")

        # Storage for fetched model IDs
        self.models = []

        # UI elements
        self.btn_fetch = ctk.CTkButton(self, text="Fetch Models", command=self.fetch_models)
        self.btn_fetch.pack(pady=10)

        self.label_status = ctk.CTkLabel(self, text="Models not fetched")
        self.label_status.pack(pady=5)

        self.entry_prompt = ctk.CTkEntry(self, placeholder_text="Enter prompt to find a model", width=350)
        self.entry_prompt.pack(pady=10)

        self.btn_find = ctk.CTkButton(self, text="Find Best Match", command=self.find_model)
        self.btn_find.pack(pady=10)

        self.label_result = ctk.CTkLabel(self, text="Best match: None", wraplength=400)
        self.label_result.pack(pady=20)

    def fetch_models(self):
        """Fetch models and update the status label."""
        self.models = backend.get_model_list(limit=500)
        self.label_status.configure(text=f"Fetched {len(self.models)} models")

    def find_model(self):
        """Search for the best matching model from the list based on the prompt."""
        if not self.models:
            self.label_result.configure(text="Please fetch models first!")
            return

        prompt = self.entry_prompt.get()
        best_match = backend.find_best_model(prompt, self.models)
        
        if best_match:
            self.label_result.configure(text=f"Best match: {best_match}")
        else:
            self.label_result.configure(text="No matching model found")

if __name__ == "__main__":
    app = App()
    app.mainloop()