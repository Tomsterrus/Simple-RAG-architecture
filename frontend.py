import customtkinter as ctk
import backend

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple RAG Architecture")
        self.geometry("400x300")

        self.btn_fetch = ctk.CTkButton(self, text="Fetch Models", command=self.fetch_models)
        self.btn_fetch.pack(pady=20)

        self.label_result = ctk.CTkLabel(self, text="Models will appear here")
        self.label_result.pack(pady=20)

    def fetch_models(self):
        # Using "Qwen" as an example filter term
        models = backend.get_model_list("Qwen")
        self.label_result.configure(text="\n".join(models))

if __name__ == "__main__":
    app = App()
    app.mainloop()