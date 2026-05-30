import customtkinter as ctk
import backend
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple RAG Architecture")
        self.geometry("800x750")

        # Loading UI
        self.label_loading = ctk.CTkLabel(self, text="Initializing System (HF API & Local LLM)...")
        self.label_loading.pack(pady=(50, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        # Main UI (hidden initially)
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        
        self.label_prompt = ctk.CTkLabel(self.main_frame, text="Enter your query:")
        self.label_prompt.pack(pady=(10, 0), anchor="w", padx=50)
        
        self.textbox_prompt = ctk.CTkTextbox(self.main_frame, width=700, height=100)
        self.textbox_prompt.pack(pady=10)

        self.btn_generate = ctk.CTkButton(self.main_frame, text="Generate RAG Answer", command=self.start_generation_thread)
        self.btn_generate.pack(pady=10)

        self.label_output = ctk.CTkLabel(self.main_frame, text="Response:")
        self.label_output.pack(pady=(10, 0), anchor="w", padx=50)
        
        self.textbox_output = ctk.CTkTextbox(self.main_frame, width=700, height=400)
        self.textbox_output.pack(pady=10)

        # Start loading background task
        threading.Thread(target=self.initialize_system, daemon=True).start()

    def initialize_system(self):
        """Pre-load model list and the local LLM."""
        backend.get_initial_data()
        self.progress_bar.set(0.4)
        backend.load_local_model()
        self.progress_bar.set(1.0)
        
        # Switch to main UI
        self.label_loading.pack_forget()
        self.progress_bar.pack_forget()
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

    def start_generation_thread(self):
        """Execute RAG process in a separate thread."""
        threading.Thread(target=self.process_rag, daemon=True).start()

    def process_rag(self):
        prompt = self.textbox_prompt.get("1.0", "end-1c").strip()
        if not prompt:
            return

        self.btn_generate.configure(state="disabled")
        self.update_display(self.textbox_output, "Searching database...\n")

        best_match = backend.find_best_model(prompt)
        
        if not best_match:
            header = (
                f"--- NO MATCH FOUND ---\n"
                f"Proceeding without database context...\n"
                f"-----------------------\n\n"
            )
            self.update_display(self.textbox_output, header)
            metadata = None
        else:
            metadata = backend.get_model_metadata(best_match)
            header = (
                f"--- SOURCE METADATA ---\n"
                f"ID: {metadata['full_name']}\n"
                f"-----------------------\n\n"
            )
            self.update_display(self.textbox_output, header)

        # Streaming RAG response
        for chunk in backend.generate_answer_stream(prompt, metadata):
            self.append_display(self.textbox_output, chunk)
        
        self.btn_generate.configure(state="normal")

    def update_display(self, widget, text):
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.see("end")

    def append_display(self, widget, text):
        widget.insert("end", text)
        widget.see("end")

if __name__ == "__main__":
    app = App()
    app.mainloop()