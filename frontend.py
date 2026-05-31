import customtkinter as ctk
import backend
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple RAG Architecture")
        self.geometry("800x650")

        # Loading UI
        self.label_loading = ctk.CTkLabel(self, text="Initializing System (HF API & Local LLM)...")
        self.label_loading.pack(pady=(150, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        # Main UI (hidden initially)
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        
        # Grid layout for main UI (2 columns: left is interaction, right is database list)
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=2)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # LEFT COLUMN (Interaction Frame)
        self.left_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        self.label_prompt = ctk.CTkLabel(self.left_frame, text="Enter your query:")
        self.label_prompt.pack(pady=(10, 0), anchor="w", padx=20)
        
        self.textbox_prompt = ctk.CTkTextbox(self.left_frame, width=550, height=100)
        self.textbox_prompt.pack(pady=10)

        self.btn_generate = ctk.CTkButton(self.left_frame, text="Generate RAG Answer", command=self.start_generation_thread)
        self.btn_generate.pack(pady=10)

        self.label_output = ctk.CTkLabel(self.left_frame, text="Response:")
        self.label_output.pack(pady=(10, 0), anchor="w", padx=20)
        
        self.textbox_output = ctk.CTkTextbox(self.left_frame, width=550, height=400)
        self.textbox_output.pack(pady=10)

        # RIGHT COLUMN (Database Viewer Frame)
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        self.label_db = ctk.CTkLabel(self.right_frame, text="Hugging Face Model Database (Top 1000 Downloads):", font=("Arial", 12, "bold"))
        self.label_db.pack(pady=(15, 5))

        # Read-only scrollable textbox to easily view and copy model names
        self.textbox_db_list = ctk.CTkTextbox(self.right_frame, width=380, height=580)
        self.textbox_db_list.pack(pady=10, padx=15, expand=True, fill="both")

        # Start loading background task
        threading.Thread(target=self.initialize_system, daemon=True).start()

    def initialize_system(self):
        """Pre-load model list and the local LLM."""
        backend.get_initial_data()
        self.progress_bar.set(0.4)
        backend.load_local_model()
        self.progress_bar.set(1.0)
        
        # Populate the database viewer textbox
        models = backend.get_loaded_models()
        db_text = "\n".join(models)
        self.update_display(self.textbox_db_list, db_text)
        self.textbox_db_list.configure(state="disabled") # Set to read-only

        # Switch to main UI
        self.label_loading.pack_forget()
        self.progress_bar.pack_forget()
        self.main_frame.pack(expand=True, fill="both", padx=10, pady=10)

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
                f"Purpose: {metadata['purpose']}\n"
                f"Library: {metadata['library']}\n"
                f"Base Model: {metadata['base_model']}\n"
                f"Weight Size: {metadata['safetensors_size_gb']} GB\n"
                f"-----------------------\n\n"
            )
            self.update_display(self.textbox_output, header)

        # Streaming RAG response
        for chunk in backend.generate_answer_stream(prompt, metadata):
            self.append_display(self.textbox_output, chunk)
        
        self.btn_generate.configure(state="normal")

    def update_display(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.see("end")

    def append_display(self, widget, text):
        widget.insert("end", text)
        widget.see("end")

if __name__ == "__main__":
    app = App()
    app.mainloop()