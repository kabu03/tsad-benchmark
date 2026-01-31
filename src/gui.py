import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
import os
import sys

sys.path.append(os.getcwd())

from src.benchmark import MODEL_REGISTRY, run_benchmark

class ModernBenchmarkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TSAD Benchmark Suite")
        self.root.geometry("700x720")
        
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            pass
            
        bg_color = "#121417"
        surface_color = "#1a1f24"
        fg_color = "#e6e6e6"
        muted_color = "#9aa4af"
        accent_color = "#4f8cff"
        self.bg_color = bg_color
        self.surface_color = surface_color
        self.fg_color = fg_color
        self.muted_color = muted_color
        self.accent_color = accent_color
        self.root.configure(bg=bg_color)
        
        base_font = ("Segoe UI", 10)
        title_font = font.Font(family="Segoe UI", size=18, weight="bold")
        header_font = font.Font(family="Segoe UI", size=11, weight="bold")
        
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color, font=base_font)
        self.style.configure("TButton", font=base_font, background=surface_color, foreground=fg_color)
        self.style.configure("TCheckbutton", background=bg_color, foreground=fg_color, font=base_font)
        self.style.map(
            "TCheckbutton",
            foreground=[("active", fg_color), ("disabled", muted_color)],
            background=[("active", bg_color), ("disabled", bg_color)]
        )
        self.style.configure("TLabelframe", background=bg_color, foreground=fg_color)
        self.style.configure("TLabelframe.Label", font=header_font, background=bg_color, foreground=fg_color)
        self.style.configure("Header.TLabel", font=title_font, foreground=fg_color)
        self.style.configure("Horizontal.TProgressbar", background=accent_color, troughcolor="#2a2d34")
        self.style.map(
            "TButton",
            background=[("active", "#2a78ff"), ("disabled", "#2a2d34")],
            foreground=[("disabled", muted_color)]
        )

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill="both", expand=True)

        lbl_title = ttk.Label(main_frame, text="Anomaly Detection Benchmark", style="Header.TLabel")
        lbl_title.pack(pady=(0, 20), anchor="center")

        config_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding=15)
        config_frame.pack(fill="x", pady=(0, 15))

        lbl_models = ttk.Label(config_frame, text="Select Algorithms to Evaluate:")
        lbl_models.pack(anchor="w", pady=(0, 10))

        self.model_vars = {}
        checks_frame = ttk.Frame(config_frame)
        checks_frame.pack(fill="both", expand=True)
        
        cols = 3
        for col in range(cols):
            checks_frame.columnconfigure(col, weight=1)
        for i, model_name in enumerate(MODEL_REGISTRY.keys()):
            var = tk.BooleanVar(value=True if "Isolation" in model_name else False)
            chk = ttk.Checkbutton(checks_frame, text=model_name, variable=var)
            chk.grid(row=i // cols, column=i % cols, sticky="w", padx=8, pady=6)
            self.model_vars[model_name] = var

        data_frame = ttk.LabelFrame(main_frame, text="Data Selection", padding=15)
        data_frame.pack(fill="x", pady=(0, 15))
        
        lbl_slider = ttk.Label(data_frame, text="Dataset Count (Smallest → Largest):")
        lbl_slider.pack(anchor="w")

        slider_frame = ttk.Frame(data_frame)
        slider_frame.pack(fill="x", pady=5)

        self.dataset_count_var = tk.IntVar(value=10)
        
        ttk.Label(slider_frame, text="1").pack(side="left")
        
        self.slider = ttk.Scale(slider_frame, from_=1, to=250, orient='horizontal', 
                                variable=self.dataset_count_var, command=self.update_slider_label)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        
        ttk.Label(slider_frame, text="250").pack(side="left")

        self.lbl_slider_val = ttk.Label(data_frame, text="10 Datasets", font=("Segoe UI", 10, "bold"), foreground=accent_color)
        self.lbl_slider_val.pack(anchor="center", pady=(5, 0))

        exec_frame = ttk.Frame(main_frame)
        exec_frame.pack(fill="x", pady=(10, 0))

        self.btn_run = ttk.Button(exec_frame, text="🚀 Run Benchmark Experiment", command=self.start_experiment, cursor="hand2")
        self.btn_run.pack(fill="x", ipady=5)
        
        self.progress = ttk.Progressbar(exec_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", pady=(15, 5))
        
        self.lbl_status = ttk.Label(exec_frame, text="Ready to start.", foreground=muted_color)
        self.lbl_status.pack()

        self.is_running = False

    def update_slider_label(self, val):
        count = int(float(val))
        self.lbl_slider_val.config(text=f"{count} Datasets")

    def start_experiment(self):
        if self.is_running:
            return
            
        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("Selection Required", "Please select at least one anomaly detection model.")
            return
            
        num_datasets = self.dataset_count_var.get()
        
        self.btn_run.config(state="disabled", text="Experiment Running...")
        self.is_running = True
        self.progress["value"] = 0
        self.lbl_status.config(text="Initializing models...", foreground=self.fg_color)
        
        self.total_steps = len(selected_models) * num_datasets
        self.current_step = 0
        
        models_to_run = [(name, MODEL_REGISTRY[name]) for name in selected_models]
        threading.Thread(target=self.run_process, args=(models_to_run, num_datasets), daemon=True).start()
        
    def run_process(self, models_to_run, num_datasets):
        try:
            run_benchmark(models_to_run, num_datasets=num_datasets, progress_callback=self.update_progress)
            self.root.after(0, self.finish_experiment, "✅ Experiment Completed Successfully!")
        except Exception as e:
            self.root.after(0, self.finish_experiment, f"❌ Error: {str(e)}")

    def update_progress(self, message):
        def _ui_update():
            self.lbl_status.config(text=message)
            if "Dataset" in message or "Processing" in message:
                current = self.progress["value"]
                if current < 95:
                    increment = 100 / (self.total_steps + 1)
                    self.progress["value"] = current + increment
        
        self.root.after(0, _ui_update)

    def finish_experiment(self, msg):
        self.lbl_status.config(text=msg, foreground="#41c985" if "Success" in msg else "#ff6b6b")
        self.btn_run.config(state="normal", text="🚀 Run Benchmark Experiment")
        self.is_running = False
        self.progress["value"] = 100
        messagebox.showinfo("Result", msg)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ModernBenchmarkGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
