"""Tkinter setup window to choose files and alert parameters."""

from __future__ import annotations

import os
import tkinter as tk
from dataclasses import replace
from tkinter import filedialog, messagebox
from typing import Optional

from proximity_alert.config import Config


class SetupWindow:
    """Setup dialog that returns an updated :class:`Config`.

    Args:
        defaults: Values used to prefill the fields.
    """

    def __init__(self, defaults: Config) -> None:
        self._defaults = defaults
        self._result: Optional[Config] = None
        self.root = tk.Tk()
        self.root.title("Proximity Alert Setup")
        self.root.geometry("600x480")
        self.input_path = defaults.input_path
        self.output_path = defaults.output_path
        self.vars = {
            "imgsz": tk.StringVar(value=str(defaults.imgsz)),
            "conf": tk.StringVar(value=f"{defaults.conf:.2f}"),
            "time_window": tk.StringVar(value=str(defaults.time_window)),
            "min_ratio": tk.StringVar(value=f"{defaults.min_ratio * 100:g}"),
            "depth_threshold": tk.StringVar(value=str(defaults.depth_threshold)),
            "height_factor": tk.StringVar(value=str(defaults.height_factor)),
        }
        self._build()

    def _build(self) -> None:
        main = tk.Frame(self.root, padx=15, pady=15)
        main.pack(fill="both", expand=True)

        files = tk.LabelFrame(main, text=" Files ", padx=10, pady=10)
        files.pack(fill="x", pady=5)
        tk.Button(files, text="1. Select Input Video", command=self._select_input, width=20).grid(
            row=0, column=0, padx=5, pady=5
        )
        self.lbl_input = tk.Label(files, text=self._label(self.input_path, "No file selected"), fg="gray")
        self.lbl_input.grid(row=0, column=1, sticky="w", padx=5)
        tk.Button(files, text="2. Select Output Path", command=self._select_output, width=20).grid(
            row=1, column=0, padx=5, pady=5
        )
        self.lbl_output = tk.Label(files, text=self._label(self.output_path, "No path selected"), fg="gray")
        self.lbl_output.grid(row=1, column=1, sticky="w", padx=5)

        params = tk.LabelFrame(main, text=" Alert Parameters ", padx=10, pady=10)
        params.pack(fill="both", expand=True, pady=10)
        fields = [
            ("YOLO Resolution (IMGSZ):", "imgsz"),
            ("YOLO Confidence Threshold:", "conf"),
            ("Time Window (Sec):", "time_window"),
            ("Min Alert Ratio (%):", "min_ratio"),
            ("Delta Z Threshold (Depth):", "depth_threshold"),
            ("2D Radius Factor (x BBox Height):", "height_factor"),
        ]
        for row, (text, key) in enumerate(fields):
            tk.Label(params, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            tk.Entry(params, textvariable=self.vars[key], width=12).grid(row=row, column=1, sticky="w", padx=5, pady=2)

        tk.Button(
            main,
            text="Start Processing",
            command=self._start,
            bg="#28a745",
            fg="white",
            font=("Arial", 10, "bold"),
            height=2,
        ).pack(fill="x", pady=5)

    @staticmethod
    def _label(path: str, empty: str) -> str:
        return os.path.basename(path) if path else empty

    def _select_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.input_path = path
            self.lbl_input.config(text=os.path.basename(path), fg="black")

    def _select_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if path:
            self.output_path = path
            self.lbl_output.config(text=os.path.basename(path), fg="black")

    def _start(self) -> None:
        if not self.input_path or not self.output_path:
            messagebox.showerror("Error", "Please select both Input and Output file paths!")
            return
        try:
            self._result = replace(
                self._defaults,
                input_path=self.input_path,
                output_path=self.output_path,
                imgsz=int(self.vars["imgsz"].get()),
                conf=float(self.vars["conf"].get()),
                time_window=float(self.vars["time_window"].get()),
                min_ratio=float(self.vars["min_ratio"].get()) / 100.0,
                depth_threshold=float(self.vars["depth_threshold"].get()),
                height_factor=float(self.vars["height_factor"].get()),
            )
        except ValueError:
            messagebox.showerror("Error", "Please verify that numerical parameters are valid!")
            return
        self.root.quit()
        self.root.destroy()

    def run(self) -> Optional[Config]:
        """Shows the window and blocks until it is closed.

        Returns:
            The chosen configuration, or ``None`` if the window was closed without
            pressing *Start Processing*.
        """
        self.root.mainloop()
        return self._result


def ask_config(defaults: Config) -> Optional[Config]:
    """Opens the setup window prefilled with ``defaults``."""
    return SetupWindow(defaults).run()
