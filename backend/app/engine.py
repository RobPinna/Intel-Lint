from __future__ import annotations

import os

from .engine_ollama import run_analysis as run_ollama_analysis
from .engine_placeholder import run_analysis
from .models import AnalyzeRequest, AnalyzeResponse


def analyze_with_selected_engine(request: AnalyzeRequest) -> AnalyzeResponse:
    engine = os.getenv("ENGINE", "placeholder").lower()

    if engine == "placeholder":
        return run_analysis(request)
    if engine == "ollama":
        return run_ollama_analysis(request)

    raise ValueError(f"Unsupported ENGINE '{engine}'. Use ENGINE=placeholder|ollama.")
