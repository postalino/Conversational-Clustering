"""Rende importabili i moduli di src/ nei test (stesso effetto dell'inserimento fatto dalla GUI)."""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
