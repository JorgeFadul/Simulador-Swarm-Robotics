"""Lanza GUI sin --mapa, modo RL Hibrido por defecto."""
import os
import subprocess
import sys

ROOT = r"C:\Users\jorge\OneDrive\Escritorio\Maestría CIM\TESIS\TESIS\simulador-swarm-2d"
PYW = r"C:\Python311\pythonw.exe"

cmd = [
    PYW, "-m", "main",
]

out = os.path.join(ROOT, "log_gui_50x50.txt")
err = os.path.join(ROOT, "log_gui_50x50_err.txt")

with open(out, "w", encoding="utf-8") as fo, open(err, "w", encoding="utf-8") as fe:
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=fo, stderr=fe)
print(f"PID={p.pid}")