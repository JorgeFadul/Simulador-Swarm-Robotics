"""Lanza GUI con RL Hibrido V2."""
import os
import subprocess

ROOT = r"C:\Users\jorge\OneDrive\Escritorio\Maestría CIM\TESIS\TESIS\simulador-swarm-2d"
PYW = r"C:\Python311\pythonw.exe"

cmd = [
    PYW, "-m", "main",
    "--estrategia-hibrido", "RL Hibrido V2",
]

out = os.path.join(ROOT, "log_gui_v2.txt")
err = os.path.join(ROOT, "log_gui_v2_err.txt")
env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"

with open(out, "w", encoding="utf-8") as fo, open(err, "w", encoding="utf-8") as fe:
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fo, stderr=fe)
print(f"PID={p.pid}")