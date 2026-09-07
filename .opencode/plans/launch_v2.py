"""Lanza entrenamiento RL Hibrido V2 en background con pythonw."""
import os
import subprocess

ROOT = r"C:\Users\jorge\OneDrive\Escritorio\Maestría CIM\TESIS\TESIS\simulador-swarm-2d"
PYW = r"C:\Python311\pythonw.exe"

cmd = [
    PYW, "-m", "src.rl.train",
    "--mapa", "mapas/mapa_rl_v2.yaml",
    "--map-known",
    "--steps", "200000",
    "--n-envs", "6",
    "--n-steps", "1024",
    "--batch-size", "64",
    "--fps-physics", "5",
    "--seed", "42",
    "--model-name", "rl_v2",
]

out = os.path.join(ROOT, "log_train_v2.txt")
err = os.path.join(ROOT, "log_train_v2_err.txt")
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONIOENCODING"] = "utf-8"

with open(out, "w", encoding="utf-8") as fo, open(err, "w", encoding="utf-8") as fe:
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fo, stderr=fe)
print(f"PID={p.pid}")