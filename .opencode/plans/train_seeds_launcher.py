"""Lanza 20 entrenamientos RL-known en mapa 30x30 con paredes, 4 en paralelo.

Cada run: 150000 pasos, se referencias como rl_30x30_seedNN.
Logs: log_train30_seedNN.txt + master en log_train30_master.txt
"""
import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = r"C:\Users\jorge\OneDrive\Escritorio\Maestría CIM\TESIS\TESIS\simulador-swarm-2d"
EXE = sys.executable
SEEDS = list(range(1, 21))
CONCURRENCY = 4

MASTER_LOG = os.path.join(ROOT, "log_train30_master.txt")
_master = open(MASTER_LOG, "w", encoding="utf-8")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _master.write(line + "\n")
    _master.flush()


def build_cmd(seed):
    name = f"rl_30x30_seed{seed:02d}"
    run_log = os.path.join(ROOT, f"log_train30_seed{seed:02d}.txt")
    cmd = [
        EXE, "-m", "src.rl.train",
        "--mapa", "mapas/mapa_rl_30x30.yaml",
        "--map-known",
        "--steps", "150000",
        "--n-steps", "1024",
        "--batch-size", "64",
        "--fps-physics", "5",
        "--seed", str(seed),
        "--model-name", name,
    ]
    return cmd, run_log, name


def run_one(job):
    cmd, run_log, name = job
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "3"
    env["MKL_NUM_THREADS"] = "3"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    try:
        with open(run_log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        status = "OK" if r.returncode == 0 else f"RC={r.returncode}"
    except Exception as e:
        status = f"ERR {e}"
    log(f"seed {name} terminado: {status} ({time.time()-t0:.0f}s)")
    return name, status, time.time() - t0


def main():
    log(f"Inicio: {len(SEEDS)} entrenamientos, {CONCURRENCY} concurrentes, exe={EXE}")
    jobs = [(build_cmd(s)) for s in SEEDS]
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for r in ex.map(run_one, jobs):
            results.append(r)
    log("TODOS TERMINADOS")
    for r in results:
        log(f"SMMARY {r[0]} {r[1]} {r[2]:.0f}s")


if __name__ == "__main__":
    main()