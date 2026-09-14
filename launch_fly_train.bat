@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe train_bio_fly.py --runs 60 --robots 10 --mapas mapas/mapa_rl_20x20_10r.yaml --kc 1000 --max-steps 5000 --max-segundos 300 --defectos-semilla 100 --out models/bio/fly_swarm_latest.npz --reporte models/bio/fly_swarm_curve3.csv --semilla 42 --reset-weights > models\bio\fly_train.log 2>&1
