"""Debug: probar si la política responde a diferentes observaciones."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO


def test_policy():
    model_path = "models/ppo_swarm_v2/final"
    model = PPO.load(model_path)

    print("=== Probando política con diferentes observaciones ===")

    # Observación aleatoria
    obs_random = np.random.randn(105).astype(np.float32)
    action_random, _ = model.predict(obs_random, deterministic=True)
    print(f"Obs aleatoria -> action: {action_random}")

    # Observación con robots en diferentes posiciones
    obs_test = np.zeros(105, dtype=np.float32)
    obs_test[:100] = 0.5  # Todo pared
    obs_test[100] = 0.1   # Robot 1 x
    obs_test[101] = 0.9   # Robot 1 y
    obs_test[102] = 0.9   # Robot 2 x
    obs_test[103] = 0.1   # Robot 2 y
    obs_test[104] = 0.0   # Paso 0
    action_test, _ = model.predict(obs_test, deterministic=True)
    print(f"Obs test (robots en esquinas) -> action: {action_test}")

    # Observación con cobertura completa
    obs_full = np.zeros(105, dtype=np.float32)
    obs_full[:100] = 1.0  # Todo inspeccionado
    obs_full[100:104] = 0.5  # Robots en centro
    obs_full[104] = 1.0  # Paso final
    action_full, _ = model.predict(obs_full, deterministic=True)
    print(f"Obs full (cobertura completa) -> action: {action_full}")

    # Observación con paso alto
    obs_late = np.zeros(105, dtype=np.float32)
    obs_late[:100] = 0.0  # Nada inspeccionado
    obs_late[100:104] = 0.5
    obs_late[104] = 0.9  # Paso 90%
    action_late, _ = model.predict(obs_late, deterministic=True)
    print(f"Obs late (paso alto) -> action: {action_late}")

    # Verificar si la política es constante
    print("\n=== Verificando variabilidad de la política ===")
    actions = []
    for i in range(10):
        obs = np.random.randn(105).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action.tolist())
    unique_actions = [list(x) for x in set(tuple(a) for a in actions)]
    print(f"10 predicciones con observaciones aleatorias:")
    print(f"  Acciones únicas: {len(unique_actions)}")
    for a in unique_actions[:5]:
        print(f"    {a}")


if __name__ == "__main__":
    test_policy()
