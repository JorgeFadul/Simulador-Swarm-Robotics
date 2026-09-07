import yaml, os
for n in range(1, 6):
    f = f"mapas/mapa_test_{n}.yaml"
    with open(f, encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    g = d["grid_size"]; w = d["map_width"]
    print(f"mapa_test_{n}: {w}x{w} grid={g} celdas={w//g}x{w//g} robots={d.get('num_robots')}")
with open("mapas/mapa_rl_30x30.yaml", encoding="utf-8") as fh:
    d = yaml.safe_load(fh)
g = d["grid_size"]; w = d["map_width"]
print(f"mapa_rl_30x30: {w}x{w} grid={g} celdas={w//g}x{w//g} robots={d.get('num_robots')}")
