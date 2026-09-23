# pipe_smooth

Stage-1 centerline refine after discrete routing (`pipe_design`). Hermite graphs + soft/hard clearance; metrics in `report.json`.

See `ROADMAP.md` for the method. Package survey: `package.md`.

---

## Generate eval instances (`gen_robust.sh`)

Batch-build **plane_slice** + **pillar_field** layouts for Stage-1 eval via `pipe_design`’s `main_gen`.

### Prerequisites

Build `main_gen` first (or set `MAIN_GEN`):

```bash
# from repo root, typical
cmake -S pipe_design -B pipe_design/build && cmake --build pipe_design/build --target main_gen
```

Script looks for `pipe_design/build/main_gen` by default.

### Usage

```bash
cd pipe_smooth
chmod +x gen_robust.sh   # once

./gen_robust.sh [OUT_DIR] [NUM] [RADIUS] [SHELL_MARGIN]
```

| Arg | Default | Meaning |
|-----|---------|---------|
| `OUT_DIR` | `pipe_smooth/data/bench` | Output root |
| `NUM` | `50` | Target YAMLs per level (`easy` / `hard`) |
| `RADIUS` | `1.5` | Pipe radius (must be `> 0`) |
| `SHELL_MARGIN` | *(omit)* | Optional; else `main_gen` default |

Examples:

```bash
# defaults → data/bench, 50 each, r=1.5
./gen_robust.sh

# custom out + count + radius
./gen_robust.sh ./data/bench_r2 20 2.0

# set binary explicitly
MAIN_GEN=/path/to/main_gen ./gen_robust.sh
```

Optional env knobs: `SIZE` (default 48), `CORRIDOR_MARGIN`, `PLANE_SLICE_ATTEMPT_MULT`, `MAX_FILL_ROUNDS`.

### Output layout

```text
OUT_DIR/
  robust_summary.txt
  easy/instance_*.yaml
  hard/instance_*.yaml
```

- **easy**: 3 groups × 3 terminals  
- **hard**: 8 groups × 8 terminals  
- Obstacle always `pillar_field`; generator always `plane_slice`.

Low success → fill rounds until `NUM` or `MAX_FILL_ROUNDS`.

### Next: Stage-1 run

Prepare usages / run smooth (obstacles currently stripped by the loader unless you inject them):

```bash
python example_load_env.py data/bench/easy/
python -m pipe_smooth.pipeline data/bench/easy/instance_0.yaml -o runs/stage1
```
