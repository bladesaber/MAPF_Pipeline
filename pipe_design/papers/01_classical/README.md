# 01_classical — paper draft

**Scope:** problem formalization + constructive 3D benchmark + CBS / PBS / PP baselines
under permanent geometric occupancy.

**Source of truth:** LaTeX under [`latex/`](latex/). Build preview:

```bash
cd pipe_design/papers/01_classical/latex && pdflatex preview.tex
```

| File | Section |
|------|---------|
| `sec_abstract.tex` | Abstract |
| `sec_problem.tex` | Intro, contributions, problem |
| `sec_method.tex` | Baseline planner stack |
| `sec_benchmark.tex` | Suite + evaluation protocol |
| `sec_results.tex` | Experiments (from `data/runs`) |
| `sec_related.tex` | Related work |
| `sec_discussion.tex` | Limitations + conclusion |
| `sec_benchmark_appendix.tex` | Generator pseudocode |

**Reproducibility (frozen for the paper):**

- Repository: https://github.com/bladesaber/MAPF_Pipeline  
- Branch: `branch_quan`  
- Commit: `cf0005685626dcaedb4c96a64fdca554312dfba1`  
- Scripts: `pipe_design/include/benchmark/gen_robust.sh`, `run_bench.sh`, `pipe_design/scripts/analyze_runs.py`  
- Results: `pipe_design/data/runs/` (36 JSONL batches)

**Numbers** must match `gen_robust.sh` / actual JSONL budgets at the commit above.

Shared reading list: `mapf_paper/ref/` (CBS, PBS, Stern benchmarks, Belov PR, Blokland pipe survey).

**Deprecated:** `tech_note_v0.md` — superseded by the LaTeX draft; do not update.
