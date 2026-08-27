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
| `sec_related.tex` | Related work (TODO cites) |
| `sec_discussion.tex` | Limitations + conclusion |
| `sec_benchmark_appendix.tex` | Generator pseudocode |

**Numbers** must match `gen_robust.sh` / actual JSONL budgets, not outdated notes.
Shared reading list: \texttt{mapf\_paper/ref/} (CBS, PBS, Stern benchmarks,
Belov PR, Blokland pipe survey, etc.).
do not update it. Remove when no longer needed for history.
