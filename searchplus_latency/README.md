# Search+ cost measurement (standalone)

Measures the per-query cost of Search+ (query-aware and query-unaware KG) and of
list-wise LLM reranking on the same 100 random test questions per dataset.

Requirements: **Python 3.8+** and **Ollama**. No other package, no repository.

```
searchplus_latency/
    latency.py        the script
    subsample.json    100 test questions per dataset (HotpotQA, 2Wiki, MuSiQue), seed 0
```

## Run

```bash
ollama serve                         # in another terminal
ollama pull gemma4:latest            # or the tag used in the paper
python latency.py                    # everything: ~900 LLM calls
python latency.py --model gemma4:e4b
```

Test without Ollama: `python latency.py --mock --n 3`.

Stop with Ctrl+C at any time and relaunch the same command to continue.
Run nothing else on the GPU while it measures.

## What is measured, per question

| Method | Steps timed |
|---|---|
| Search+ query-aware | LLM KG extraction (MineKG.py prompt with the question) + graph construction (BuildGraphsPerQuestion.py) + BFS / NSP |
| Search+ query-unaware | same, MineKG.py prompt without the question |
| List-wise LLM | one ranking call (BaselineLLM.py prompt, full passages, BM25 order, temperature 0) |

All three methods run on the same questions, the order of the methods rotates
question by question, the first calls are an untimed warm-up, and all calls use
the same context window (`--num-ctx`, default 16384) so no prompt is truncated.
The code of the prompts, of the graph builder and of BFS/NSP is copied from the
repository unchanged.

Options: `--datasets musique`, `--modes aware unaware`, `--n 50`,
`--listwise-max-chars 200` (old BaselineLLM.py setting), `--num-ctx 0` (Ollama default),
`--num-predict -1 --out results_nolimit` (list-wise without the 1024-token generation limit,
which with thinking models also counts the thinking tokens).

## Output (`results/`)

* `summary.csv` — per dataset and method: LLM time (mean, std, median, p95), total
  time, graph-build / BFS / NSP time in ms, prompt and output tokens, tokens/s,
  invalid-output rate, call-error rate, incomplete permutations (list-wise).
* `latency_table.tex` — LaTeX table for the paper.
* `perquery_<dataset>.jsonl` — every measurement.

Send back the `results/` folder.

## Without the 1024-token limit (only the failed answers are redone)

The list-wise call keeps BaselineLLM.py's limit of 1024 generated tokens, which
with thinking models also counts the thinking tokens. To remove it without
redoing everything:

```bash
python latency.py --num-predict -1 --out results_nolimit --rerun-failed-from results
```

It copies every measurement from `results/` into `results_nolimit/` and calls the
LLM again only for list-wise answers that failed or reached the limit (with
temperature 0, the others would be identical). Send back `results_nolimit/` too.
