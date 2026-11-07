| Query | kind | max | keyword (BM25) | semantic | hybrid (RRF) |
|---|---|---|---|---|---|
| `"event loop"` | original | 1.0 | 1.0 | 0.4 | 0.6 |
| `"regular expression"` | original | 0.4 | 0.4 | 0.2 | 0.4 |
| `unicode normalization` | original | 0.4 | 0.4 | 0.0 | 0.2 |
| `"garbage collector"` | original | 0.8 | 0.6 | 0.4 | 0.4 |
| `"context manager"` | original | 0.6 | 0.2 | 0.4 | 0.2 |
| `thread lock` | original | 0.6 | 0.4 | 0.2 | 0.4 |
| `json decode` | original | 0.2 | 0.2 | 0.2 | 0.2 |
| `pickle unpickling security` | original | 0.8 | 0.2 | 0.6 | 0.4 |
| `decorator wraps` | original | 0.2 | 0.2 | 0.0 | 0.2 |
| `"virtual environment"` | original | 0.8 | 0.6 | 0.8 | 0.8 |
| `run many jobs at once on all cores` | paraphrase | 1.0 | 0.2 | 0.2 | 0.2 |
| `turn objects into bytes and restore them later` | paraphrase | 1.0 | 0.0 | 0.2 | 0.0 |
| `spot textual patterns with wildcards` | paraphrase | 0.8 | 0.0 | 0.0 | 0.0 |
| `isolated sandbox with its own installed packages` | paraphrase | 0.8 | 0.0 | 0.4 | 0.4 |
| `wait for many network operations without stalling` | paraphrase | 1.0 | 0.0 | 0.0 | 0.0 |
| **mean, 10 original** | | 0.58 | **0.42** | **0.32** | **0.38** |
| **mean, 5 paraphrases** | | 0.92 | **0.04** | **0.16** | **0.12** |
| **mean, all 15** | | 0.69 | **0.29** | **0.27** | **0.29** |
