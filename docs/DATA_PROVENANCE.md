# Where the data lives, and what the repository carries instead

The datasets are **not** in this repository and are not going to be. They live on
Rivanna at

```
/project/shakeri-lab/Alireza_timeseries/benchmark/
```

which today holds about 295 GB: 4.6 GB of `datasets_raw/`, 4.3 GB of
`datasets_processed/`, 54 GB of `outputs/`, 38 GB of `logs/` and 172 GB under
`surrogate/outputs/`. Four of the datasets — DCLP3, DCLP5, PEDAP and IOBP2 — are
real clinical trial data governed by their own agreements. Neither the size nor
the terms belong in a git remote.

That part already works. `.gitignore` excludes every dataset directory, and the
history confirms it held: `.git` is 19 MB across 3,946 files, and no clinical
data has ever been committed. Of the ten data-shaped tracked files, eight are
third-party upstream demo CSVs; the two that are ours are a 35 KB results table
and a constant risk matrix.

What was missing is the other half: if the bytes are not in the repository, the
repository has to say **which bytes**. Otherwise "which split did this run use"
has no machine answer, and a dataset can change under a finished result with
nothing to notice.

## The rule

> Datasets live on `/project`. Git carries their hashes, and the code that made
> them.

Three pieces implement it.

**`tools/dataset_manifest.py`** writes `MANIFEST.sha256.json` into a dataset
directory: every file with its size and SHA-256, plus one `directory_sha256`
over the whole tree. That one digest is what a config, a paper table or an
experiment record cites.

```bash
python tools/dataset_manifest.py write  datasets_processed/DCLP3
python tools/dataset_manifest.py verify datasets_processed/DCLP3
```

`verify` exits non-zero and lists what moved. Run it before believing a result
you did not produce this week. Hashing the whole tree is a batch job, not
login-node work — use `tools/manifest.slurm`.

**`generate_splits.py` is now tracked.** It was in `.gitignore`, and its own
docstring said so while also claiming the splits it writes are "frozen for
reproducibility". Both cannot be true. The split decides which patient is in
test for every model in the benchmark; a frozen split whose generator exists
only as one file on one filesystem is reproducible exactly as long as that file
survives. `apply_splits.py` and `preprocess.py` were already tracked, so the
repository could apply a split it could not produce.

**`.githooks/pre-commit`** refuses to stage anything from a dataset, output or
log directory, and anything over 5 MiB. Install it once per clone:

```bash
git config core.hooksPath .githooks
```

The ignore rules state an intention; the hook enforces it. With ~570
uncommitted entries in a working tree at any time, a single `git add -A` is all
that stands between the current state and a dataset in the history — and a
clinical dataset pushed once cannot be unpushed, only rotated.

## What still needs a person

- **Rights pointers.** No dataset directory records an IRB or DUA reference.
  Four of them need one, and no script can invent it.
- **A version, not a date.** The manifests give each dataset a content hash;
  deciding what counts as version 1 of DCLP3-processed is a human call.
- **Where `outputs/` belongs.** 54 GB of results under `/project` is archive
  space used as a workshop. `/scratch` is the workshop and is purged at 90 days,
  which is a feature for anything regenerable and a hazard for anything not.
