# Analysis scripts

Small scripts behind numbers quoted in the report that the pipeline does not produce itself.

- `pead_check.py`: earnings-surprise quintiles, announcement-day reaction vs later drift, drift by period, clean-event version.
- `goodnews.py`: how many best-news earnings reports rose or fell afterwards.
- `twobytwo.py`: the 2x2 split (rose/fell on the day x rose/fell afterwards) of the best-news group.

Run from the repo's virtual environment with `RUN_ROOT` pointing at a finished run folder (the one that contains `S20`, `S09`, ...).
