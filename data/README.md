# Starlink Dataset Preparation

The experiments use public real-world Starlink traces released with the
[StarNet measurement study](https://github.com/ConnectedSystemsLab/StarNet).
The collection covers Chicago, Osnabrueck, and Victoria and combines
continuous throughput measurements with latency, serving-satellite geometry,
candidate-satellite counts, local weather, and timestamps.

## Official Sources

| Region used here | Official release | Prepared label | Window stride |
|---|---|---:|---:|
| Chicago, United States | [US data](https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/IgBaSUJKNIjlQI6KeF1tWc6-AZMwiOlyueqSY4KzOYsVEfw?e=pD0Ix8) | `chi` | 46 |
| Osnabrueck, Germany | [Germany data](https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/IgBfbwMBpJDTSqbqvty7kK52AalnjRwNyiW-Xomy-iUD01A?e=ycoFNf) | `osn` | 29 |
| Victoria, Canada | [Canada data](https://uillinoisedu-my.sharepoint.com/:f:/g/personal/zikunliu_illinois_edu/IgDD3A4WdkPjQ4ME-CCIpk0GAfO40d2gSYm2k5skQMJVf-E?e=6eekvm) | `vic` | 6 |

Follow the official project to obtain and preprocess a regional trace into
`dataset_tp_sat.pkl`. Then create the forecasting windows used by D-LACER:

```bash
python scripts/prepare_starnet.py \
  --input /path/to/dataset_tp_sat.pkl \
  --output data/processed/chi.npz \
  --name chi \
  --step-len 46
```

Use `--step-len 29` for Osnabrueck and `--step-len 6` for Victoria.

## Processing Pipeline

`src/dlacer/data.py` performs the following operations:

1. sorts records chronologically and removes incomplete feature rows;
2. encodes serving-satellite identity and derives second, minute, hour, and
   weekday features;
3. uses 14 synchronized channels, including throughput, latency, satellite
   geometry, candidate count, weather, and time information;
4. maps 75 historical one-second observations to the next 15 throughput
   targets;
5. constructs target-free context summaries for latent-state routing; and
6. creates four purged chronological splits with proportions 55/15/15/15.

The purge interval is computed from the input length, prediction horizon, and
window stride so adjacent splits do not share raw observations.

## Dataset Citation

```bibtex
@article{liu2025vivisecting,
  title     = {Vivisecting Starlink Throughput: Measurement and Prediction},
  author    = {Liu, Zikun and Reidys, Fan-Xue Gabriella and Tanveer, Sarah and Vasisht, Deepak},
  journal   = {Proceedings of the ACM on Networking},
  volume    = {3},
  number    = {CoNEXT4},
  pages     = {1--23},
  year      = {2025},
  publisher = {ACM New York, NY, USA}
}
```
