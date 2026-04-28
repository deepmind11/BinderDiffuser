# Figures

Generated artifacts referenced from the top-level `README.md`.

| File | What it shows |
|------|---------------|
| `hero.png` | Three-panel motif-scaffolding cartoon: target + motif → diffused scaffold → validated binder |
| `pipeline_diagram.png` | End-to-end pipeline stages (target → RFdiffusion → ProteinMPNN → AlphaFold → filter+rank) |
| `self_consistency_scatter.png` | scRMSD vs mean pLDDT scatter coloured by ipTM (illustrative cohort, n=100) |
| `metrics_violin.png` | Per-metric distribution with top-10 designs overlaid (illustrative cohort) |
| `sample_designs.csv` | Synthetic 100-design cohort used to render the scatter and violin (for layout/visualization only — not real RFdiffusion+AF outputs) |

## Reproducing

The figures here were rendered programmatically; the script lives at the top
of the repo and can be regenerated with the helpers in
`src/binderdiffuser/viz.py`. The synthetic cohort in `sample_designs.csv` is
labelled illustrative because it was generated to exercise the plotting code,
not by running RFdiffusion + ColabFold. The figure regeneration is part of
the dev loop:

```python
from binderdiffuser.viz import self_consistency_scatter, metrics_violin
import pandas as pd
df = pd.read_csv("figures/sample_designs.csv")
self_consistency_scatter(df, "figures/self_consistency_scatter.png")
metrics_violin(df, "figures/metrics_violin.png")
```
