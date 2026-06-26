# S-MAS Policy Evaluation Summary

Evaluated across **30 seeds** for a maximum duration of **5.0 days** per seed.

| Policy | Mean Survival (Days) | Orbits Survived | Survived 5d (%) | Final SoC (%) | Final Fuel (%) | Target Images | SAA Violations |
|---|---|---|---|---|---|---|---|
| No-op (Passive) | 4.98 ± 0.11 | 73.2 | 96.7% | 44.8% | 100.0% | 0.0 | 0.0 |
| Random Policy | 5.00 ± 0.02 | 73.4 | 96.7% | 69.0% | 0.0% | 9282.1 | 0.0 |
| Rule-Based Heuristic | 5.00 ± 0.00 | 73.5 | 100.0% | 39.5% | 16.4% | 804.4 | 0.0 |
| IPPO (Independent PPO) | 5.00 ± 0.00 | 73.5 | 100.0% | 62.8% | 0.0% | 9335.7 | 0.0 |
| MAPPO (Ours) | 4.99 ± 0.04 | 73.3 | 93.3% | 66.3% | 43.7% | 1457.6 | 0.0 |
