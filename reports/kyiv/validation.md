# Validation (24 Aug – 20 Sep 2026, weekly refits)


## onset

Baseline mean log-loss over horizons: 0.6020

Model choice (all features):

| model   | half_life_days   | recalibration   |   mean_logloss |
|:--------|:-----------------|:----------------|---------------:|
| logreg  | -                | False           |         0.5155 |
| logreg  | -                | True            |         0.4965 |
| logreg  | 45               | False           |         0.5633 |
| logreg  | 45               | True            |         0.5447 |
| logreg  | 14               | False           |         0.6548 |
| logreg  | 14               | True            |         0.6464 |
| hgb     | 45               | False           |         0.6371 |
| hgb     | 45               | True            |         0.6277 |
| hgb     | 14               | False           |         0.6684 |
| hgb     | 14               | True            |         0.6523 |

Feature groups added in turn (logreg, half-life none, recalibration True):

| features          |   mean_logloss |   vs_baseline |
|:------------------|---------------:|--------------:|
| Kyiv history      |         0.5499 |       -0.0521 |
| + regional alerts |         0.5466 |       -0.0554 |
| + Air Force       |         0.517  |       -0.085  |
| + alert map       |         0.4835 |       -0.1185 |
| + Monitoring UA   |         0.4921 |       -0.1099 |
| + Mykolaiv Vanek  |         0.4965 |       -0.1055 |

**Chosen for onset: logreg, half-life none, recalibration True, features '+ alert map'.**

## clear

Baseline mean log-loss over horizons: 0.5649

Model choice (all features):

| model   | half_life_days   | recalibration   |   mean_logloss |
|:--------|:-----------------|:----------------|---------------:|
| logreg  | -                | False           |         0.629  |
| logreg  | -                | True            |         0.5982 |
| logreg  | 45               | False           |         0.7671 |
| logreg  | 45               | True            |         0.7229 |
| logreg  | 14               | False           |         0.8097 |
| logreg  | 14               | True            |         0.7699 |
| hgb     | 45               | False           |         0.7573 |
| hgb     | 45               | True            |         0.6157 |
| hgb     | 14               | False           |         0.7462 |
| hgb     | 14               | True            |         0.6167 |

Feature groups added in turn (logreg, half-life none, recalibration True):

| features                 |   mean_logloss |   vs_baseline |
|:-------------------------|---------------:|--------------:|
| Kyiv history             |         0.5964 |        0.0316 |
| + regional alerts        |         0.551  |       -0.0139 |
| + Air Force              |         0.5562 |       -0.0086 |
| + alert map              |         0.5447 |       -0.0202 |
| + Monitoring UA          |         0.5705 |        0.0057 |
| + Mykolaiv Vanek         |         0.5982 |        0.0333 |
| baseline only (no model) |         0.5649 |        0      |

**Chosen for clear: logreg, half-life none, recalibration True, features '+ alert map'.**