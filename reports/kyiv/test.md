# Test (21 Sep – 11 Oct 2026, weekly refits; choices frozen in choice_<task>.json before this run)


## onset (logreg, half-life none, recalibration True, features '+ alert map')

|   h |    n |   rate |   logloss_model |   brier_model |   auc_model |   logloss_baseline |   brier_baseline |   auc_baseline |   brier_skill |
|----:|-----:|-------:|----------------:|--------------:|------------:|-------------------:|-----------------:|---------------:|--------------:|
|  15 | 1215 |  0.133 |           0.362 |         0.108 |       0.709 |              0.403 |            0.117 |          0.493 |         0.078 |
|  30 | 1215 |  0.256 |           0.545 |         0.18  |       0.674 |              0.592 |            0.197 |          0.489 |         0.086 |
|  60 | 1215 |  0.442 |           0.69  |         0.245 |       0.632 |              0.719 |            0.261 |          0.496 |         0.061 |
| 120 | 1215 |  0.658 |           0.669 |         0.231 |       0.615 |              0.688 |            0.247 |          0.464 |         0.066 |
| 180 | 1215 |  0.778 |           0.552 |         0.177 |       0.621 |              0.581 |            0.195 |          0.452 |         0.093 |

Ablation on test (information only; Brier skill vs baseline per horizon):

| features          |   skill 15m |   skill 30m |   skill 60m |   skill 120m |   skill 180m |   mean_logloss |
|:------------------|------------:|------------:|------------:|-------------:|-------------:|---------------:|
| Kyiv history      |      -0.003 |       0.02  |       0.058 |        0.103 |        0.147 |          0.564 |
| + regional alerts |      -0.01  |       0.006 |       0.037 |        0.085 |        0.14  |          0.572 |
| + Air Force       |       0.058 |       0.053 |       0.038 |        0.069 |        0.103 |          0.57  |
| + alert map       |       0.078 |       0.086 |       0.061 |        0.066 |        0.093 |          0.564 |
| + Monitoring UA   |       0.071 |       0.072 |       0.048 |        0.061 |        0.091 |          0.568 |
| + Mykolaiv Vanek  |       0.082 |       0.084 |       0.05  |        0.058 |        0.088 |          0.567 |

Scenario view (test):

| scenario                                                      | risk budget   | forecast   |   threshold (val) | share of times 'go' (test)   | caught by an alert when 'go' (test)   |
|:--------------------------------------------------------------|:--------------|:-----------|------------------:|:-----------------------------|:--------------------------------------|
| Livoberezhna: cross now if P(alert within 30 min) ≤ threshold | ≤10% caught   | model      |             0.262 | 54.4%                        | 16.2%                                 |
| Livoberezhna: cross now if P(alert within 30 min) ≤ threshold | ≤10% caught   | baseline   |             0.008 | 0.0%                         | n/a                                   |
| Livoberezhna: cross now if P(alert within 30 min) ≤ threshold | ≤20% caught   | model      |             0.922 | 100.0%                       | 25.6%                                 |
| Livoberezhna: cross now if P(alert within 30 min) ≤ threshold | ≤20% caught   | baseline   |             0.269 | 86.5%                        | 25.2%                                 |

## clear (logreg, half-life none, recalibration True, features '+ alert map')

|   h |   n |   rate |   logloss_model |   brier_model |   auc_model |   logloss_baseline |   brier_baseline |   auc_baseline |   brier_skill |
|----:|----:|-------:|----------------:|--------------:|------------:|-------------------:|-----------------:|---------------:|--------------:|
|  15 | 787 |  0.189 |           0.428 |         0.135 |       0.763 |              0.492 |            0.156 |          0.552 |         0.136 |
|  30 | 787 |  0.33  |           0.547 |         0.183 |       0.754 |              0.638 |            0.223 |          0.549 |         0.182 |
|  60 | 787 |  0.53  |           0.597 |         0.205 |       0.749 |              0.688 |            0.247 |          0.569 |         0.169 |
| 120 | 787 |  0.736 |           0.543 |         0.178 |       0.712 |              0.565 |            0.188 |          0.567 |         0.05  |
| 180 | 787 |  0.835 |           0.4   |         0.125 |       0.755 |              0.438 |            0.133 |          0.559 |         0.059 |

Ablation on test (information only; Brier skill vs baseline per horizon):

| features          |   skill 15m |   skill 30m |   skill 60m |   skill 120m |   skill 180m |   mean_logloss |
|:------------------|------------:|------------:|------------:|-------------:|-------------:|---------------:|
| Kyiv history      |      -0.049 |      -0.075 |      -0.09  |       -0.068 |       -0.006 |          0.6   |
| + regional alerts |       0.106 |       0.145 |       0.143 |        0.086 |        0.119 |          0.511 |
| + Air Force       |       0.135 |       0.175 |       0.16  |        0.095 |        0.128 |          0.499 |
| + alert map       |       0.136 |       0.182 |       0.169 |        0.05  |        0.059 |          0.503 |
| + Monitoring UA   |       0.13  |       0.165 |       0.135 |        0.038 |        0.069 |          0.512 |
| + Mykolaiv Vanek  |       0.127 |       0.167 |       0.137 |        0.031 |        0.047 |          0.513 |

Scenario view (test):

| scenario                                                                          | forecast   |   rows (15-min) | right decision   | told to wait, all-clear came   | told to go, alert lasted >30 min   |
|:----------------------------------------------------------------------------------|:-----------|----------------:|:-----------------|:-------------------------------|:-----------------------------------|
| Arsenalna, all alerts: wait if P(all-clear within 30 min) ≥ 0.5                   | model      |             787 | 73.6%            | 67.6%                          | 75.0%                              |
| Arsenalna, all alerts: wait if P(all-clear within 30 min) ≥ 0.5                   | baseline   |             787 | 65.3%            | 19.0%                          | 66.6%                              |
| Arsenalna, red level (two-level system): wait if P(all-clear within 30 min) ≥ 0.5 | model      |              41 | 90.2%            | 100.0%                         | 87.9%                              |
| Arsenalna, red level (two-level system): wait if P(all-clear within 30 min) ≥ 0.5 | baseline   |              41 | 68.3%            | 0.0%                           | 70.0%                              |

All-clear within 30 min by alert level (test):

| level_now   |   rows |   all-clear within 30 min |   model brier |   baseline brier |
|:------------|-------:|--------------------------:|--------------:|-----------------:|
| yellow      |    746 |                     0.332 |         0.188 |            0.224 |
| red         |     41 |                     0.293 |         0.079 |            0.21  |
## Risk bands at fixed thresholds (added AFTER the test run; the model is unchanged — this only re-tabulates the same test predictions)


**Livoberezhna: P(new alert within 30 min)** (test, 1215 moments; overall observed 26%)

| forecast   | band   | share of 15-min moments   | what happened (observed share)   |
|:-----------|:-------|:--------------------------|:---------------------------------|
| model      | <10%   | 17%                       | 13%                              |
| model      | 10–20% | 23%                       | 17%                              |
| model      | 20–30% | 21%                       | 21%                              |
| model      | 30–50% | 26%                       | 35%                              |
| model      | ≥50%   | 12%                       | 48%                              |
| baseline   | <10%   | 2%                        | 20%                              |
| baseline   | 10–20% | 51%                       | 26%                              |
| baseline   | 20–30% | 41%                       | 25%                              |
| baseline   | 30–50% | 6%                        | 31%                              |
| baseline   | ≥50%   | 0%                        | —                                |

**Arsenalna: P(all-clear within 30 min)** (test, 787 moments; overall observed 33%)

| forecast   | band   | share of 15-min moments   | what happened (observed share)   |
|:-----------|:-------|:--------------------------|:---------------------------------|
| model      | <10%   | 19%                       | 13%                              |
| model      | 10–20% | 25%                       | 17%                              |
| model      | 20–30% | 18%                       | 32%                              |
| model      | 30–50% | 19%                       | 42%                              |
| model      | ≥50%   | 19%                       | 68%                              |
| baseline   | <10%   | 0%                        | 0%                               |
| baseline   | 10–20% | 9%                        | 18%                              |
| baseline   | 20–30% | 22%                       | 31%                              |
| baseline   | 30–50% | 66%                       | 36%                              |
| baseline   | ≥50%   | 3%                        | 19%                              |