
### volunteer

Chosen on validation per horizon: 1 h: hgb + online recal, train from 2025-01-01, features '+Air Force text'; 3 h: hgb + online recal, train from 2025-01-01, features '+Air Force text'; 6 h: logreg + online recal, train from 2025-12-01, features '+Air Force text'; 12 h: logreg + online recal, train from 2025-12-01, features '+Air Force text'; 24 h: logreg, train from 2025-12-01, features '+spatial'

Ablation, Brier skill vs seasonal baseline on test (0 = baseline, higher is better):

| features        |      1 |      3 |      6 |     12 |     24 |
|:----------------|-------:|-------:|-------:|-------:|-------:|
| calendar        | -0.052 | -0.147 | -0.191 | -0.264 | -0.917 |
| +recent rates   |  0.002 | -0.029 | -0.046 | -0.105 | -0.577 |
| +own history    |  0.032 |  0.014 |  0.001 | -0.023 | -0.222 |
| +spatial        |  0.03  |  0.013 |  0.014 |  0.022 | -0.053 |
| +Air Force text |  0.035 |  0.041 |  0.029 |  0.058 |  0.025 |

Selected model vs seasonal baseline, test:

|   horizon_h |   onset rate |   Brier model |   Brier baseline |   skill |   log-loss model |   log-loss baseline |   AUC model |   AUC baseline |
|------------:|-------------:|--------------:|-----------------:|--------:|-----------------:|--------------------:|------------:|---------------:|
|           1 |        0.234 |         0.148 |            0.153 |   0.035 |            0.452 |               0.491 |       0.775 |          0.749 |
|           3 |        0.494 |         0.176 |            0.183 |   0.041 |            0.519 |               0.55  |       0.814 |          0.799 |
|           6 |        0.67  |         0.135 |            0.139 |   0.029 |            0.423 |               0.449 |       0.854 |          0.853 |
|          12 |        0.798 |         0.076 |            0.08  |   0.058 |            0.266 |               0.296 |       0.919 |          0.913 |
|          24 |        0.869 |         0.047 |            0.045 |  -0.053 |            0.162 |               0.15  |       0.975 |          0.969 |

Operational, test (warning thresholds set on validation):

|   horizon_h |   cap_false_per_loc_week |   threshold_from_val |   test_false_per_loc_week |   onsets_warned |   onsets |   seasonal_onsets_warned |
|------------:|-------------------------:|---------------------:|--------------------------:|----------------:|---------:|-------------------------:|
|           1 |                        1 |                0.678 |                     0.917 |           0.083 |     4257 |                    0.09  |
|           1 |                        3 |                0.601 |                     2.865 |           0.179 |     4257 |                    0.178 |
|           1 |                        7 |                0.505 |                     7.365 |           0.314 |     4257 |                    0.322 |
|           3 |                        1 |                0.882 |                     1.135 |           0.339 |     4248 |                    0.376 |
|           3 |                        3 |                0.814 |                     3.281 |           0.506 |     4248 |                    0.499 |
|           3 |                        7 |                0.737 |                     7.469 |           0.651 |     4248 |                    0.65  |
|           6 |                        1 |                0.92  |                     1.573 |           0.572 |     4236 |                    0.614 |
|           6 |                        3 |                0.869 |                     4.073 |           0.735 |     4236 |                    0.746 |
|           6 |                        7 |                0.767 |                     9.458 |           0.869 |     4236 |                    0.898 |
|          12 |                        1 |                0.931 |                     2.177 |           0.861 |     4189 |                    0.894 |
|          12 |                        3 |                0.824 |                     4.344 |           0.959 |     4189 |                    0.934 |
|          12 |                        7 |                0.632 |                     6.823 |           0.983 |     4189 |                    0.981 |
|          24 |                        1 |                0.936 |                     0.26  |           0.93  |     4128 |                    0.948 |
|          24 |                        3 |                0.733 |                     0.615 |           0.981 |     4128 |                    0.987 |
|          24 |                        7 |                0.471 |                     1.781 |           0.994 |     4128 |                    0.995 |

### official

Chosen on validation per horizon: 1 h: hgb + online recal, train from 2025-01-01, features '+own history'; 3 h: logreg + online recal, train from 2025-12-01, features '+Air Force text'; 6 h: hgb + online recal, train from 2025-01-01, features '+Air Force text'; 12 h: logreg + online recal, train from 2025-12-01, features '+Air Force text'; 24 h: logreg + online recal, train from 2025-01-01, features '+Air Force text'

Ablation, Brier skill vs seasonal baseline on test (0 = baseline, higher is better):

| features        |      1 |      3 |      6 |     12 |     24 |
|:----------------|-------:|-------:|-------:|-------:|-------:|
| calendar        | -0.02  | -0.053 | -0.102 | -0.043 | -0.072 |
| +recent rates   |  0.005 | -0.026 | -0.04  | -0.022 | -0.03  |
| +own history    |  0.131 |  0.118 |  0.095 |  0.114 |  0.001 |
| +spatial        |  0.143 |  0.119 |  0.083 |  0.148 | -0.003 |
| +Air Force text |  0.12  |  0.114 |  0.011 |  0.127 |  0.001 |

Selected model vs seasonal baseline, test:

|   horizon_h |   onset rate |   Brier model |   Brier baseline |   skill |   log-loss model |   log-loss baseline |   AUC model |   AUC baseline |
|------------:|-------------:|--------------:|-----------------:|--------:|-----------------:|--------------------:|------------:|---------------:|
|           1 |        0.116 |         0.083 |            0.095 |   0.131 |            0.275 |               0.332 |       0.834 |          0.76  |
|           3 |        0.275 |         0.138 |            0.156 |   0.114 |            0.418 |               0.477 |       0.848 |          0.807 |
|           6 |        0.389 |         0.161 |            0.163 |   0.011 |            0.481 |               0.511 |       0.842 |          0.837 |
|          12 |        0.491 |         0.137 |            0.157 |   0.127 |            0.428 |               0.547 |       0.891 |          0.858 |
|          24 |        0.581 |         0.14  |            0.14  |   0.001 |            0.473 |               0.545 |       0.88  |          0.873 |

Operational, test (warning thresholds set on validation):

|   horizon_h |   cap_false_per_loc_week |   threshold_from_val |   test_false_per_loc_week |   onsets_warned |   onsets |   seasonal_onsets_warned |
|------------:|-------------------------:|---------------------:|--------------------------:|----------------:|---------:|-------------------------:|
|           1 |                        1 |                0.509 |                     1.052 |           0.139 |     1961 |                    0.024 |
|           1 |                        3 |                0.42  |                     2.542 |           0.197 |     1961 |                    0.065 |
|           1 |                        7 |                0.354 |                     5.542 |           0.287 |     1961 |                    0.139 |
|           3 |                        1 |                0.77  |                     0.635 |           0.209 |     1957 |                    0.145 |
|           3 |                        3 |                0.715 |                     2.146 |           0.346 |     1957 |                    0.264 |
|           3 |                        7 |                0.641 |                     5.177 |           0.508 |     1957 |                    0.404 |
|           6 |                        1 |                0.9   |                     1.062 |           0.444 |     1947 |                    0.224 |
|           6 |                        3 |                0.852 |                     3.438 |           0.596 |     1947 |                    0.418 |
|           6 |                        7 |                0.746 |                     9.49  |           0.737 |     1947 |                    0.613 |
|          12 |                        1 |                0.932 |                     1.562 |           0.624 |     1928 |                    0.564 |
|          12 |                        3 |                0.839 |                     4.896 |           0.814 |     1928 |                    0.76  |
|          12 |                        7 |                0.626 |                    15.854 |           0.939 |     1928 |                    0.96  |
|          24 |                        1 |                0.922 |                     6.562 |           0.873 |     1879 |                    0.707 |
|          24 |                        3 |                0.682 |                    17.219 |           0.969 |     1879 |                    0.976 |
|          24 |                        7 |                0.498 |                    21.375 |           0.977 |     1879 |                    0.976 |

### Per location, 3 h horizon, test (skill vs seasonal baseline; n/a if < 20 onset hours)

| location                |   volunteer: onset hours |   volunteer: skill |   official: onset hours |   official: skill |
|:------------------------|-------------------------:|-------------------:|------------------------:|------------------:|
| Cherkaska oblast        |                      363 |              0.06  |                     225 |             0.053 |
| Chernihivska oblast     |                      356 |             -0.038 |                     343 |             0.146 |
| Chernivetska oblast     |                       79 |              0.096 |                       6 |           nan     |
| Dnipropetrovska oblast  |                      635 |              0.03  |                     324 |             0.037 |
| Donetska oblast         |                      268 |              0.014 |                     134 |             0.07  |
| Ivano-Frankivska oblast |                       24 |              0.131 |                       0 |           nan     |
| Kharkivska oblast       |                      620 |              0.007 |                     400 |             0.24  |
| Khersonska oblast       |                      277 |              0.039 |                     333 |             0.081 |
| Khmelnytska oblast      |                      155 |              0.084 |                      44 |             0.057 |
| Kirovohradska oblast    |                      447 |              0.067 |                     266 |             0.145 |
| Kyiv City               |                      346 |             -0.025 |                     216 |             0.076 |
| Kyivska oblast          |                      566 |              0.068 |                     165 |             0.089 |
| Lvivska oblast          |                      261 |              0.08  |                       0 |           nan     |
| Mykolaivska oblast      |                      586 |              0.044 |                     436 |             0.123 |
| Odeska oblast           |                      473 |              0.044 |                     364 |             0.097 |
| Poltavska oblast        |                      451 |              0.048 |                     348 |             0.166 |
| Rivnenska oblast        |                      223 |              0.064 |                      38 |             0.061 |
| Sumska oblast           |                      459 |             -0.008 |                     321 |             0.21  |
| Ternopilska oblast      |                       50 |              0.066 |                      21 |             0.022 |
| Vinnytska oblast        |                      294 |              0.073 |                     102 |             0.092 |
| Volynska oblast         |                      284 |              0.062 |                      19 |           nan     |
| Zakarpatska oblast      |                        0 |            nan     |                       0 |           nan     |
| Zaporizka oblast        |                      373 |             -0.044 |                     236 |             0.128 |
| Zhytomyrska oblast      |                      371 |              0.082 |                      96 |             0.081 |