
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