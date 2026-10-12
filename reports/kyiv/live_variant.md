# Live variant: the chosen models without regional alert features

Regional alert history (Kyiv oblast, neighbours, country) is published once a day, so a forecast for the
current minute cannot use it. Same model type, recency weighting and recalibration as the frozen choice.


## onset

| period                     | model                         |   mean log-loss |   baseline log-loss |   skill 15 min |   skill 30 min |   skill 60 min |   skill 120 min |   skill 180 min |
|:---------------------------|:------------------------------|----------------:|--------------------:|---------------:|---------------:|---------------:|----------------:|----------------:|
| validation 24 Aug – 20 Sep | chosen (with regional alerts) |           0.483 |               0.602 |          0.125 |          0.168 |          0.2   |           0.236 |           0.241 |
| validation 24 Aug – 20 Sep | live (no regional alerts)     |           0.488 |               0.602 |          0.12  |          0.152 |          0.185 |           0.227 |           0.235 |
| test 21 Sep – 11 Oct       | chosen (with regional alerts) |           0.564 |               0.597 |          0.078 |          0.086 |          0.061 |           0.066 |           0.093 |
| test 21 Sep – 11 Oct       | live (no regional alerts)     |           0.569 |               0.597 |          0.065 |          0.062 |          0.044 |           0.063 |           0.096 |

## clear

| period                     | model                         |   mean log-loss |   baseline log-loss |   skill 15 min |   skill 30 min |   skill 60 min |   skill 120 min |   skill 180 min |
|:---------------------------|:------------------------------|----------------:|--------------------:|---------------:|---------------:|---------------:|----------------:|----------------:|
| validation 24 Aug – 20 Sep | chosen (with regional alerts) |           0.545 |               0.565 |          0.172 |          0.149 |          0.085 |           0.034 |          -0.001 |
| validation 24 Aug – 20 Sep | live (no regional alerts)     |           0.592 |               0.565 |         -0.013 |         -0.011 |         -0.034 |           0.002 |          -0.003 |
| test 21 Sep – 11 Oct       | chosen (with regional alerts) |           0.503 |               0.564 |          0.136 |          0.182 |          0.169 |           0.05  |           0.059 |
| test 21 Sep – 11 Oct       | live (no regional alerts)     |           0.585 |               0.564 |          0.006 |         -0.005 |         -0.046 |          -0.101 |          -0.042 |

Without regional alerts the clear model loses to the baseline on validation, so the live clear forecast is the baseline.
