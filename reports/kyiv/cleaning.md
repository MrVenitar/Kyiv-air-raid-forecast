# Kyiv alert labels: cleaning report

KMDA posts used: 5371 ({'start': 2717, 'end': 2654}), first 2022-02-25 12:40:38 UTC, last 2026-10-12 09:56:32 UTC.
Episodes: 2653; anomalies: {'repeated_start_posts': 47, 'orphan_all_clears': 1, 'level_changes': 17}.
Still running at the end of the data: 0.

## Durations (minutes) by year

|   year |   count |   mean |   std |   min |   25% |   50% |   75% |   95% |    max |
|-------:|--------:|-------:|------:|------:|------:|------:|------:|------:|-------:|
|   2022 |     604 |   73.5 | 144.7 |   1.2 |  28.6 |  51.9 |  88.4 | 181.4 | 3294.3 |
|   2023 |     300 |   79.5 |  68.8 |   0.8 |  27.1 |  55.1 | 112.8 | 217.5 |  411.8 |
|   2024 |     501 |   64.8 |  80   |   0.8 |  22.9 |  34   |  73.1 | 207.4 |  596.9 |
|   2025 |     511 |   86.8 | 116   |   1   |  22.4 |  34.5 |  88.5 | 362.4 |  631.4 |
|   2026 |     737 |   58.7 |  80.7 |   0.6 |  17.3 |  31.4 |  63.5 | 210.2 |  764.1 |

## KMDA vs official channel (Kyiv City), where both exist

The official dataset stores most Kyiv alerts twice (4141 rows -> 2267 merged episodes).

| period                  |   KMDA episodes |   official episodes | KMDA starts with a official start ±5 min   | official starts with a KMDA start ±5 min   |   median start offset KMDA−official (min) | KMDA all-clears with a official end ±5 min   | median duration KMDA / official (min)   |
|:------------------------|----------------:|--------------------:|:-------------------------------------------|:-------------------------------------------|------------------------------------------:|:---------------------------------------------|:----------------------------------------|
| 2022-03-15 – 2023-01-01 |             510 |                 505 | 99.0%                                      | 100.0%                                     |                                     -0.05 | 98.4%                                        | 49 / 49                                 |
| 2023-01-01 – 2024-01-01 |             300 |                 301 | 100.0%                                     | 99.7%                                      |                                     -0.02 | 99.7%                                        | 55 / 55                                 |
| 2024-01-01 – 2025-01-01 |             501 |                 502 | 100.0%                                     | 99.8%                                      |                                     -0.02 | 100.0%                                       | 34 / 34                                 |
| 2025-01-01 – 2026-01-01 |             511 |                 511 | 100.0%                                     | 100.0%                                     |                                     -0.02 | 99.8%                                        | 34 / 35                                 |
| 2026-01-01 – 2026-09-06 |             494 |                 448 | 90.1%                                      | 99.3%                                      |                                      0    | 90.3%                                        | 30 / 30                                 |

## KMDA vs volunteer (eTryvoga) Kyiv City

| period                  |   KMDA episodes |   volunteer episodes | KMDA starts with a volunteer start ±5 min   | volunteer starts with a KMDA start ±5 min   |   median start offset KMDA−volunteer (min) | KMDA all-clears with a volunteer end ±5 min   | median duration KMDA / volunteer (min)   |
|:------------------------|----------------:|---------------------:|:--------------------------------------------|:--------------------------------------------|-------------------------------------------:|:----------------------------------------------|:-----------------------------------------|
| 2026-06-01 – 2026-09-06 |             264 |                  255 | 50.8%                                       | 52.2%                                       |                                      -0.21 | 79.2%                                         | 30 / 23                                  |
| 2026-09-06 – 2026-10-12 |             242 |                  157 | 25.6%                                       | 39.5%                                       |                                      -0.1  | 44.6%                                         | 36 / 30                                  |

Volunteer Kyiv alerts since the switch: 157, of which 29% have a guessed end (start + 30 min).

## The two-level system (since 06:00 Kyiv, 6 Sep 2026)

Episodes: 243 in 36 days (single-level system in Jan–Aug 2026: 494 in 248 days).
Level at start / highest level / path:

| level_start   |   starts at |   highest |
|:--------------|------------:|----------:|
| yellow        |         224 |       218 |
| red           |          17 |        23 |
| single        |           2 |         2 |

Most common level paths:

| level_path        |   episodes |
|:------------------|-----------:|
| yellow            |        218 |
| red               |         13 |
| yellow>red>yellow |          4 |
| red>yellow        |          4 |
| yellow>red        |          2 |
| single>red>yellow |          1 |
| single>red        |          1 |

Duration by highest level (minutes):

| level_max   |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| red         |      23 |  123   | 190.4 |   8.9 |  23.6 |  46.6 | 127.8 | 764.1 |
| single      |       2 |   29.6 |  26.1 |  11.2 |  20.4 |  29.6 |  38.8 |  48   |
| yellow      |     218 |   64.5 |  81.4 |   1.5 |  14.5 |  34   |  76.5 | 550.4 |