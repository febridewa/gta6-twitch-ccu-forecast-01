# GTA VI launch-day Twitch CCU forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/febridewa/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/febridewa/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 2026-11-19 (UTC), from everything recorded up to the
end of the day's first hour — as five quantiles an hour (5%, 25%, 50%, 75%,
95%). One program, `main.py` (Python 3.13, standard library only), reads
`/input/<id>.in` and writes `/output/<id>.out`.

Play along or enter your own forecast at
[make.fun/bounty/gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
— and if you sign up through my referral link,
[make.fun/r/6etrfz79](https://make.fun/r/6etrfz79), we both get a head start.

## Method

The reference class is 38–42 past big game launches' hourly Twitch curves,
shipped inside each case as `comparables`. Three estimators are fitted at run
time from those curves and blended by geometric mean:

1. **DST-aligned log-log OLS on rolling-midnight launches.** Launches are
   aligned by the hour their US-East audience unlocks (04:00Z in daylight
   time, 05:00Z in standard time — the morning step lands on a different UTC
   hour depending on the date; GTA VI's US-East unlock is 05:00Z, so its
   first US hour is hour 6). Training set: launches whose hour-1 print is at
   least 8% of their day's peak, whose peak lands in the evening (hour ≥ 16),
   and which had no early access.
2. **Plain per-hour log-log OLS on all usable comparables** — log viewers(h)
   on log viewers(1). This keeps the very biggest launches (Elden Ring,
   Cyberpunk 2077: simultaneous global unlocks, hour-1 prints near 900k) as
   anchors for the scale the final case sits at. Its training set is
   windowed to launches within 20× of the target's hour-1 print, so the
   simultaneous-unlock minnows (hour-1 prints in the hundreds, growing
   100–1000× by evening) cannot drag its slope down.
3. **Huber-robust per-hour log-log OLS** on the same rolling set as (1),
   down-weighting launches whose curve is an outlier for that hour.

Each estimator's spread is the empirical quantiles of its own residuals,
shrunk by 0.7. The target's log hour-1 print is winsorised to the training
range, so a GTA-VI-sized print does not extrapolate the log-linear slope into
nonsense (the biggest launches' evening ratios sit near 0.9–1.3× their hour-1
print). Weights are 1 + 0.08 × (release year − 2018): recent launches count
more. Blend weights: 0.5 / 0.25 / 0.25. Any hour an estimator cannot fit
falls back to the other estimators, then to the house persistence shape, so
the program always writes a valid file. Deterministic: pure arithmetic, no
randomness, no network.

## Validation (local, against the practice answers)

Practice answers were re-read from the kit's documented source
(SullyGnome.com's hourly averages, the same source the house used; all four
hour-1 prints reproduced exactly) so the model could be scored locally
before submission:

| check | result |
|---|---|
| Practice pinball loss (4 cases × 23 h × 5 levels) | **0.1554** (house baseline 0.3871) |
| Per case | p00 0.202, p01 0.146, p02 0.104, p03 0.170 |
| Leave-one-launch-out, GTA-VI-like launches (rolling, evening peak, no early access) | mean 0.335, median 0.223 (baseline 0.482 / 0.372) |
| Leave-one-launch-out, big launches (hour-1 ≥ 50k) | mean 0.233, median 0.187 (baseline 0.252 / 0.247) |
| Output format | 23 rows, hours 2–24, quantiles non-decreasing, ≤ 65,536 bytes |
| Determinism | two runs byte-identical |
| Runtime | ~0.05 s per case (limit 8 s) |

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
