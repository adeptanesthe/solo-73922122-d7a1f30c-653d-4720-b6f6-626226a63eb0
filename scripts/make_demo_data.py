#!/usr/bin/env python3
"""Generate deterministic demo payloads (demo/*.json).

Scenario: four factories report integer summaries of raw transducer
counts (speed ~1e6, pressure ~5e5 -- large base values, small variance).
True physical model:

    power = 0.020 * speed + 0.011 * (pressure + zero_correction) + 1500 + noise

F3's pressure sensor zero-point drifted: its true correction is +500
counts but it was onboarded with 0.  The demo corrects it via the
versioned correction endpoint.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "demo")

A, B, INTERCEPT, NOISE = 0.020, 0.011, 1500.0, 1.0

# factory: (id, n, speed_mu, pressure_mu, true_correction, reported_correction)
FACTORIES = [
    ("F1", 1440, 1_000_000, 500_000, 0, 0),
    ("F2", 1200, 1_000_030, 500_008, -40, -40),
    ("F3", 1500, 1_000_060, 500_016, 500, 0),   # <- zero-point drift, to be fixed
    ("F4", 1320, 999_970, 499_992, 10, 10),
]


def simulate(fid, n, s_mu, p_mu, corr_true, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(s_mu, 25)))
        p = int(round(rng.gauss(p_mu, 12)))
        y = A * s + B * (p + corr_true) + INTERCEPT + rng.gauss(0, NOISE)
        rows.append((s, p, int(round(y))))
    sums = [sum(r[i] for r in rows) for i in range(3)]
    outer = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    return {
        "factory_id": fid,
        "n": n,
        "sums": sums,
        "outer": outer,
    }, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    factories = []
    f3_rows = None
    for k, (fid, n, s_mu, p_mu, c_true, c_reported) in enumerate(FACTORIES):
        body, rows = simulate(fid, n, s_mu, p_mu, c_true, seed=7000 + k)
        body["correction"] = c_reported
        factories.append(body)
        if fid == "F3":
            f3_rows = rows

    with open(os.path.join(OUT, "create_version.json"), "w") as fh:
        json.dump({"factories": factories}, fh, indent=1)

    with open(os.path.join(OUT, "correction_update.json"), "w") as fh:
        json.dump({"factory_id": "F3", "correction": 500}, fh, indent=1)

    s, p, _ = f3_rows[0]
    with open(os.path.join(OUT, "predict_f3.json"), "w") as fh:
        json.dump({"factory_id": "F3", "speed": s, "pressure": p}, fh, indent=1)

    print(f"wrote demo payloads to {os.path.abspath(OUT)}")
    print(f"F3 sample reading: speed={s} pressure={p} (true correction +500)")


if __name__ == "__main__":
    main()
