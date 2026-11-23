"""End-to-end API tests: versioning, correction update, conflicts, predict."""
import random

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def make_rows(n, speed_mu, press_mu, coef, intercept, noise, seed,
              press_shift=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(speed_mu, 25)))
        p = int(round(rng.gauss(press_mu, 12)))
        y = coef[0] * s + coef[1] * (p + press_shift) + intercept + rng.gauss(0, noise)
        rows.append((s, p, int(round(y))))
    return rows


def payload(factory_id, rows, correction=0):
    n = len(rows)
    return {
        "factory_id": factory_id,
        "n": n,
        "sums": [sum(r[i] for r in rows) for i in range(3)],
        "outer": [[sum(r[i] * r[j] for r in rows) for j in range(3)]
                  for i in range(3)],
        "correction": correction,
    }


def demo_factories():
    rows = [
        make_rows(400, 1_000_000 + 30 * i, 500_000 + 8 * i, (0.02, 0.011),
                  1500, 4, seed=200 + i)
        for i in range(4)
    ]
    # F3's pressure sensor is really off by +25 but reports 0
    rows[2] = make_rows(400, 1_000_060, 500_016, (0.02, 0.011), 1500, 4,
                        seed=999, press_shift=25)
    return [payload(f"F{i+1}", r) for i, r in enumerate(rows)]


@pytest.fixture()
def client():
    app = create_app(":memory:")
    with TestClient(app) as c:
        yield c


def create(client, factories=None):
    r = client.post("/versions", json={"factories": factories or demo_factories()})
    assert r.status_code == 201, r.text
    return r.json()


def test_create_and_query_current(client):
    body = create(client)
    assert body["version"] == 1
    assert body["lambda"] in (0.01, 0.1, 1.0)
    assert set(body["factory_errors"]) == {"F1", "F2", "F3", "F4"}
    assert set(body["lambda_scores"]) == {"0.01", "0.1", "1.0"}
    cur = client.get("/versions/current").json()
    assert cur == body
    by_id = client.get("/versions/1").json()
    assert by_id == body


def test_predict_uses_current_version(client):
    create(client)
    r = client.post("/predict", json={"factory_id": "F1", "speed": 1_000_030,
                                      "pressure": 500_008})
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert isinstance(body["power"], float)


def test_correction_update_atomic_and_versioned(client):
    before = create(client)
    err_before = before["factory_errors"]["F3"]
    pred_before = client.post("/predict", json={
        "factory_id": "F3", "speed": 1_000_060, "pressure": 500_016
    }).json()

    r = client.post("/versions/1/correction",
                    json={"factory_id": "F3", "correction": 25})
    assert r.status_code == 200, r.text
    after = r.json()
    assert after["version"] == 2
    assert after["corrections"]["F3"] == 25
    # errors + model changed together with the correction
    assert after["factory_errors"]["F3"] < err_before
    assert after["model"] != before["model"]

    cur = client.get("/versions/current").json()
    assert cur["version"] == 2
    assert cur["factory_errors"] == after["factory_errors"]

    pred_after = client.post("/predict", json={
        "factory_id": "F3", "speed": 1_000_060, "pressure": 500_016
    }).json()
    assert pred_after["version"] == 2
    assert pred_after["power"] != pred_before["power"]

    # old version is immutable history
    assert client.get("/versions/1").json()["corrections"]["F3"] == 0


def test_conflict_keeps_current_version(client):
    create(client)
    client.post("/versions/1/correction",
                json={"factory_id": "F3", "correction": 25})
    # replay against stale version 1
    r = client.post("/versions/1/correction",
                    json={"factory_id": "F3", "correction": 99})
    assert r.status_code == 409
    cur = client.get("/versions/current").json()
    assert cur["version"] == 2
    assert cur["corrections"]["F3"] == 25


def test_update_unknown_factory(client):
    create(client)
    r = client.post("/versions/1/correction",
                    json={"factory_id": "NOPE", "correction": 1})
    assert r.status_code == 404
    assert client.get("/versions/current").json()["version"] == 1


def test_update_unknown_version(client):
    create(client)
    r = client.post("/versions/99/correction",
                    json={"factory_id": "F1", "correction": 1})
    assert r.status_code == 404


def test_predict_unknown_factory(client):
    create(client)
    r = client.post("/predict", json={"factory_id": "NOPE", "speed": 1,
                                      "pressure": 2})
    assert r.status_code == 404


def test_predict_without_model(client):
    r = client.post("/predict", json={"factory_id": "F1", "speed": 1,
                                      "pressure": 2})
    assert r.status_code == 503


def test_predict_during_recompute_uses_current_version(client, monkeypatch):
    import threading

    import app.main as main_mod

    create(client)
    orig_train = main_mod.train
    started = threading.Event()
    release = threading.Event()

    def slow_train(stats):
        started.set()
        assert release.wait(10)
        return orig_train(stats)

    monkeypatch.setattr(main_mod, "train", slow_train)
    outcome = {}

    def do_update():
        r = client.post("/versions/1/correction",
                        json={"factory_id": "F3", "correction": 500})
        outcome["status"] = r.status_code

    t = threading.Thread(target=do_update)
    t.start()
    assert started.wait(10)
    # recomputation is blocked mid-flight: predictions keep using version 1
    for _ in range(5):
        r = client.post("/predict", json={"factory_id": "F3",
                                          "speed": 1_000_060,
                                          "pressure": 500_016})
        assert r.status_code == 200
        assert r.json()["version"] == 1
    assert client.get("/versions/current").json()["version"] == 1
    release.set()
    t.join(10)
    assert outcome["status"] == 200
    assert client.get("/versions/current").json()["version"] == 2


def test_factory_count_bounds(client):
    fs = demo_factories()
    assert client.post("/versions", json={"factories": fs[:2]}).status_code == 422
    too_many = [dict(f, factory_id=f"X{i}") for i, f in enumerate(fs * 3)]
    assert client.post("/versions",
                       json={"factories": too_many}).status_code == 422


def test_duplicate_factory_ids_rejected(client):
    fs = demo_factories()
    fs[1]["factory_id"] = fs[0]["factory_id"]
    assert client.post("/versions", json={"factories": fs}).status_code == 422


def test_inconsistent_stats_rejected(client):
    fs = demo_factories()
    fs[0]["outer"][0][0] = 1  # impossible: n*sum(s^2) < sum(s)^2
    r = client.post("/versions", json={"factories": fs})
    assert r.status_code == 422
    assert client.get("/versions/current").status_code == 404


def test_huge_base_values_variance_preserved(client):
    body = create(client)  # demo factories sit around 1e6 / 5e5
    assert body["model"]["std"]["0"] > 0
    assert body["model"]["std"]["1"] > 0
    # model explains the data well: worst LOFO error far below power variance
    assert max(body["factory_errors"].values()) < 1e4


def test_beyond_int64_values_keep_variance(client):
    # readings ~1e12: sums ~1e15, outer sums ~1e27 -- beyond float64's exact
    # integer range; variance of a few units must still survive.
    rng = random.Random(42)
    factories = []
    for i in range(3):
        rows = []
        for _ in range(60):
            s = 10**12 + 1000 * i + rng.randint(-4, 4)
            p = 10**12 + rng.randint(-3, 3)
            y = 3 * (s - 10**12) + 2 * (p - 10**12) + 5000 + rng.randint(-2, 2)
            rows.append((s, p, y))
        factories.append(payload(f"G{i}", rows))
    body = create(client, factories)
    assert body["model"]["std"]["0"] > 0
    assert body["model"]["std"]["1"] > 0
    errs = body["factory_errors"]
    # interpolating factory fits almost perfectly -> variance really usable
    assert errs["G1"] < 100.0
    # edge factories extrapolate in LOFO, but if variance had collapsed to
    # zero the speed feature would be dropped and errors would be ~1e6+
    assert max(errs.values()) < 1e5
