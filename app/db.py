"""SQLite persistence.

A version bundles, atomically: the per-factory corrections, the fitted
model and the per-factory LOFO errors.  The `meta.current_version`
pointer is flipped as the last statement of the same transaction, so
readers (predictions) always see one consistent committed version --
including while a recomputation is still in progress.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    id          INTEGER PRIMARY KEY,
    lambda      REAL NOT NULL,
    scores_json TEXT NOT NULL,
    model_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS factory_stats (
    version_id  INTEGER NOT NULL REFERENCES versions(id),
    factory_id  TEXT    NOT NULL,
    n           INTEGER NOT NULL,
    sums_json   TEXT    NOT NULL,   -- [sum speed, sum pressure, sum power]
    outer_json  TEXT    NOT NULL,   -- 3x3 outer-product sums
    correction  TEXT    NOT NULL,   -- integer zero-point correction
    PRIMARY KEY (version_id, factory_id)
);
CREATE TABLE IF NOT EXISTS factory_errors (
    version_id  INTEGER NOT NULL REFERENCES versions(id),
    factory_id  TEXT    NOT NULL,
    mse         REAL    NOT NULL,
    PRIMARY KEY (version_id, factory_id)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Conflict(Exception):
    """The version an update was based on is no longer current."""


def _jints(values):
    """Serialise (possibly huge) integers as strings, losslessly."""
    return json.dumps([str(v) for v in values])


def _jint_matrix(rows):
    return json.dumps([[str(v) for v in row] for row in rows])


def _parse_ints(text):
    return [int(v) for v in json.loads(text)]


def _parse_matrix(text):
    return [[int(v) for v in row] for row in json.loads(text)]


class Store:
    def __init__(self, path: str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self):
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------- reads
    def current_version_id(self):
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='current_version'"
            ).fetchone()
            return int(row["value"]) if row else None

    def get_version(self, version_id: int):
        with self._lock:
            v = self._conn.execute(
                "SELECT * FROM versions WHERE id=?", (version_id,)
            ).fetchone()
            if v is None:
                return None
            stats = self._conn.execute(
                "SELECT * FROM factory_stats WHERE version_id=? ORDER BY factory_id",
                (version_id,),
            ).fetchall()
            errors = self._conn.execute(
                "SELECT factory_id, mse FROM factory_errors WHERE version_id=?",
                (version_id,),
            ).fetchall()
        return self._assemble(v, stats, errors)

    def get_current(self):
        vid = self.current_version_id()
        return None if vid is None else self.get_version(vid)

    @staticmethod
    def _assemble(v, stats, errors):
        return {
            "version": v["id"],
            "lambda": v["lambda"],
            "lambda_scores": {float(k): val for k, val in
                              json.loads(v["scores_json"]).items()},
            "model": json.loads(v["model_json"]),
            "created_at": v["created_at"],
            "factories": [
                {
                    "factory_id": s["factory_id"],
                    "n": s["n"],
                    "sums": _parse_ints(s["sums_json"]),
                    "outer": _parse_matrix(s["outer_json"]),
                    "correction": int(s["correction"]),
                }
                for s in stats
            ],
            "factory_errors": {e["factory_id"]: e["mse"] for e in errors},
        }

    # ---------------------------------------------------------- writes
    def create_version(self, factories, result) -> int:
        """Store a freshly trained version and make it current, atomically."""
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                new_id = self._insert_version(cur, factories, result)
                self._set_current(cur, new_id)
                self._conn.commit()
                return new_id
            except Exception:
                self._conn.rollback()
                raise

    def replace_correction(self, base_version_id: int, factories,
                           result) -> int:
        """Store a new version (one factory's correction replaced).

        `factories` is the full factory list of the base version with the
        new correction already applied.  Raises Conflict if base_version_id
        is not the current version; on any failure the current version is
        left untouched.
        """
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                row = cur.execute(
                    "SELECT value FROM meta WHERE key='current_version'"
                ).fetchone()
                current = int(row["value"]) if row else None
                if current != base_version_id:
                    raise Conflict(
                        f"version {base_version_id} is not current "
                        f"(current: {current})"
                    )
                new_id = self._insert_version(cur, factories, result)
                self._set_current(cur, new_id)
                self._conn.commit()
                return new_id
            except Exception:
                self._conn.rollback()
                raise

    # ---------------------------------------------------------- helpers
    def _insert_version(self, cur, factories, result) -> int:
        row = cur.execute("SELECT COALESCE(MAX(id), 0) + 1 AS nid FROM versions").fetchone()
        new_id = int(row["nid"])
        cur.execute(
            "INSERT INTO versions (id, lambda, scores_json, model_json, created_at)"
            " VALUES (?,?,?,?,?)",
            (
                new_id,
                result.lam,
                json.dumps({str(k): v for k, v in result.lambda_scores.items()}),
                json.dumps(result.model.to_dict()),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        for f in factories:
            cur.execute(
                "INSERT INTO factory_stats"
                " (version_id, factory_id, n, sums_json, outer_json, correction)"
                " VALUES (?,?,?,?,?,?)",
                (
                    new_id,
                    f["factory_id"],
                    f["n"],
                    _jints(f["sums"]),
                    _jint_matrix(f["outer"]),
                    str(int(f["correction"])),
                ),
            )
        for fid, mse in zip([f["factory_id"] for f in factories],
                            result.factory_errors):
            cur.execute(
                "INSERT INTO factory_errors (version_id, factory_id, mse)"
                " VALUES (?,?,?)",
                (new_id, fid, float(mse)),
            )
        return new_id

    def _set_current(self, cur, version_id: int):
        cur.execute(
            "INSERT INTO meta (key, value) VALUES ('current_version', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(version_id),),
        )
