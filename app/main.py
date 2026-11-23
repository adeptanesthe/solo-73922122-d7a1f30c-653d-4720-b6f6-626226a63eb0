"""FastAPI service: shared fan power-consumption model across factories."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from . import db as dbmod
from .model import RidgeModel, train
from .schemas import (CorrectionUpdate, CreateVersionRequest, PredictRequest)
from .stats import InvalidStats, Stats, apply_correction

DB_PATH = os.environ.get("DB_PATH", os.path.join("data", "platform.db"))


def create_app(db_path: str | None = None) -> FastAPI:
    path = db_path if db_path is not None else DB_PATH
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    store = dbmod.Store(path)

    app = FastAPI(title="Fan Power Model Platform")
    app.state.store = store

    # ------------------------------------------------------------- helpers
    def _train_from_dicts(factories):
        """Apply each factory's correction and fit; may raise InvalidStats."""
        stats = [
            apply_correction(
                Stats.build(f["n"], f["sums"], f["outer"]), f["correction"]
            )
            for f in factories
        ]
        return train(stats)

    def _public(view: dict) -> dict:
        return {
            "version": view["version"],
            "lambda": view["lambda"],
            "lambda_scores": view["lambda_scores"],
            "factory_errors": view["factory_errors"],
            "corrections": {f["factory_id"]: f["correction"]
                            for f in view["factories"]},
            "model": view["model"],
            "created_at": view["created_at"],
        }

    # ------------------------------------------------------------- routes
    @app.get("/health")
    def health():
        return {"status": "ok", "current_version": store.current_version_id()}

    @app.post("/versions", status_code=201)
    def create_version(req: CreateVersionRequest):
        """Onboard 3-8 factories and train a fresh model version."""
        try:
            result = _train_from_dicts([f.model_dump() for f in req.factories])
        except InvalidStats as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        factories = [f.model_dump() for f in req.factories]
        new_id = store.create_version(factories, result)
        return _public(store.get_version(new_id))

    @app.get("/versions/current")
    def get_current():
        view = store.get_current()
        if view is None:
            raise HTTPException(status_code=404, detail="no version yet")
        return _public(view)

    @app.get("/versions/{version_id}")
    def get_version(version_id: int):
        view = store.get_version(version_id)
        if view is None:
            raise HTTPException(status_code=404,
                                detail=f"version {version_id} not found")
        return _public(view)

    @app.post("/versions/{version_id}/correction")
    def update_correction(version_id: int, req: CorrectionUpdate):
        """Replace one factory's zero-point correction, based on a version.

        The correction, the retrained model and the per-factory errors take
        effect together as a new current version.  If the base version is
        not current (conflict) or recomputation fails, the current version
        is kept.
        """
        base = store.get_version(version_id)
        if base is None:
            raise HTTPException(status_code=404,
                                detail=f"version {version_id} not found")
        current = store.current_version_id()
        if current != version_id:
            raise HTTPException(
                status_code=409,
                detail=f"version conflict: based on {version_id}, "
                       f"current is {current}; current version kept",
            )
        factories = []
        found = False
        for f in base["factories"]:
            f = dict(f)
            if f["factory_id"] == req.factory_id:
                f["correction"] = req.correction
                found = True
            factories.append(f)
        if not found:
            raise HTTPException(
                status_code=404,
                detail=f"factory {req.factory_id!r} not in version {version_id}",
            )
        try:
            result = _train_from_dicts(factories)
        except InvalidStats as exc:
            # Recomputation failed: current version stays in effect.
            raise HTTPException(status_code=422,
                                detail=f"recomputation failed: {exc}")
        try:
            new_id = store.replace_correction(version_id, factories, result)
        except dbmod.Conflict as exc:
            raise HTTPException(status_code=409,
                                detail=f"{exc}; current version kept")
        return _public(store.get_version(new_id))

    @app.post("/predict")
    def predict(req: PredictRequest):
        """Predict per-minute power from raw speed/pressure readings.

        Always served from the committed current version (also while a
        recomputation is in flight); the version used is returned.
        """
        view = store.get_current()
        if view is None:
            raise HTTPException(status_code=503, detail="no model version yet")
        corr = {f["factory_id"]: f["correction"] for f in view["factories"]}
        if req.factory_id not in corr:
            raise HTTPException(
                status_code=404,
                detail=f"unknown factory {req.factory_id!r}",
            )
        model = RidgeModel.from_dict(view["model"])
        power = model.predict(req.speed, req.pressure + corr[req.factory_id])
        return {
            "factory_id": req.factory_id,
            "power": power,
            "version": view["version"],
        }

    return app


# Run with:  uvicorn --factory app.main:create_app
# (factory pattern keeps `import app.main` free of side effects)
