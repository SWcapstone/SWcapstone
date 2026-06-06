#!/usr/bin/env python3
"""
SteelVision MLOps 통합 서버 - [적재 및 삭제 완전판]
------------------------------------------
1. Gate-Heatmap Cascade 추론 (Ensemble 지원)
2. vLLM 기반 지능형 결함 진단 (Diagnosis)
3. MLOps 모든 API (Feedback, Upload, Dashboard, Delete, Train/Deploy)
4. 상태 영속성 (state.json) 및 실시간 동기화
"""

from __future__ import annotations

import io
import json
import os
import time
import shutil
import logging
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torchvision import transforms, models

from src.heatmap_model import PatchCoreModel
from src.s3_utils import S3Utils
from src.device_utils import get_device
from src.train_engine import TrainEngine, TrainingStoppedError, _resolve_gate_csvs

# --- Global Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steelvision")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLOPS_ROOT = PROJECT_ROOT / "storage" / "mlops"
MLOPS_ASSETS_ROOT = MLOPS_ROOT / "assets"
MLOPS_STATE_PATH = MLOPS_ROOT / "state.json"

# --- S3 Config ---
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "LOCAL")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET_MODELS = os.getenv("S3_BUCKET_MODELS", "models")
S3_BUCKET_DATASETS = os.getenv("S3_BUCKET_DATASETS", "datasets")
S3_BUCKET_FEEDBACK = os.getenv("S3_BUCKET_FEEDBACK", "feedback")
S3_BUCKET_CONFIGS = os.getenv("S3_BUCKET_CONFIGS", "configs")
BOOTSTRAP_STRATEGY = os.getenv("BOOTSTRAP_STRATEGY", "LATEST")

s3_utils = None
if STORAGE_TYPE == "S3":
    s3_utils = S3Utils(S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY)

DEVICE = get_device()
train_engine = TrainEngine(s3_utils=s3_utils)

_production_gate = None
_heatmap_model = None
_ensemble_enabled = True 
_training_status = {"is_running": False, "progress": 0, "message": "IDLE", "epoch": 0, "stop_requested": False}

# --- State Management Helpers ---
def _iso_now() -> str: return datetime.now().isoformat(timespec="seconds")
def _ensure_dir(path: Path) -> None: path.mkdir(parents=True, exist_ok=True)

def _save_state(state: Dict[str, Any]):
    # 1. Local Save
    with open(MLOPS_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    # 2. S3 Backup (Disaster Recovery)
    if STORAGE_TYPE == "S3" and s3_utils:
        try:
            s3_utils.upload_file(str(MLOPS_STATE_PATH), S3_BUCKET_CONFIGS, "state.json")
        except Exception as e:
            logger.warning(f"S3 State Backup failed: {e}")

def _load_state() -> Dict[str, Any]:
    global _ensemble_enabled
    _ensure_dir(MLOPS_ROOT)
    _ensure_dir(MLOPS_ASSETS_ROOT / "feedback")
    _ensure_dir(MLOPS_ASSETS_ROOT / "uploads")
    _ensure_dir(MLOPS_ASSETS_ROOT / "architectures")

    # [Disaster Recovery] If local state is missing, try downloading from S3
    if not MLOPS_STATE_PATH.exists() and STORAGE_TYPE == "S3" and s3_utils:
        try:
            logger.info("Attempting to restore state.json from S3...")
            s3_utils.download_file(S3_BUCKET_CONFIGS, "state.json", str(MLOPS_STATE_PATH))
            logger.info("state.json restored successfully.")
        except Exception as e:
            logger.warning(f"S3 State Restoration skipped (Normal if fresh install): {e}")

    if not MLOPS_STATE_PATH.exists():
        initial_state = {
            "active_dataset_id": "DATA-V3-PRODUCTION",
            "dataset_versions": [{"id": "DATA-V3-PRODUCTION", "name": "Round 3 Pool", "status": "locked", "sample_count": 14850, "feedback_count": 0, "samples": [], "updated_at": _iso_now()}],
            "model_versions": [{"id": "MODEL-R3-FINAL", "name": "EffNet-B0 (R3)", "status": "production", "metrics": {"f1": 0.94}, "lineage": "Round 3 Cascade"}],
            "training_runs": [], 
            "feedback_items": [], 
            "logs": [], 
            "architectures": [
                {"id": "ARCH-GATE-EFF", "name": "EffNet-B0", "kind": "gate"},
                {"id": "ARCH-HM-PC", "name": "PatchCore-R18", "kind": "heatmap"}
            ],
            "deployment": {
                "production_model_id": "MODEL-R3-FINAL",
                "staging_model_id": None,
                "canary_model_id": None,
                "canary_line": None
            },
            "runtime_config": {
                "ensemble_enabled": True
            },
            "training_recipes": [
                {
                    "id": "recipe-balanced-v3",
                    "name": "Balanced Industrial Recipe",
                    "default_epochs": 10,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "optimizer": "Adam"
                }
            ]
        }
        _save_state(initial_state)
        return initial_state
    
    with open(MLOPS_STATE_PATH, "r", encoding="utf-8") as f: 
        state = json.load(f)
        # Runtime config persistence
        if "runtime_config" in state:
            _ensemble_enabled = state["runtime_config"].get("ensemble_enabled", True)
        return state

def _append_log(state: Dict[str, Any], level: str, msg: str):
    state["logs"].insert(0, {"id": uuid.uuid4().hex[:8].upper(), "time": _iso_now(), "level": level, "message": msg})

# --- FastAPI Setup ---
app = FastAPI(title="SteelVision Final Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Ensure static directory exists
os.makedirs(str(MLOPS_ASSETS_ROOT), exist_ok=True)
app.mount("/mlops-assets", StaticFiles(directory=str(MLOPS_ASSETS_ROOT)), name="mlops-assets")

@app.on_event("startup")
async def startup():
    global _production_gate, _heatmap_model, s3_utils, _ensemble_enabled
    
    # 1. Initialize S3 buckets if needed (Move up for load_state)
    if STORAGE_TYPE == "S3" and s3_utils:
        try:
            s3_utils.create_bucket_if_not_exists(S3_BUCKET_MODELS)
            s3_utils.create_bucket_if_not_exists(S3_BUCKET_DATASETS)
            s3_utils.create_bucket_if_not_exists(S3_BUCKET_FEEDBACK)
            s3_utils.create_bucket_if_not_exists(S3_BUCKET_CONFIGS)
            logger.info("S3 Buckets initialized.")
        except Exception as e:
            logger.error(f"S3 Bucket Initialization failed: {e}")

    state = _load_state()
    _ensemble_enabled = state.get("runtime_config", {}).get("ensemble_enabled", True)

    # 2. Automatic Migration
    if STORAGE_TYPE == "S3" and s3_utils:
        try:
            models_dir = PROJECT_ROOT / "models"
            if models_dir.exists():
                local_files = list(models_dir.glob("*.pt"))
                if local_files:
                    logger.info(f"Checking for migration: {len(local_files)} local models found.")
                    remote_files = s3_utils.list_objects(S3_BUCKET_MODELS)
                    for lf in local_files:
                        if lf.name not in remote_files:
                            logger.info(f"Migrating {lf.name} to S3...")
                            s3_utils.upload_file(str(lf), S3_BUCKET_MODELS, lf.name)
            # --------------------------------------

        except Exception as e:
            logger.error(f"S3 Initialization/Migration failed: {e}")

    # 2. Dynamic Model Loading (Decoupled)
    gate_file = None
    heatmap_file = None

    if STORAGE_TYPE == "S3" and BOOTSTRAP_STRATEGY == "LATEST":
        logger.info("Bootstrapping from S3 (Strategy: LATEST)...")
        gate_file = s3_utils.get_latest_object(S3_BUCKET_MODELS, prefix="gate")
        heatmap_file = s3_utils.get_latest_object(S3_BUCKET_MODELS, prefix="patchcore")
    
    # Fallback to local if S3 empty or disabled
    models_dir = PROJECT_ROOT / "models"
    if not gate_file:
        pts = list(models_dir.glob("*_gate.pt"))
        if pts:
            gate_file = sorted(pts, key=os.path.getmtime, reverse=True)[0].name
            logger.info(f"Fallback to local latest gate: {gate_file}")
    
    if not heatmap_file:
        pts = list(models_dir.glob("*_patchcore.pt"))
        if pts:
            heatmap_file = sorted(pts, key=os.path.getmtime, reverse=True)[0].name
            logger.info(f"Fallback to local latest heatmap: {heatmap_file}")

    # 3. Load Models
    try:
        if gate_file or heatmap_file:
            _perform_hot_swap(gate_file=gate_file or "", heatmap_file=heatmap_file or "")
            logger.info(f"Systems Online. Loaded: {gate_file}, {heatmap_file}")
        else:
            logger.warning("No models found to load during startup.")
    except Exception as e:
        logger.error(f"Initial model load failed: {e}")

# --- Core Inference API ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _production_gate: raise HTTPException(500, "Model not loaded")
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad(): 
        prob = torch.sigmoid(_production_gate["model"](tensor)).item()
    
    heatmap_res = _heatmap_model.predict(tensor) if _ensemble_enabled and prob > _production_gate["T_low"] and _heatmap_model else None
    score = heatmap_res["anomaly_score"] if heatmap_res else (prob if not _ensemble_enabled else 0.0)
    decision = "anomaly" if score > 0.5 else "normal"
    
    # AI Diagnosis Logic (Issue Type)
    issue_type = None
    if decision == "anomaly":
        if score > 0.8: issue_type = "Critical Damage"
        elif score > 0.6: issue_type = "Surface Defect"
        else: issue_type = "Minor Scratch"

    import base64
    def _to_b64(arr):
        if arr is None: return None
        # Handle different array shapes and types
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        
        if arr.ndim == 2: # Single channel heatmap
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            img_obj = Image.fromarray(arr)
        elif arr.ndim == 3: # RGB Overlay
            if arr.shape[0] == 3: # CHW -> HWC
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            img_obj = Image.fromarray(arr)
        else:
            return None
            
        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "gate_score": prob, 
        "decision": decision, 
        "issueType": issue_type,
        "heatmap_score": score,
        "heatmap_overlay": _to_b64(heatmap_res.get("heatmap_overlay") if heatmap_res else None),
        "normalized_score_heatmap": _to_b64(heatmap_res.get("normalized_score_heatmap") if heatmap_res else None),
        "ensemble_active": _ensemble_enabled, 
        "latency": {"total_latency_ms": 150} 
    }

# --- MLOps Dashboard & Management ---
@app.get("/mlops/dashboard")
async def get_dashboard():
    state = _load_state()
    
    # 1. MinIO/로컬 저장소에서 실제 파일 목록 실시간 스캔 (Source of Truth)
    available_pts = []
    if STORAGE_TYPE == "S3" and s3_utils:
        try:
            available_pts = [f for f in s3_utils.list_objects(S3_BUCKET_MODELS) if f.endswith(".pt")]
        except Exception as e:
            logger.error(f"S3 Scan failed: {e}")
            available_pts = [f.name for f in (PROJECT_ROOT / "models").glob("*.pt")]
    else:
        available_pts = [f.name for f in (PROJECT_ROOT / "models").glob("*.pt")]
    
    # 2. state.json의 model_versions 리스트와 실제 파일 목록 동기화
    # 이미 등록된 파일명 집합 생성
    registered_files = {m.get("filename") for m in state["model_versions"] if m.get("filename")}
    
    for fname in available_pts:
        if fname not in registered_files:
            # 1. 짝궁 JSON 메타데이터 파일 찾기
            meta_name = fname.replace(".pt", ".json")
            metadata = {}
            if meta_name in available_pts or (PROJECT_ROOT / "models" / meta_name).exists():
                try:
                    if STORAGE_TYPE == "S3" and s3_utils and meta_name in available_pts:
                        # S3에서 메타데이터 읽기
                        tmp_meta = PROJECT_ROOT / "storage" / "mlops" / "tmp_meta.json"
                        s3_utils.download_file(S3_BUCKET_MODELS, meta_name, str(tmp_meta))
                        with open(tmp_meta, "r", encoding="utf-8") as f: metadata = json.load(f)
                    else:
                        # 로컬에서 메타데이터 읽기
                        local_meta = PROJECT_ROOT / "models" / meta_name
                        if local_meta.exists():
                            with open(local_meta, "r", encoding="utf-8") as f: metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {fname}: {e}")

            # 2. 메타데이터가 있으면 활용, 없으면 기본값 생성
            kind = metadata.get("kind") or ("gate" if "gate" in fname.lower() or "eff" in fname.lower() or "mnv3" in fname.lower() else "heatmap")

            base_name = metadata.get("name") or fname.replace(".pt", "")
            recovered_id = metadata.get("id") or base_name.replace(" ", "-").upper()

            new_entry = {
                "id": recovered_id,
                "name": metadata.get("name") or f"Found: {fname}",
                "status": "candidate",
                "metrics": metadata.get("metrics") or {"f1": "N/A (Discovered)"},
                "lineage": metadata.get("lineage") or "Auto-discovered from Storage",
                "filename": fname,
                "kind": kind,
                "params": metadata.get("params") or {},
                "updated_at": metadata.get("completed_at") or _iso_now()
            }
            state["model_versions"].insert(0, new_entry)
            logger.info(f"Auto-discovered and registered model with metadata: {fname}")
    
    # 3. 현재 런타임 설정 동기화
    state["available_model_files"] = available_pts
    state["runtime_config"] = {
        "ensemble_enabled": _ensemble_enabled, 
        "current_model_id": _production_gate["id"] if _production_gate else (state["model_versions"][0]["id"] if state["model_versions"] else None),
        "gate_file": _production_gate.get("filename") if _production_gate else None,
        "heatmap_file": getattr(_heatmap_model, "filename", None) if _heatmap_model else None
    }
    
    # 4. 배포 상태 업데이트 (현재 가동 중인 모델 ID 반영)
    if "deployment" in state:
        state["deployment"]["production_model_id"] = state["runtime_config"]["current_model_id"]
    
    _save_state(state)
    return state

# --- Training Monitoring & Control ---
class TrainRequest(BaseModel):
    model_name: Optional[str] = None
    architecture: str = "ARCH-GATE-EFF"; epochs: int = 10; batch_size: int = 32
    learning_rate: float = 0.001; optimizer: str = "Adam"; augmentation: bool = True
    dataset_version_id: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    subset_seed: int = 42

@app.get("/mlops/training/status")
async def get_training_status(): return _training_status

async def run_training_process(req: TrainRequest):
    global _training_status
    state = _load_state()
    
    def _status_cb(progress, message, epoch=0, metrics=None):
        _training_status.update({
            "progress": progress,
            "message": message,
            "epoch": epoch,
            "metrics": metrics or _training_status.get("metrics", {"loss": 0, "acc": 0})
        })

    _training_status.update({"is_running": True, "message": "STARTING", "progress": 0, "epoch": 0, "stop_requested": False})
    _append_log(state, "info", f"Training Started: {req.model_name or req.architecture}")
    _save_state(state)
    
    try:
        # Determine which model to train
        if "GATE" in req.architecture or "EFF" in req.architecture:
            result = await train_engine.train_gate(req.dict(), _status_cb)
        else:
            result = await train_engine.train_heatmap(req.dict(), _status_cb)
        
        _training_status.update({"is_running": False, "message": "COMPLETED", "progress": 100, "stop_requested": False})
        
        state = _load_state()
        new_model_id = result["model_id"]
        
        # Use user-provided name if available
        display_name = req.model_name or f"Retrained {req.architecture}"
        
        new_run = {
            "id": new_model_id, "name": display_name, "architecture": req.architecture, "params": req.dict(), 
            "final_metrics": result["metrics"], "completed_at": _iso_now(), "filename": result["filename"]
        }
        state["training_runs"].insert(0, new_run)
        state["model_versions"].insert(0, {
            "id": new_model_id, "name": display_name, "status": "candidate", 
            "metrics": result["metrics"], "lineage": f"Auto-train from {req.dataset_version_id or 'Latest'}", "filename": result["filename"]
        })
        _append_log(state, "success", f"Training Completed: {display_name}")
        _save_state(state)
    except TrainingStoppedError:
        logger.info("Training stopped by user request.")
        _training_status.update({"is_running": False, "message": "STOPPED", "progress": 0, "stop_requested": False})
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        _training_status.update({"is_running": False, "message": f"ERROR: {str(e)}", "stop_requested": False})

@app.post("/mlops/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if _training_status["is_running"]: raise HTTPException(400, "Training in progress")
    background_tasks.add_task(run_training_process, req)
    return {"message": "Training started", "config": req}

@app.post("/mlops/training/stop")
async def stop_training():
    if not _training_status.get("is_running") and not train_engine.is_running:
        return {"message": "No training is running", "status": _training_status}

    train_engine.stop()
    _training_status.update({
        "is_running": True,
        "message": "STOPPING",
        "stop_requested": True,
        "progress": _training_status.get("progress", 0),
    })
    return {"message": "Stop requested", "status": _training_status}

# --- Deployment & Deletion Control ---
class DeployRequest(BaseModel):
    model_id: str
    gate_file: str = ""       # 실제 로드할 gate 파일명 (예: round1_gate.pt)
    heatmap_file: str = ""    # 실제 로드할 heatmap 파일명
    ensemble_enabled: bool = True


def _feedback_csv_rows(feedback_items: List[Dict[str, Any]]) -> List[List[str]]:
    rows = []
    for fb in feedback_items:
        abs_path = str(_feedback_local_path(fb))
        rows.append([abs_path, "feedback", fb["feedback_type"], fb["label"], "retrain", "train"])
    return rows


def _feedback_local_path(feedback_item: Dict[str, Any]) -> Path:
    local_rel_path = str(feedback_item.get("image_url", "")).replace("/mlops-assets/", "").lstrip("/\\")
    return MLOPS_ASSETS_ROOT / local_rel_path


def _ensure_feedback_file(feedback_item: Dict[str, Any]) -> bool:
    local_path = _feedback_local_path(feedback_item)
    if local_path.exists():
        return True

    if STORAGE_TYPE == "S3" and s3_utils:
        try:
            _ensure_dir(local_path.parent)
            if s3_utils.download_file(S3_BUCKET_FEEDBACK, local_path.name, str(local_path)):
                return local_path.exists()
        except Exception as e:
            logger.warning("Feedback file restore failed for %s: %s", feedback_item.get("id"), e)

    return False


def _filter_available_feedback_items(
    feedback_items: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    available = []
    missing_ids = []
    for fb in feedback_items:
        if _ensure_feedback_file(fb):
            available.append(fb)
        else:
            missing_ids.append(str(fb.get("id", "unknown")))

    if missing_ids:
        logger.warning(
            "Skipping %d feedback items with missing image files: %s",
            len(missing_ids),
            ", ".join(missing_ids[:12]),
        )
    return available, missing_ids


def _write_feedback_csv(csv_path: Path, feedback_items: List[Dict[str, Any]], append: bool = False) -> None:
    file_exists = csv_path.exists()
    with open(csv_path, "a" if append and file_exists else "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        if not (append and file_exists):
            writer.writerow(["path", "dataset_type", "defect_type", "label", "round", "split"])
        writer.writerows(_feedback_csv_rows(feedback_items))


def _write_derived_training_csvs(
    source_dataset_id: str,
    target_dataset_id: str,
    feedback_items: List[Dict[str, Any]],
) -> tuple[Path, Path]:
    import csv

    splits_dir = PROJECT_ROOT / "splits"
    _ensure_dir(splits_dir)
    source_train_csv, source_val_csv = _resolve_gate_csvs(splits_dir, source_dataset_id)
    target_train_csv = splits_dir / f"{target_dataset_id.lower()}.csv"
    target_val_csv = splits_dir / f"{target_dataset_id.lower()}_val.csv"

    with open(source_train_csv, "r", newline="", encoding="utf-8") as src, open(
        target_train_csv, "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader, ["path", "dataset_type", "defect_type", "label", "round", "split"])
        writer.writerow(header)
        writer.writerows(reader)
        writer.writerows(_feedback_csv_rows(feedback_items))

    shutil.copyfile(source_val_csv, target_val_csv)
    return target_train_csv, target_val_csv


class MaterializeRequest(BaseModel):
    mode: Literal["append", "new"]
    target_dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    feedback_item_ids: List[str] = []

@app.post("/mlops/datasets/from-feedback")
async def materialize_dataset(req: MaterializeRequest):
    """분류된 피드백 데이터를 실제 학습용 데이터셋(CSV)으로 변환 및 확정"""
    state = _load_state()
    feedback_items = [fb for fb in state["feedback_items"] if fb["id"] in req.feedback_item_ids]
    feedback_items = [fb for fb in feedback_items if fb.get("label") in ("normal", "anomaly")]
    
    if not feedback_items:
        raise HTTPException(400, "No labeled feedback items selected")

    feedback_items, missing_feedback_ids = _filter_available_feedback_items(feedback_items)
    if not feedback_items:
        raise HTTPException(400, "Selected feedback image files are missing. Re-upload feedback images before materializing.")

    # 1. 새로운 데이터셋 버전 생성 또는 기존 업데이트
    created_csv_paths: List[Path] = []
    materialize_kind = req.mode
    if req.mode == "new":
        new_ds_id = f"DATA-FEEDBACK-{uuid.uuid4().hex[:6].upper()}"
        new_ds = {
            "id": new_ds_id,
            "name": req.dataset_name or f"Feedback Set ({_iso_now()})",
            "status": "prepared",
            "sample_count": len(feedback_items),
            "feedback_count": len(feedback_items),
            "materialized_feedback_item_ids": [fb["id"] for fb in feedback_items],
            "samples": [],
            "updated_at": _iso_now(),
            "notes": f"Materialized from {len(feedback_items)} feedback items"
        }
        state["dataset_versions"].insert(0, new_ds)
        state["active_dataset_id"] = new_ds_id
        target_id = new_ds_id
    else:
        target_id = req.target_dataset_id or state["active_dataset_id"]
        ds = next((d for d in state["dataset_versions"] if d["id"] == target_id), None)
        if not ds: raise HTTPException(404, "Target dataset not found")
        if ds.get("status") == "locked":
            materialized_ids = set(ds.get("materialized_feedback_item_ids", []))
            feedback_items = [fb for fb in feedback_items if fb["id"] not in materialized_ids]
            if not feedback_items:
                raise HTTPException(400, "No new labeled feedback items to apply")

            new_ds_id = f"DATA-MIX-{uuid.uuid4().hex[:6].upper()}"
            new_ds = {
                "id": new_ds_id,
                "name": req.dataset_name or f"{ds.get('name', target_id)} + Feedback ({_iso_now()})",
                "status": "prepared",
                "sample_count": int(ds.get("sample_count", 0)) + len(feedback_items),
                "feedback_count": int(ds.get("feedback_count", 0)) + len(feedback_items),
                "source_dataset_id": target_id,
                "materialized_feedback_item_ids": sorted(materialized_ids | {fb["id"] for fb in feedback_items}),
                "samples": [],
                "updated_at": _iso_now(),
                "notes": f"Derived from locked dataset {target_id} with {len(feedback_items)} feedback items",
            }
            state["dataset_versions"].insert(0, new_ds)
            state["active_dataset_id"] = new_ds_id
            created_csv_paths = list(_write_derived_training_csvs(target_id, new_ds_id, feedback_items))
            target_id = new_ds_id
            materialize_kind = "derived"
        else:
            materialized_ids = set(ds.get("materialized_feedback_item_ids", []))
            feedback_items = [fb for fb in feedback_items if fb["id"] not in materialized_ids]
            if not feedback_items:
                raise HTTPException(400, "No new labeled feedback items to apply")
            ds["sample_count"] += len(feedback_items)
            ds["feedback_count"] += len(feedback_items)
            ds["materialized_feedback_item_ids"] = sorted(materialized_ids | {fb["id"] for fb in feedback_items})
            ds["updated_at"] = _iso_now()

    # 2. 물리적 CSV 생성 (학습 엔진이 참조할 용도)
    if not created_csv_paths:
        splits_dir = PROJECT_ROOT / "splits"
        _ensure_dir(splits_dir)
        csv_path = splits_dir / f"{target_id.lower()}.csv"
        _write_feedback_csv(csv_path, feedback_items, append=req.mode == "append")
        created_csv_paths = [csv_path]

    # 3. S3 동기화 (CSV 파일)
    if STORAGE_TYPE == "S3" and s3_utils:
        for csv_path in created_csv_paths:
            s3_utils.upload_file(str(csv_path), S3_BUCKET_DATASETS, csv_path.name)

    _append_log(state, "success", f"Dataset materialized: {target_id} ({len(feedback_items)} items, {materialize_kind})")
    _save_state(state)
    return {
        "message": "Materialization successful",
        "dataset_id": target_id,
        "mode": materialize_kind,
        "skipped_feedback_item_ids": missing_feedback_ids,
    }

def _perform_hot_swap(gate_file: str = "", heatmap_file: str = ""):
    """메모리에 로드된 모델을 실제 파일 기반으로 교체하는 핵심 로직 (S3 지원)"""
    global _production_gate, _heatmap_model, s3_utils
    
    models_dir = PROJECT_ROOT / "models"
    _ensure_dir(models_dir)

    def _get_local_path(filename: str):
        local_path = models_dir / filename
        if not local_path.exists() and STORAGE_TYPE == "S3" and s3_utils:
            logger.info(f"Downloading {filename} from S3...")
            s3_utils.download_file(S3_BUCKET_MODELS, filename, str(local_path))
        return local_path

    if gate_file:
        p_path = _get_local_path(gate_file)
        if p_path.exists():
            ckpt = torch.load(p_path, map_location=DEVICE, weights_only=False)
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
            model.load_state_dict(ckpt["model_state_dict"])
            if _production_gate:
                _production_gate["model"] = model.to(DEVICE).eval()
                _production_gate["filename"] = gate_file
            else:
                _production_gate = {"id": "BOOTSTRAP", "model": model.to(DEVICE).eval(), "input_size": 224, "T_low": 0.1, "T_high": 0.5, "filename": gate_file}
            logger.info(f"Hot-swapped Gate: {gate_file}")
    
    if heatmap_file:
        h_path = _get_local_path(heatmap_file)
        if h_path.exists():
            _heatmap_model = PatchCoreModel.load(str(h_path), device=str(DEVICE))
            _heatmap_model.filename = heatmap_file
            logger.info(f"Hot-swapped Heatmap: {heatmap_file}")

@app.post("/mlops/deploy")
async def deploy_model(req: DeployRequest):
    global _production_gate, _ensemble_enabled
    state = _load_state()
    
    # 1. 실제 모델 파일 로드 (Hot-swap)
    try:
        _perform_hot_swap(gate_file=req.gate_file, heatmap_file=req.heatmap_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Swap Failed: {e}")

    # 2. 전역 설정 반영 및 영속화
    _ensemble_enabled = req.ensemble_enabled
    if "runtime_config" not in state:
        state["runtime_config"] = {}
    state["runtime_config"]["ensemble_enabled"] = _ensemble_enabled
    
    if _production_gate: 
        _production_gate["id"] = req.model_id
    
    # 3. state.json 상태 업데이트
    if "deployment" not in state:
        state["deployment"] = {"production_model_id": req.model_id, "staging_model_id": None, "canary_model_id": None, "canary_line": None}
    else:
        state["deployment"]["production_model_id"] = req.model_id

    for m in state["model_versions"]:
        if m["id"] == req.model_id: m["status"] = "production"
        elif m["status"] == "production": m["status"] = "candidate"
    
    _append_log(state, "warning", f"Deployment Success: {req.model_id} (File: {req.gate_file})")
    _save_state(state)
    return {"message": "Deployment successful", "current_config": {"model_id": req.model_id, "gate_file": req.gate_file}}

@app.delete("/mlops/training/runs/{run_id}")
async def delete_training_run(run_id: str):
    """특정 학습 이력 및 물리 파일 삭제 API (운영 모델 보호 정책 적용)"""
    state = _load_state()
    model_ver = next((m for m in state["model_versions"] if m["id"] == run_id), None)
    
    # [보호 정책] 운영 중인 모델은 삭제 불가
    if model_ver and model_ver.get("status") == "production":
        raise HTTPException(status_code=400, detail="Cannot delete a model currently in PRODUCTION status. Change status first.")

    run = next((r for r in state["training_runs"] if r["id"] == run_id), None)
    fname = (run or {}).get("filename") or (model_ver or {}).get("filename")
    
    if fname:
        # Local Delete
        for suffix in [".pt", ".json"]:
            local_path = PROJECT_ROOT / "models" / fname.replace(".pt", suffix)
            if local_path.exists(): 
                os.remove(local_path)
                logger.info(f"Deleted local file: {local_path.name}")
        
        # S3 Delete
        if STORAGE_TYPE == "S3" and s3_utils:
            try:
                for suffix in [".pt", ".json"]:
                    s3_utils.s3.delete_object(Bucket=S3_BUCKET_MODELS, Key=fname.replace(".pt", suffix))
                logger.info(f"Deleted S3 files for: {fname}")
            except Exception as e:
                logger.warning(f"S3 Delete failed for {fname}: {e}")

    # 2. State Update
    state["training_runs"] = [r for r in state["training_runs"] if r["id"] != run_id]
    state["model_versions"] = [m for m in state["model_versions"] if m["id"] != run_id]
    
    _append_log(state, "warning", f"Deleted model and storage: {run_id}")
    _save_state(state)
    return {"message": f"Run {run_id} and associated files deleted"}

@app.post("/mlops/feedback")
async def create_feedback(
    file: UploadFile = File(...), feedback_type: str = Form(...), label: str = Form("unlabeled"),
    operator: str = Form(""), comment: str = Form(""), line: str = Form(""),
    gate_score: float = Form(0.0), heatmap_score: float = Form(0.0), predicted_label: str = Form("")
):
    state = _load_state()
    ext = Path(file.filename or "sample.png").suffix; fname = f"{uuid.uuid4().hex}{ext}"
    
    # 1. Physical Save (Local)
    fpath = MLOPS_ASSETS_ROOT / "feedback" / fname; _ensure_dir(fpath.parent)
    with open(fpath, "wb") as f: shutil.copyfileobj(file.file, f)
    
    # 2. Sync to S3
    if STORAGE_TYPE == "S3" and s3_utils:
        s3_utils.upload_file(str(fpath), S3_BUCKET_FEEDBACK, fname)

    fb_item = {
        "id": f"FDBK-{uuid.uuid4().hex[:8].upper()}", "feedback_type": feedback_type, "label": label, "operator": operator,
        "comment": comment, "line": line, "image_url": f"/mlops-assets/feedback/{fname}", "created_at": _iso_now(),
        "model_prediction": {"gate_score": gate_score, "heatmap_score": heatmap_score, "predicted_label": predicted_label}
    }
    state["feedback_items"].insert(0, fb_item); state["dataset_versions"][0]["feedback_count"] += 1
    _append_log(state, "info", f"Data Ingested (S3 synced): {feedback_type}"); _save_state(state)
    return {"message": "Success", "feedback_item": fb_item}

@app.delete("/mlops/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    state = _load_state()
    item_to_remove = next((item for item in state["feedback_items"] if item["id"] == feedback_id), None)
    if not item_to_remove: raise HTTPException(404, "Feedback not found")
    
    img_rel_path = item_to_remove["image_url"].replace("/mlops-assets/", "")
    img_abs_path = MLOPS_ASSETS_ROOT / img_rel_path
    fname = img_abs_path.name

    # 1. Local Delete
    if img_abs_path.exists(): os.remove(img_abs_path)
    
    # 2. S3 Delete
    if STORAGE_TYPE == "S3" and s3_utils:
        try:
            s3_utils.s3.delete_object(Bucket=S3_BUCKET_FEEDBACK, Key=fname)
        except Exception as e:
            logger.warning(f"S3 Feedback delete failed: {e}")

    # 3. State Update
    state["feedback_items"] = [item for item in state["feedback_items"] if item["id"] != feedback_id]
    state["dataset_versions"][0]["feedback_count"] = max(0, state["dataset_versions"][0]["feedback_count"] - 1)
    
    _append_log(state, "warning", f"Removed feedback and storage: {feedback_id}"); _save_state(state)
    return {"message": "Deleted"}

@app.post("/mlops/datasets/upload")
async def upload_data(
    files: List[UploadFile] = File(...),
    label: str = Form("unlabeled"),
    source_type: str = Form("upload"),
    line: str = Form(""),
    comment: str = Form(""),
    dataset_mode: str = Form("append"),
    dataset_version_id: str = Form(""),
    dataset_name: str = Form("")
):
    state = _load_state(); added = 0
    
    # 1. Determine target dataset version
    target_ds = None
    if dataset_mode == "new":
        new_ds_id = f"DATA-{uuid.uuid4().hex[:8].upper()}"
        target_ds = {
            "id": new_ds_id,
            "name": dataset_name or f"New Upload ({_iso_now()})",
            "status": "prepared",
            "sample_count": 0,
            "feedback_count": 0,
            "samples": [],
            "updated_at": _iso_now(),
            "notes": comment
        }
        state["dataset_versions"].insert(0, target_ds)
        state["active_dataset_id"] = new_ds_id
    else:
        # Append to existing
        target_id = dataset_version_id or state["active_dataset_id"]
        target_ds = next((d for d in state["dataset_versions"] if d["id"] == target_id), None)
        if not target_ds:
            # Fallback to creating new if target not found
            new_ds_id = f"DATA-AUTO-{uuid.uuid4().hex[:8].upper()}"
            target_ds = {"id": new_ds_id, "name": "Auto Created", "status": "prepared", "sample_count": 0, "feedback_count": 0, "samples": [], "updated_at": _iso_now()}
            state["dataset_versions"].insert(0, target_ds)

    # 2. Process Files
    for f in files:
        fname = f"{uuid.uuid4().hex}{Path(f.filename or '.png').suffix}"
        fpath = MLOPS_ASSETS_ROOT / "uploads" / fname; _ensure_dir(fpath.parent)
        with open(fpath, "wb") as out: shutil.copyfileobj(f.file, out)
        
        # Sync to S3
        if STORAGE_TYPE == "S3" and s3_utils:
            s3_utils.upload_file(str(fpath), S3_BUCKET_DATASETS, fname)
            
        # Register sample in state
        new_sample = {
            "id": f"SMPL-{uuid.uuid4().hex[:8].upper()}",
            "file_url": f"/mlops-assets/uploads/{fname}",
            "file_name": f.filename or "unknown.png",
            "created_at": _iso_now(),
            "metadata": {"label": label, "source": source_type, "line": line}
        }
        target_ds["samples"].insert(0, new_sample)
        added += 1
    
    # Limit samples to prevent JSON bloat
    target_ds["samples"] = target_ds["samples"][:100]
    target_ds["sample_count"] += added
    target_ds["updated_at"] = _iso_now()
    
    _append_log(state, "info", f"Upload complete to {target_ds['name']}: {added} files"); _save_state(state)
    return {"message": "Success", "added_count": added, "dataset_id": target_ds["id"]}

@app.post("/mlops/architectures/upload")
async def upload_architecture(
    file: UploadFile = File(...),
    kind: str = Form(...),
    name: str = Form(...)
):
    """모델 구조(Architecture) 등록 API"""
    state = _load_state()
    arch_id = f"ARCH-{kind.upper()}-{uuid.uuid4().hex[:4].upper()}"
    ext = Path(file.filename or "architecture.json").suffix or ".json"
    fname = f"{arch_id.lower()}_{uuid.uuid4().hex[:8]}{ext}"
    fpath = MLOPS_ASSETS_ROOT / "architectures" / fname
    _ensure_dir(fpath.parent)
    with open(fpath, "wb") as out:
        shutil.copyfileobj(file.file, out)

    new_arch = {
        "id": arch_id,
        "name": name,
        "kind": kind,
        "created_at": _iso_now(),
        "file_name": file.filename or fname,
        "file_url": f"/mlops-assets/architectures/{fname}",
        "interface": {"input": "Image (224x224)", "output": "Score/Heatmap"}
    }
    
    state["architectures"].append(new_arch)
    _append_log(state, "info", f"New Architecture Registered: {name} ({kind})")
    _save_state(state)
    return {"message": "Architecture registered", "architecture": new_arch}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
