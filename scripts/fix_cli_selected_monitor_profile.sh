#!/usr/bin/env bash
set -euo pipefail

cat > src/faceid/cli.py << 'PY'
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from faceid.db import (
    add_embedding,
    add_person,
    connect,
    delete_person,
    get_person,
    init_db,
    list_people,
    load_embeddings,
    log_event,
    set_primary_photo,
)
from faceid.engine import FaceEngine
from faceid.index import FaceIndex
from faceid.tracker import IoUTracker

app = typer.Typer(add_completion=False)
console = Console()


def ensure_db(db_path: str) -> None:
    con = connect(db_path)
    init_db(con)
    con.close()


@app.command()
def init(
    db: str = typer.Option("data/faceid.sqlite3", "--db", help="SQLite DB path"),
):
    """Initialize local database."""
    ensure_db(db)
    console.print(f"[green]Initialized DB:[/green] {db}")


@app.command()
def people(
    db: str = typer.Option("data/faceid.sqlite3", "--db", help="SQLite DB path"),
):
    """List people in the registry."""
    ensure_db(db)
    con = connect(db)
    rows = list_people(con)
    table = Table(title="People Registry")
    table.add_column("person_id", justify="right")
    table.add_column("full_name")
    table.add_column("category")
    table.add_column("phone")
    table.add_column("org")
    table.add_column("status")
    table.add_column("primary_photo_path")
    for p in rows:
        table.add_row(
            str(p.person_id),
            p.full_name,
            p.category,
            p.phone or "",
            p.org or "",
            p.status,
            p.primary_photo_path or "",
        )
    console.print(table)
    con.close()


@app.command()
def enroll(
    full_name: str = typer.Option(..., "--full-name", help="Full name"),
    category: str = typer.Option(..., "--category", help="personnel|stakeholder|visitor|watchlist"),
    phone: Optional[str] = typer.Option(None, "--phone", help="Phone number"),
    org: Optional[str] = typer.Option(None, "--org", help="Organisation"),
    images: List[Path] = typer.Option(..., "--images", help="One or more image paths"),
    db: str = typer.Option("data/faceid.sqlite3", "--db", help="SQLite DB path"),
    min_det_score: float = typer.Option(0.5, "--min-det-score", help="Minimum detection score to accept a face"),
):
    """Enroll a person from one or more photos (stores embeddings and a primary photo path)."""
    ensure_db(db)
    engine = FaceEngine()
    con = connect(db)

    pid = add_person(
        con,
        full_name=full_name,
        category=category,
        phone=phone,
        org=org,
        status="active",
        primary_photo_path=None,
    )

    added = 0
    primary_photo: Optional[str] = None

    for img_path in images:
        if not img_path.exists():
            console.print(f"[yellow]Skipping missing image:[/yellow] {img_path}")
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            console.print(f"[yellow]Skipping unreadable image:[/yellow] {img_path}")
            continue

        faces = engine.detect(bgr)
        if not faces:
            console.print(f"[yellow]No face detected:[/yellow] {img_path}")
            continue

        best = max(faces, key=lambda f: f.quality)
        if best.det_score < min_det_score:
            console.print(f"[yellow]Low detection score ({best.det_score:.2f}); skipping:[/yellow] {img_path}")
            continue

        add_embedding(con, person_id=pid, model_version=engine.model_version, vector=best.embedding, quality=best.quality)
        added += 1
        if primary_photo is None:
            primary_photo = str(img_path)

        console.print(f"[green]Added embedding[/green] from {img_path.name} (det={best.det_score:.2f}, q={best.quality:.1f})")

    if added == 0:
        console.print("[red]No embeddings were added. Deleting person record.[/red]")
        delete_person(con, pid)
        con.close()
        raise typer.Exit(code=1)

    if primary_photo is not None:
        set_primary_photo(con, pid, primary_photo)

    con.close()
    console.print(f"[green]Enrolled person_id={pid}[/green] ({added} embeddings).")


@app.command()
def remove(
    person_id: int = typer.Argument(..., help="Person ID to delete"),
    db: str = typer.Option("data/faceid.sqlite3", "--db", help="SQLite DB path"),
):
    """Delete a person and their embeddings."""
    ensure_db(db)
    con = connect(db)
    p = get_person(con, person_id)
    if not p:
        console.print("[red]Person not found.[/red]")
        con.close()
        raise typer.Exit(code=1)
    delete_person(con, person_id)
    con.close()
    console.print(f"[green]Deleted[/green] person_id={person_id} ({p.full_name}).")


def _safe_imread(path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(path)
        return img if img is not None else None
    except Exception:
        return None


def _thumb_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop square and resize."""
    if img_bgr is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    crop = img_bgr[y0 : y0 + m, x0 : x0 + m]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _render_history_panel(
    height: int,
    history: Deque[dict],
    photo_cache: Dict[int, Optional[np.ndarray]],
    panel_w: int = 420,
) -> np.ndarray:
    panel = np.zeros((height, panel_w, 3), dtype=np.uint8)

    cv2.putText(panel, "RECENT MATCHES", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    top = 44
    card_h = 104
    gap = 10
    max_cards = max(1, (height - top - 18) // (card_h + gap))
    items = list(history)[-max_cards:]

    y = top
    for item in items:
        pid = item["person_id"]
        name = item["name"]
        category = item["category"]
        phone = item["phone"]
        org = item["org"]
        score = item["score"]
        ts = item["ts_str"]

        cv2.rectangle(panel, (10, y), (panel_w - 10, y + card_h), (0, 255, 0), 1)

        # Thumb
        thumb_size = 84
        tx, ty = 18, y + 10
        photo = photo_cache.get(pid)
        if photo is not None:
            t = _thumb_square(photo, thumb_size)
            panel[ty : ty + thumb_size, tx : tx + thumb_size] = t
        else:
            cv2.rectangle(panel, (tx, ty), (tx + thumb_size, ty + thumb_size), (0, 255, 0), 1)
            cv2.putText(panel, "NO", (tx + 20, ty + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(panel, "PHOTO", (tx + 2, ty + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        x = tx + thumb_size + 14
        cv2.putText(panel, f"{name}"[:32], (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(panel, f"{category} | {org}"[:40], (x, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(panel, f"{phone}"[:28], (x, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(panel, f"{ts}  score={score:.2f}", (x, y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        y += card_h + gap

    cv2.putText(panel, "q=quit | click=select | Enter=start monitor | c=clear", (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return panel


@dataclass
class _GridCell:
    idx: int
    x1: int
    y1: int
    x2: int
    y2: int


class NonBlockingFaceGrid:
    """Non-blocking face selection grid (drawn and controlled inside the main loop)."""

    def __init__(self, window: str = "Select Faces", thumb: int = 140, cols: int = 4):
        self.window = window
        self.thumb = int(thumb)
        self.cols = int(cols)

        self.active = False
        self._grid_base: Optional[np.ndarray] = None
        self._cells: List[_GridCell] = []
        self._selected: Set[int] = set()

        self.snapshot_frame: Optional[np.ndarray] = None
        self.snapshot_faces: List = []

    def open(self, frame_bgr: np.ndarray, faces: List) -> None:
        self.snapshot_frame = frame_bgr.copy()
        self.snapshot_faces = faces
        self._selected = set()
        self._grid_base, self._cells = self._build_grid()
        self.active = True

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        def _on_mouse(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            for cell in self._cells:
                if cell.x1 <= x <= cell.x2 and cell.y1 <= y <= cell.y2:
                    if cell.idx in self._selected:
                        self._selected.remove(cell.idx)
                    else:
                        self._selected.add(cell.idx)
                    break

        cv2.setMouseCallback(self.window, _on_mouse)

    def close(self) -> None:
        if self.active:
            try:
                cv2.destroyWindow(self.window)
            except Exception:
                pass
        self.active = False
        self._grid_base = None
        self._cells = []
        self._selected = set()
        self.snapshot_frame = None
        self.snapshot_faces = []

    def selected_indices(self) -> List[int]:
        return sorted(self._selected)

    def _build_grid(self) -> Tuple[np.ndarray, List[_GridCell]]:
        assert self.snapshot_frame is not None

        pad = 16
        top_pad = 64

        thumbs = []
        for f in self.snapshot_faces:
            x1, y1, x2, y2 = f.bbox
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(self.snapshot_frame.shape[1] - 1, x2)
            y2 = min(self.snapshot_frame.shape[0] - 1, y2)
            crop = self.snapshot_frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((self.thumb, self.thumb, 3), dtype=np.uint8)
            thumbs.append(_thumb_square(crop, self.thumb))

        n = len(thumbs)
        cols = max(1, self.cols)
        rows = (n + cols - 1) // cols

        W = cols * self.thumb + (cols + 1) * pad
        H = top_pad + rows * self.thumb + (rows + 1) * pad
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        cv2.putText(canvas, "CLICK to select | ENTER/'s' start | r recapture | ESC close",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cells: List[_GridCell] = []
        for i, t in enumerate(thumbs):
            r = i // cols
            c = i % cols
            x = pad + c * (self.thumb + pad)
            y = top_pad + pad + r * (self.thumb + pad)
            canvas[y:y + self.thumb, x:x + self.thumb] = t
            cells.append(_GridCell(i, x, y, x + self.thumb, y + self.thumb))
            cv2.putText(canvas, f"{i+1}", (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return canvas, cells

    def draw(self) -> Optional[np.ndarray]:
        if not self.active or self._grid_base is None:
            return None
        img = self._grid_base.copy()
        for cell in self._cells:
            if cell.idx in self._selected:
                cv2.rectangle(img, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 255, 0), 4)
                cv2.putText(img, "SELECTED", (cell.x1 + 6, cell.y2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img


def _render_selected_monitor(
    tiles: List[dict],
    tile_w: int = 360,
    tile_h: int = 270,
    cols: int = 2,
) -> np.ndarray:
    """
    Each tile dict contains:
      sel_crop, live_crop, enroll_photo, name, category, phone, org, decision, score, ts_str
    """
    n = max(1, len(tiles))
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols
    pad = 12
    top = 56

    W = cols * tile_w + (cols + 1) * pad
    H = top + rows * tile_h + (rows + 1) * pad
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    cv2.putText(canvas, "SELECTED MONITOR | independent search + enrolled photo | c=clear | q=quit",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

    def draw_box(img, x, y, w, h, label, content):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if content is None:
            cv2.putText(img, "NO IMAGE", (x + 10, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return
        thumb = _thumb_square(content, min(w, h))
        thumb = cv2.resize(thumb, (w, h), interpolation=cv2.INTER_AREA)
        img[y:y + h, x:x + w] = thumb
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        x = pad + c * (tile_w + pad)
        y = top + pad + r * (tile_h + pad)

        cv2.rectangle(canvas, (x, y), (x + tile_w, y + tile_h), (0, 255, 0), 1)

        decision = tile.get("decision", "UNKNOWN")
        name = tile.get("name", "UNKNOWN")
        category = tile.get("category", "")
        phone = tile.get("phone", "")
        org = tile.get("org", "")
        score = tile.get("score", None)
        ts = tile.get("ts_str", "")

        cv2.putText(canvas, f"Selected {i+1} | {decision}", (x + 10, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

        img_top = y + 46
        img_h = 120
        img_w = (tile_w - 30) // 2

        sel_img = tile.get("live_crop") if tile.get("live_crop") is not None else tile.get("sel_crop")
        draw_box(canvas, x + 10, img_top, img_w, img_h, "Selected/Live", sel_img)
        draw_box(canvas, x + 20 + img_w, img_top, img_w, img_h, "Enrolled", tile.get("enroll_photo"))

        info_y = img_top + img_h + 30
        cv2.putText(canvas, f"Name: {name}"[:44], (x + 10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        info_y += 20
        if category or org:
            cv2.putText(canvas, f"Category: {category} | Org: {org}"[:52], (x + 10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            info_y += 20
        if phone:
            cv2.putText(canvas, f"Phone: {phone}"[:52], (x + 10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            info_y += 20
        if score is not None:
            cv2.putText(canvas, f"Score: {float(score):.2f}", (x + 10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if ts:
            cv2.putText(canvas, f"Time: {ts}", (x + 10, y + tile_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return canvas


@app.command()
def live(
    camera: str = typer.Option("0", "--camera", help="Camera index (0) or RTSP URL"),
    db: str = typer.Option("data/faceid.sqlite3", "--db", help="SQLite DB path"),
    camera_id: str = typer.Option("laptop_cam", "--camera-id", help="Camera ID for logging"),
    top_k: int = typer.Option(3, "--top-k", help="Top-K matches to consider"),
    match_threshold: float = typer.Option(0.45, "--match-threshold", help="Cosine threshold for 'Match'"),
    possible_threshold: float = typer.Option(0.32, "--possible-threshold", help="Cosine threshold for 'Possible match'"),
    det_interval: int = typer.Option(2, "--det-interval", help="Run detection every N frames"),
    min_det_score: float = typer.Option(0.55, "--min-det-score", help="Minimum face detection score"),
    min_quality: float = typer.Option(200.0, "--min-quality", help="Minimum quality score"),
    log_every_seconds: float = typer.Option(2.0, "--log-every-seconds", help="Log at most once per track per X seconds"),
    history_size: int = typer.Option(10, "--history-size", help="Max unique matches in side panel"),
    template_threshold: float = typer.Option(0.60, "--template-threshold", help="Similarity to selected template for live updates"),
):
    """
    Main window runs as usual.
    Click main window to open a face selection grid (non-blocking).
    Press Enter/'s' to start Selected Monitor:
      - Immediately runs independent DB search on selected face embedding(s)
      - Shows selected crop + enrolled photo + identity info
    """
    ensure_db(db)
    engine = FaceEngine()
    con = connect(db)

    vectors, person_ids, _ = load_embeddings(con, engine.model_version)
    if vectors.size == 0:
        console.print("[red]No embeddings found. Enroll at least one person first.[/red]")
        con.close()
        raise typer.Exit(code=1)

    index = FaceIndex(vectors=vectors, person_ids=person_ids)
    tracker = IoUTracker(iou_threshold=0.3, ttl_seconds=1.5)

    cap = cv2.VideoCapture(int(camera)) if camera.isdigit() else cv2.VideoCapture(camera)
    if not cap.isOpened():
        console.print("[red]Could not open camera/stream.[/red]")
        con.close()
        raise typer.Exit(code=1)

    main_win = "NCS FaceID CLI (Prototype)"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)

    click_state = {"requested": False}

    def on_main_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["requested"] = True

    cv2.setMouseCallback(main_win, on_main_mouse)

    selector = NonBlockingFaceGrid()
    monitor_win = "Selected Monitor"
    monitor_active = False

    selected_templates: List[np.ndarray] = []
    selected_tiles: List[dict] = []

    # history (dedup, max N)
    history: Deque[dict] = deque()
    history_ids: Set[int] = set()
    max_unique = max(1, int(history_size))

    photo_cache: Dict[int, Optional[np.ndarray]] = {}

    def ensure_photo_cached(pid: int) -> None:
        if pid in photo_cache:
            return
        p = get_person(con, pid)
        if p and p.primary_photo_path:
            photo_cache[pid] = _safe_imread(p.primary_photo_path)
        else:
            photo_cache[pid] = None

    def add_unique_match(pid: int, score: float) -> None:
        nonlocal history, history_ids
        if pid in history_ids:
            return
        while len(history) >= max_unique:
            oldest = history.popleft()
            history_ids.discard(int(oldest["person_id"]))
        p = get_person(con, pid)
        if not p:
            return
        ensure_photo_cached(pid)
        now = time.time()
        history.append(
            {
                "person_id": pid,
                "name": p.full_name,
                "category": p.category,
                "phone": p.phone or "",
                "org": p.org or "",
                "score": float(score),
                "ts_str": time.strftime("%H:%M:%S", time.localtime(now)),
            }
        )
        history_ids.add(pid)

    frame_i = 0
    track_state: Dict[int, Tuple[float, Optional[str], float, Optional[int]]] = {}
    latest_faces = []

    console.print("[green]Controls:[/green] click=select | Enter=start monitor | r recapture | c clear | q quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # click -> open selector from current frame snapshot
        if click_state["requested"]:
            click_state["requested"] = False
            faces_now = [f for f in engine.detect(frame) if f.det_score >= min_det_score]
            if faces_now:
                selector.open(frame, faces_now)

        # detection & tracking
        frame_i += 1
        if frame_i % det_interval == 0:
            faces = engine.detect(frame)
            latest_faces = [f for f in faces if f.det_score >= min_det_score]
            bboxes = [f.bbox for f in latest_faces]
            assignments = tracker.update(bboxes)
        else:
            assignments = {tid: t.bbox for tid, t in tracker.tracks.items()}

        bbox_to_face = {f.bbox: f for f in latest_faces}

        # main window recognition
        for tid, bbox in assignments.items():
            x1, y1, x2, y2 = bbox

            decision = "TRACK"
            conf = 0.0
            matched_pid = None

            if bbox in bbox_to_face:
                f = bbox_to_face[bbox]
                if f.quality < min_quality:
                    decision = "LOW_QUALITY"
                else:
                    matches = index.search(f.embedding, top_k=top_k)
                    if matches:
                        best = matches[0]
                        conf = best.score
                        if conf >= match_threshold:
                            decision = "MATCH"
                            matched_pid = best.person_id
                        elif conf >= possible_threshold:
                            decision = "POSSIBLE"
                            matched_pid = best.person_id
                        else:
                            decision = "UNKNOWN"
                    else:
                        decision = "UNKNOWN"

            # color: UNKNOWN red else green
            box_color = (0, 255, 0)
            if decision == "UNKNOWN":
                box_color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"ID {tid}: {decision}"
            if decision in ("MATCH", "POSSIBLE") and matched_pid is not None:
                p = get_person(con, matched_pid)
                if p:
                    label = f"ID {tid}: {decision} {p.full_name} ({conf:.2f})"
                else:
                    label = f"ID {tid}: {decision} person_id={matched_pid} ({conf:.2f})"
            elif decision == "LOW_QUALITY":
                label = f"ID {tid}: LOW QUALITY"

            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            if decision == "MATCH" and matched_pid is not None:
                add_unique_match(matched_pid, conf)

            now = time.time()
            last = track_state.get(tid, (0.0, None, 0.0, None))
            if (now - last[0]) >= log_every_seconds and decision in ("MATCH", "POSSIBLE", "UNKNOWN"):
                log_event(
                    con,
                    camera_id=camera_id,
                    track_id=tid,
                    decision=decision,
                    confidence=float(conf),
                    matched_person_id=matched_pid,
                    details=None,
                )
                track_state[tid] = (now, decision, conf, matched_pid)

        # monitor live updates (optional): if selected templates exist, try to find corresponding faces in latest_faces
        if monitor_active and selected_templates:
            best_per_t: List[Tuple[float, Optional[object]]] = [(0.0, None) for _ in selected_templates]
            for f in latest_faces:
                if f.quality < min_quality:
                    continue
                sims = [ _cosine(f.embedding, t) for t in selected_templates ]
                j = int(np.argmax(sims)) if sims else -1
                if j >= 0 and sims[j] >= template_threshold and sims[j] > best_per_t[j][0]:
                    best_per_t[j] = (sims[j], f)

            for i, (_sim, f) in enumerate(best_per_t):
                if f is None:
                    continue
                x1, y1, x2, y2 = f.bbox
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
                selected_tiles[i]["live_crop"] = crop
                selected_tiles[i]["ts_str"] = time.strftime("%H:%M:%S", time.localtime(time.time()))

                matches = index.search(f.embedding, top_k=top_k)
                if matches:
                    best = matches[0]
                    selected_tiles[i]["score"] = float(best.score)
                    if best.score >= match_threshold:
                        p1 = get_person(con, best.person_id)
                        selected_tiles[i]["decision"] = "MATCH"
                        if p1:
                            selected_tiles[i]["name"] = p1.full_name
                            selected_tiles[i]["category"] = p1.category
                            selected_tiles[i]["phone"] = p1.phone or ""
                            selected_tiles[i]["org"] = p1.org or ""
                            if selected_tiles[i].get("enroll_photo") is None and p1.primary_photo_path:
                                selected_tiles[i]["enroll_photo"] = _safe_imread(p1.primary_photo_path)
                    else:
                        selected_tiles[i]["decision"] = "UNKNOWN"
                else:
                    selected_tiles[i]["decision"] = "UNKNOWN"

            monitor_canvas = _render_selected_monitor(selected_tiles)
            cv2.imshow(monitor_win, monitor_canvas)

        # selector draw
        if selector.active:
            grid = selector.draw()
            if grid is not None:
                cv2.imshow(selector.window, grid)

        # main window composite
        h = frame.shape[0]
        panel = _render_history_panel(h, history, photo_cache, panel_w=420)
        combined = np.hstack([frame, panel])
        cv2.imshow(main_win, combined)

        # single key handler
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

        if k == ord("c"):
            monitor_active = False
            selected_templates = []
            selected_tiles = []
            selector.close()
            try:
                cv2.destroyWindow(monitor_win)
            except Exception:
                pass

        if selector.active:
            if k == 27:  # ESC
                selector.close()
            elif k == ord("r"):
                faces_now = [f for f in engine.detect(frame) if f.det_score >= min_det_score]
                if faces_now:
                    selector.close()
                    selector.open(frame, faces_now)
            elif k in (10, 13, ord("s")):  # Enter or s
                idxs = selector.selected_indices()
                if idxs and selector.snapshot_frame is not None:
                    # Build templates + tiles immediately (independent DB search)
                    selected_templates = [selector.snapshot_faces[i].embedding for i in idxs]

                    selected_tiles = []
                    now_ts = time.strftime("%H:%M:%S", time.localtime(time.time()))

                    for i_sel in idxs:
                        f0 = selector.snapshot_faces[i_sel]
                        sx1, sy1, sx2, sy2 = f0.bbox
                        sx1 = max(0, sx1); sy1 = max(0, sy1)
                        sx2 = min(selector.snapshot_frame.shape[1] - 1, sx2)
                        sy2 = min(selector.snapshot_frame.shape[0] - 1, sy2)
                        sel_crop = selector.snapshot_frame[sy1:sy2, sx1:sx2].copy() if (sy2 > sy1 and sx2 > sx1) else None

                        matches0 = index.search(f0.embedding, top_k=top_k)

                        tile = {
                            "sel_crop": sel_crop,
                            "live_crop": None,
                            "enroll_photo": None,
                            "name": "UNKNOWN",
                            "category": "",
                            "phone": "",
                            "org": "",
                            "decision": "UNKNOWN",
                            "score": None,
                            "ts_str": now_ts,
                        }

                        if matches0:
                            best0 = matches0[0]
                            tile["score"] = float(best0.score)
                            if best0.score >= match_threshold:
                                p0 = get_person(con, best0.person_id)
                                tile["decision"] = "MATCH"
                                if p0:
                                    tile["name"] = p0.full_name
                                    tile["category"] = p0.category
                                    tile["phone"] = p0.phone or ""
                                    tile["org"] = p0.org or ""
                                    if p0.primary_photo_path:
                                        tile["enroll_photo"] = _safe_imread(p0.primary_photo_path)
                                else:
                                    tile["name"] = f"person_id={best0.person_id}"
                            else:
                                tile["decision"] = "UNKNOWN"

                        selected_tiles.append(tile)

                    monitor_active = True
                    cv2.namedWindow(monitor_win, cv2.WINDOW_NORMAL)
                    cv2.imshow(monitor_win, _render_selected_monitor(selected_tiles))
                    selector.close()

    cap.release()
    cv2.destroyAllWindows()
    con.close()
PY

# sanity check: compile file so we catch indentation/syntax errors immediately
python -m py_compile src/faceid/cli.py

pip install -e . --force-reinstall >/dev/null

echo "Fixed CLI. Now run:"
echo "  faceid live --camera 0 --db data/faceid.sqlite3"
