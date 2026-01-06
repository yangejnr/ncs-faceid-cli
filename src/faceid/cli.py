from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Deque, Set

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from faceid.db import (
    connect,
    init_db,
    add_person,
    list_people,
    delete_person,
    add_embedding,
    load_embeddings,
    get_person,
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
    """Enroll a person from one or more photos (stores embeddings)."""
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


def _thumb(img_bgr: np.ndarray, size: int = 84) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    crop = img_bgr[y0:y0+m, x0:x0+m]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


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

    items = list(history)[-max_cards:]  # oldest->newest; show what fits at bottom
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

        thumb_size = 84
        tx, ty = 18, y + 10
        photo = photo_cache.get(pid)
        if photo is not None:
            t = _thumb(photo, size=thumb_size)
            panel[ty:ty+thumb_size, tx:tx+thumb_size] = t
        else:
            cv2.rectangle(panel, (tx, ty), (tx + thumb_size, ty + thumb_size), (0, 255, 0), 1)
            cv2.putText(panel, "NO", (tx + 20, ty + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(panel, "PHOTO", (tx + 2, ty + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        x = tx + thumb_size + 14
        cv2.putText(panel, f"{name}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(panel, f"{category} | {org}", (x, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(panel, f"{phone}", (x, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(panel, f"{ts}  score={score:.2f}", (x, y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        y += card_h + gap

    cv2.putText(panel, "Press 'q' to quit", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(panel, f"Stored unique matches: {len(history)}", (230, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return panel


@app.command()
def live(
    camera: str = typer.Option("0", "--camera", help="Camera index (0) or RTSP URL"),
    db: str = typer.Option("data/faceid.sqlite3", "--db", help="SQLite DB path"),
    camera_id: str = typer.Option("laptop_cam", "--camera-id", help="Camera ID for logging"),
    top_k: int = typer.Option(3, "--top-k", help="Top-K matches to consider"),
    match_threshold: float = typer.Option(0.45, "--match-threshold", help="Cosine threshold for 'Match' (tune this)"),
    possible_threshold: float = typer.Option(0.32, "--possible-threshold", help="Cosine threshold for 'Possible match'"),
    det_interval: int = typer.Option(2, "--det-interval", help="Run detection every N frames (tracking in between)"),
    min_det_score: float = typer.Option(0.55, "--min-det-score", help="Minimum face detection score"),
    min_quality: float = typer.Option(200.0, "--min-quality", help="Minimum quality score (bbox size/conf)"),
    log_every_seconds: float = typer.Option(2.0, "--log-every-seconds", help="Log at most once per track per X seconds"),
    history_size: int = typer.Option(10, "--history-size", help="Max unique matches stored in the side panel"),
):
    """
    Live camera recognition with a deduplicated match log:
      - A person is added ONCE to the log.
      - Same person will not be repeated while in view.
      - Max N unique persons retained; when full, oldest is evicted (FIFO).
    """
    ensure_db(db)
    engine = FaceEngine()
    con = connect(db)

    vectors, person_ids, _qualities = load_embeddings(con, engine.model_version)
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

    console.print("[green]Press 'q' to quit.[/green]")
    frame_i = 0

    track_state: Dict[int, Tuple[float, Optional[str], float, Optional[int]]] = {}
    latest_faces = []

    photo_cache: Dict[int, Optional[np.ndarray]] = {}

    # Deduplicated history
    history: Deque[dict] = deque()
    history_ids: Set[int] = set()
    max_unique = max(1, int(history_size))

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
            return  # do not repeat

        # Evict oldest if full
        while len(history) >= max_unique:
            oldest = history.popleft()
            history_ids.discard(int(oldest["person_id"]))

        p = get_person(con, pid)
        if not p:
            return
        ensure_photo_cached(pid)
        now = time.time()
        history.append({
            "person_id": pid,
            "name": p.full_name,
            "category": p.category,
            "phone": p.phone or "",
            "org": p.org or "",
            "score": float(score),
            "ts": now,
            "ts_str": time.strftime("%H:%M:%S", time.localtime(now)),
        })
        history_ids.add(pid)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_i += 1

        if frame_i % det_interval == 0:
            faces = engine.detect(frame)
            latest_faces = [f for f in faces if f.det_score >= min_det_score]
            bboxes = [f.bbox for f in latest_faces]
            assignments = tracker.update(bboxes)
        else:
            assignments = {tid: t.bbox for tid, t in tracker.tracks.items()}

        bbox_to_face = {f.bbox: f for f in latest_faces}

        for tid, bbox in assignments.items():
            x1, y1, x2, y2 = bbox

            label = f"ID {tid}: Tracking"
            decision = "TRACK"
            conf = 0.0
            matched_pid = None

            if bbox in bbox_to_face:
                f = bbox_to_face[bbox]
                if f.quality >= min_quality:
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
                            matched_pid = None
                    else:
                        decision = "UNKNOWN"
                else:
                    decision = "LOW_QUALITY"

            # Overlay label
            if decision in ("MATCH", "POSSIBLE") and matched_pid is not None:
                p = get_person(con, matched_pid)
                if p:
                    label = f"ID {tid}: {decision} {p.full_name} ({conf:.2f})"
                else:
                    label = f"ID {tid}: {decision} person_id={matched_pid} ({conf:.2f})"
            elif decision == "UNKNOWN":
                label = f"ID {tid}: UNKNOWN"
            elif decision == "LOW_QUALITY":
                label = f"ID {tid}: LOW QUALITY"

            # Box color: UNKNOWN -> red, otherwise green
            box_color = (0, 255, 0)
            if decision == "UNKNOWN":
                box_color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            # Add to history only on MATCH, and only once per person (dedup)
            if decision == "MATCH" and matched_pid is not None:
                add_unique_match(matched_pid, conf)

            # Rate-limited event logging
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

        # Render side panel with deduplicated history
        h, w = frame.shape[:2]
        panel = _render_history_panel(height=h, history=history, photo_cache=photo_cache, panel_w=420)

        combined = np.hstack([frame, panel])
        cv2.imshow("NCS FaceID CLI (Prototype)", combined)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    con.close()
