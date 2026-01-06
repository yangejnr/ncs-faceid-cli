from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

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

    # Create person record (primary_photo_path will be set after we successfully add embeddings)
    pid = add_person(con, full_name=full_name, category=category, phone=phone, org=org, status="active", primary_photo_path=None)
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

        # Take the best face by quality
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


def _render_person_panel(
    height: int,
    person_name: str,
    category: str,
    phone: str,
    org: str,
    score: float,
    decision: str,
    img_bgr: Optional[np.ndarray],
    panel_w: int = 360,
) -> np.ndarray:
    panel = np.zeros((height, panel_w, 3), dtype=np.uint8)

    y = 24
    line_h = 26

    def put(text: str, y_pos: int, scale: float = 0.6, thickness: int = 2):
        cv2.putText(panel, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)

    # Title
    put("MATCH DETAILS", y, 0.75, 2)
    y += int(line_h * 1.4)

    # Photo at top (if available)
    if img_bgr is not None:
        max_w = panel_w - 24
        h_img, w_img = img_bgr.shape[:2]
        if w_img > 0 and h_img > 0:
            scale = max_w / float(w_img)
            new_w = int(w_img * scale)
            new_h = int(h_img * scale)
            resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # cap height to keep text visible
            max_h = int(height * 0.45)
            if new_h > max_h:
                scale2 = max_h / float(new_h)
                new_w2 = max(1, int(new_w * scale2))
                new_h2 = max(1, int(new_h * scale2))
                resized = cv2.resize(resized, (new_w2, new_h2), interpolation=cv2.INTER_AREA)

            ph, pw = resized.shape[:2]
            panel[ y:y+ph, 12:12+pw ] = resized
            y += ph + 18

    # Details
    put(f"Decision: {decision}", y); y += line_h
    put(f"Score: {score:.2f}", y); y += line_h
    put(f"Name: {person_name}", y); y += line_h
    put(f"Category: {category}", y); y += line_h
    put(f"Phone: {phone}", y); y += line_h
    put(f"Org: {org}", y); y += line_h

    # Footer note
    y = height - 16
    cv2.putText(panel, "Press 'q' to quit", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return panel


@app.command()
def live(
    camera: str = typer.Option("0", "--camera", help="Camera index (0) or RTSP URL"),
    db: str = typer.Option("data/faceid.sqlite3", "--db", help="SQLite DB path"),
    camera_id: str = typer.Option("laptop_cam", "--camera-id", help="Camera ID for logging"),
    top_k: int = typer.Option(3, "--top-k", help="Top-K matches to consider"),
    # Updated sensible defaults for your demo
    match_threshold: float = typer.Option(0.45, "--match-threshold", help="Cosine threshold for 'Match' (tune this)"),
    possible_threshold: float = typer.Option(0.32, "--possible-threshold", help="Cosine threshold for 'Possible match'"),
    det_interval: int = typer.Option(2, "--det-interval", help="Run detection every N frames (tracking in between)"),
    min_det_score: float = typer.Option(0.55, "--min-det-score", help="Minimum face detection score"),
    # Updated sensible default (you observed ~280-320 during enrollment)
    min_quality: float = typer.Option(200.0, "--min-quality", help="Minimum quality score (bbox size/conf)"),
    log_every_seconds: float = typer.Option(2.0, "--log-every-seconds", help="Log at most once per track per X seconds"),
):
    """
    Live camera recognition: shows green boxes + TrackIDs and a right-side panel with matched person's stored photo + details.
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

    # track_id -> (last_logged_ts, last_decision, last_conf, last_person_id)
    track_state: Dict[int, Tuple[float, Optional[str], float, Optional[int]]] = {}

    latest_faces = []
    photo_cache: Dict[int, Optional[np.ndarray]] = {}

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

        # Best card to show in the panel (prefer MATCH over POSSIBLE, and higher score)
        best_card = None  # (priority, score, person_id)
        # priority: MATCH=2, POSSIBLE=1
        # We will show only one panel (best in frame) to keep UI clean.

        for tid, bbox in assignments.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

            if decision in ("MATCH", "POSSIBLE") and matched_pid is not None:
                p = get_person(con, matched_pid)
                if p:
                    label = f"ID {tid}: {decision} {p.full_name} ({conf:.2f})"
                    priority = 2 if decision == "MATCH" else 1
                    candidate = (priority, conf, matched_pid)
                    if best_card is None or candidate > best_card:
                        best_card = candidate
                else:
                    label = f"ID {tid}: {decision} person_id={matched_pid} ({conf:.2f})"
            elif decision == "UNKNOWN":
                label = f"ID {tid}: UNKNOWN"
            elif decision == "LOW_QUALITY":
                label = f"ID {tid}: LOW QUALITY"

            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Rate-limited logging
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

        # Render side panel
        h, w = frame.shape[:2]
        panel = np.zeros((h, 360, 3), dtype=np.uint8)

        if best_card is not None:
            _priority, score, pid = best_card
            p = get_person(con, pid)
            if p:
                # Load cached photo if available
                img = photo_cache.get(pid)
                if img is None and p.primary_photo_path:
                    try:
                        loaded = cv2.imread(p.primary_photo_path)
                        img = loaded if loaded is not None else None
                    except Exception:
                        img = None
                    photo_cache[pid] = img

                panel = _render_person_panel(
                    height=h,
                    person_name=p.full_name,
                    category=p.category,
                    phone=p.phone or "",
                    org=p.org or "",
                    score=score,
                    decision="MATCH" if _priority == 2 else "POSSIBLE",
                    img_bgr=img,
                    panel_w=360,
                )
            else:
                cv2.putText(panel, "No person data", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(panel, "No match in frame", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(panel, "Press 'q' to quit", (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        combined = np.hstack([frame, panel])
        cv2.imshow("NCS FaceID CLI (Prototype)", combined)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    con.close()
