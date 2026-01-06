from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from faceid.db import connect, init_db, add_person, list_people, delete_person, add_embedding, load_embeddings, get_person, log_event
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
    db: str = typer.Option("data/faceid.sqlite3", help="SQLite DB path"),
):
    """Initialize local database."""
    ensure_db(db)
    console.print(f"[green]Initialized DB:[/green] {db}")


@app.command()
def people(
    db: str = typer.Option("data/faceid.sqlite3", help="SQLite DB path"),
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
    for p in rows:
        table.add_row(str(p.person_id), p.full_name, p.category, p.phone or "", p.org or "", p.status)
    console.print(table)
    con.close()


@app.command()
def enroll(
    full_name: str = typer.Option(..., help="Full name"),
    category: str = typer.Option(..., help="personnel|stakeholder|visitor|watchlist"),
    phone: Optional[str] = typer.Option(None, help="Phone number"),
    org: Optional[str] = typer.Option(None, help="Organisation"),
    images: List[Path] = typer.Option(..., help="One or more image paths"),
    db: str = typer.Option("data/faceid.sqlite3", help="SQLite DB path"),
    min_det_score: float = typer.Option(0.5, help="Minimum detection score to accept a face"),
):
    """Enroll a person from one or more photos (stores embeddings)."""
    ensure_db(db)
    engine = FaceEngine()
    con = connect(db)

    pid = add_person(con, full_name=full_name, category=category, phone=phone, org=org, status="active")
    added = 0

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
        console.print(f"[green]Added embedding[/green] from {img_path.name} (det={best.det_score:.2f}, q={best.quality:.1f})")

    con.close()

    if added == 0:
        console.print("[red]No embeddings were added. Deleting person record.[/red]")
        con = connect(db)
        delete_person(con, pid)
        con.close()
        raise typer.Exit(code=1)

    console.print(f"[green]Enrolled person_id={pid}[/green] ({added} embeddings).")


@app.command()
def remove(
    person_id: int = typer.Argument(..., help="Person ID to delete"),
    db: str = typer.Option("data/faceid.sqlite3", help="SQLite DB path"),
):
    """Delete a person and their embeddings."""
    ensure_db(db)
    con = connect(db)
    p = get_person(con, person_id)
    if not p:
        console.print("[red]Person not found.[/red]")
        raise typer.Exit(code=1)
    delete_person(con, person_id)
    con.close()
    console.print(f"[green]Deleted[/green] person_id={person_id} ({p.full_name}).")


@app.command()
def live(
    camera: str = typer.Option("0", help="Camera index (0) or RTSP URL"),
    db: str = typer.Option("data/faceid.sqlite3", help="SQLite DB path"),
    camera_id: str = typer.Option("laptop_cam", help="Camera ID for logging"),
    top_k: int = typer.Option(3, help="Top-K matches to consider"),
    match_threshold: float = typer.Option(0.35, help="Cosine threshold for 'Match' (tune this)"),
    possible_threshold: float = typer.Option(0.28, help="Cosine threshold for 'Possible match'"),
    det_interval: int = typer.Option(2, help="Run detection every N frames (tracking in between)"),
    min_det_score: float = typer.Option(0.5, help="Minimum face detection score"),
    min_quality: float = typer.Option(1200.0, help="Minimum quality score (bbox size/conf)"),
    log_every_seconds: float = typer.Option(2.0, help="Log at most once per track per X seconds"),
):
    """
    Live camera recognition: shows green boxes + TrackIDs and matched identity panel.
    """
    ensure_db(db)
    engine = FaceEngine()
    con = connect(db)

    vectors, person_ids, _qualities = load_embeddings(con, engine.model_version)
    if vectors.size == 0:
        console.print("[red]No embeddings found. Enroll at least one person first.[/red]")
        raise typer.Exit(code=1)

    index = FaceIndex(vectors=vectors, person_ids=person_ids)
    tracker = IoUTracker(iou_threshold=0.3, ttl_seconds=1.5)

    cap = cv2.VideoCapture(int(camera)) if camera.isdigit() else cv2.VideoCapture(camera)
    if not cap.isOpened():
        console.print("[red]Could not open camera/stream.[/red]")
        raise typer.Exit(code=1)

    console.print("[green]Press 'q' to quit.[/green]")
    frame_i = 0

    # track_id -> (last_logged_ts, last_decision, last_conf, last_person_id)
    track_state = {}

    # We re-detect every det_interval frames
    latest_faces = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_i += 1

        if frame_i % det_interval == 0:
            faces = engine.detect(frame)
            # filter low score
            latest_faces = [f for f in faces if f.det_score >= min_det_score]
            bboxes = [f.bbox for f in latest_faces]
            assignments = tracker.update(bboxes)
        else:
            # Update tracker with no new detections: keep last assignments
            assignments = {tid: t.bbox for tid, t in tracker.tracks.items()}

        # Map bbox to embedding/quality when detection ran
        bbox_to_face = {f.bbox: f for f in latest_faces}

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

            # Build label
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

        cv2.imshow("NCS FaceID CLI (Prototype)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    con.close()
