from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Optional
import numpy as np

DEFAULT_DB_PATH = os.path.join("data", "faceid.sqlite3")


@dataclass(frozen=True)
class Person:
    person_id: int
    full_name: str
    category: str
    phone: Optional[str]
    org: Optional[str]
    status: str
    primary_photo_path: Optional[str]


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def _column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info({table});").fetchall()
    cols = {r[1] for r in rows}  # r[1] is column name
    return column in cols


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS people (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            category TEXT NOT NULL,
            phone TEXT,
            org TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            vector BLOB NOT NULL,
            dims INTEGER NOT NULL,
            quality REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(person_id) REFERENCES people(person_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL DEFAULT (datetime('now')),
            camera_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            decision TEXT NOT NULL,
            confidence REAL NOT NULL,
            matched_person_id INTEGER,
            details TEXT,
            FOREIGN KEY(matched_person_id) REFERENCES people(person_id)
        );
        """
    )

    # Lightweight schema migration: add primary_photo_path if missing
    if not _column_exists(con, "people", "primary_photo_path"):
        con.execute("ALTER TABLE people ADD COLUMN primary_photo_path TEXT;")

    con.commit()


def add_person(
    con: sqlite3.Connection,
    full_name: str,
    category: str,
    phone: Optional[str],
    org: Optional[str],
    status: str = "active",
    primary_photo_path: Optional[str] = None,
) -> int:
    cur = con.execute(
        "INSERT INTO people (full_name, category, phone, org, status, primary_photo_path) VALUES (?, ?, ?, ?, ?, ?)",
        (full_name.strip(), category.strip(), phone, org, status, primary_photo_path),
    )
    con.commit()
    return int(cur.lastrowid)


def set_primary_photo(con: sqlite3.Connection, person_id: int, primary_photo_path: str) -> None:
    con.execute(
        "UPDATE people SET primary_photo_path = ? WHERE person_id = ?",
        (primary_photo_path, int(person_id)),
    )
    con.commit()


def list_people(con: sqlite3.Connection) -> list[Person]:
    rows = con.execute(
        "SELECT person_id, full_name, category, phone, org, status, primary_photo_path FROM people ORDER BY person_id"
    ).fetchall()
    return [Person(*row) for row in rows]


def get_person(con: sqlite3.Connection, person_id: int) -> Optional[Person]:
    row = con.execute(
        "SELECT person_id, full_name, category, phone, org, status, primary_photo_path FROM people WHERE person_id = ?",
        (person_id,),
    ).fetchone()
    return Person(*row) if row else None


def delete_person(con: sqlite3.Connection, person_id: int) -> None:
    con.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
    con.commit()


def add_embedding(
    con: sqlite3.Connection,
    person_id: int,
    model_version: str,
    vector: np.ndarray,
    quality: float,
) -> int:
    vec = np.asarray(vector, dtype=np.float32)
    cur = con.execute(
        "INSERT INTO embeddings (person_id, model_version, vector, dims, quality) VALUES (?, ?, ?, ?, ?)",
        (person_id, model_version, vec.tobytes(), int(vec.size), float(quality)),
    )
    con.commit()
    return int(cur.lastrowid)


def load_embeddings(con: sqlite3.Connection, model_version: str) -> tuple[np.ndarray, list[int], list[float]]:
    rows = con.execute(
        "SELECT person_id, vector, dims, quality FROM embeddings WHERE model_version = ?",
        (model_version,),
    ).fetchall()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), [], []

    person_ids: list[int] = []
    qualities: list[float] = []
    vectors: list[np.ndarray] = []
    dims_expected = None

    for pid, blob, dims, q in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size != int(dims):
            continue
        if dims_expected is None:
            dims_expected = v.size
        if v.size != dims_expected:
            continue
        vectors.append(v)
        person_ids.append(int(pid))
        qualities.append(float(q))

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), [], []

    mat = np.vstack(vectors).astype(np.float32)
    return mat, person_ids, qualities


def log_event(
    con: sqlite3.Connection,
    camera_id: str,
    track_id: int,
    decision: str,
    confidence: float,
    matched_person_id: Optional[int],
    details: Optional[str] = None,
) -> None:
    con.execute(
        "INSERT INTO events (camera_id, track_id, decision, confidence, matched_person_id, details) VALUES (?, ?, ?, ?, ?, ?)",
        (camera_id, int(track_id), decision, float(confidence), matched_person_id, details),
    )
    con.commit()
