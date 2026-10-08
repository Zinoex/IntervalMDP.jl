#!/usr/bin/env python3
"""Harness telemetry recorder.

Appends one event to the harness SQLite store (created on first use):
    events(id INTEGER PK AUTOINCREMENT, ts TEXT DEFAULT datetime('now'),
           event_name TEXT NOT NULL, details TEXT)

Every row carries `"_recorder": "record_event.py"` in its details.

Usage:
    record_event.py <eventName> '<json details>'
Env:
    HARNESS_TELEMETRY_DB  override DB path (default: harness/tools/telemetry/telemetry.db)
"""
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "..", "telemetry", "telemetry.db")


def record(event_name, details, db_path=None):
    db_path = db_path or os.environ.get("HARNESS_TELEMETRY_DB", DEFAULT_DB)
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except ValueError:
            details = {"raw": details}
    if isinstance(details, dict):
        details.setdefault("_recorder", "record_event.py")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts TEXT NOT NULL DEFAULT (datetime('now')),"
            " event_name TEXT NOT NULL,"
            " details TEXT)"
        )
        con.execute(
            "INSERT INTO events (event_name, details) VALUES (?, ?)",
            (event_name, json.dumps(details)),
        )
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: record_event.py <eventName> [jsonDetails]", file=sys.stderr)
        sys.exit(1)
    record(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "{}")
    print("recorded: " + sys.argv[1])
