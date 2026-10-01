import sqlite3, threading, re, os
from pathlib import Path

DB_PATH = Path(os.environ.get('TRANSLATOR_DB', Path(__file__).resolve().parent.parent / 'data' / 'translator.db'))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_lock = threading.RLock()

def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=NORMAL')
    return c

def init_db():
    with _lock, _conn() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS translations (
          source TEXT NOT NULL, source_lang TEXT NOT NULL, target_lang TEXT NOT NULL,
          translated TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY(source, source_lang, target_lang)
        );
        CREATE TABLE IF NOT EXISTS glossary (
          source TEXT NOT NULL, source_lang TEXT NOT NULL, target_lang TEXT NOT NULL,
          target TEXT NOT NULL, kind TEXT DEFAULT 'manual', confidence REAL DEFAULT 1.0,
          PRIMARY KEY(source, source_lang, target_lang)
        );
        CREATE TABLE IF NOT EXISTS projects (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, config_json TEXT NOT NULL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS project_files (
          project_id TEXT NOT NULL, filename TEXT NOT NULL, path TEXT NOT NULL,
          status TEXT DEFAULT 'pending', progress REAL DEFAULT 0,
          PRIMARY KEY(project_id, filename)
        );
        ''')

def get_translation(source, sl, tl):
    with _lock, _conn() as c:
        r=c.execute('SELECT translated FROM translations WHERE source=? AND source_lang=? AND target_lang=?',(source,sl,tl)).fetchone()
        return r[0] if r else None

def put_translation(source, sl, tl, translated):
    if not source or not translated: return
    with _lock, _conn() as c:
        c.execute('INSERT INTO translations(source,source_lang,target_lang,translated) VALUES(?,?,?,?) ON CONFLICT(source,source_lang,target_lang) DO UPDATE SET translated=excluded.translated,updated_at=CURRENT_TIMESTAMP',(source,sl,tl,translated))

def get_glossary(sl, tl):
    with _lock, _conn() as c:
        return {r[0]:r[1] for r in c.execute('SELECT source,target FROM glossary WHERE source_lang=? AND target_lang=?',(sl,tl)).fetchall()}

def put_glossary(source, sl, tl, target, kind='auto', confidence=0.5):
    if not source or not target: return
    with _lock, _conn() as c:
        c.execute('INSERT INTO glossary(source,source_lang,target_lang,target,kind,confidence) VALUES(?,?,?,?,?,?) ON CONFLICT(source,source_lang,target_lang) DO UPDATE SET target=excluded.target,kind=excluded.kind,confidence=excluded.confidence',(source,sl,tl,target,kind,confidence))

def upsert_project(pid,name,config_json):
    with _lock, _conn() as c:
        c.execute('INSERT INTO projects(id,name,config_json) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,config_json=excluded.config_json,updated_at=CURRENT_TIMESTAMP',(pid,name,config_json))

def get_project(pid):
    with _lock, _conn() as c:
        return c.execute('SELECT id,name,config_json,created_at,updated_at FROM projects WHERE id=?',(pid,)).fetchone()
