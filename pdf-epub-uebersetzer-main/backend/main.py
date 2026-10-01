import uuid
import shutil
import threading
import traceback
from pathlib import Path
import json
import hashlib
from memory import init_db, upsert_project, get_project
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from pdf_processor import process_pdf, assemble_pdf, get_pdf_page_count, split_pdf
from epub_processor import process_epub, get_epub_section_count, split_epub

init_db()

print("NEXUS Translator: zentraler Google-Web-Rate-Limiter aktiv (ca. 3.5 Requests/s)")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
SPLIT_DIR = BASE_DIR / "splits"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
SPLIT_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESUME_INDEX = DATA_DIR / "resume_index.json"

app = FastAPI(title="PDF Übersetzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Verhindert, dass der Browser die Weboberfläche (index.html, JS, CSS)
    zwischenspeichert. Ohne das kann es passieren, dass nach einem Update der
    Frontend-Datei im Browser trotzdem hartnäckig die alte Version angezeigt
    wird, egal wie oft man neu lädt - besonders bei localhost-Seiten ohne
    HTTPS neigen Browser zu aggressivem Caching."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheMiddleware)

# Einfache In-Memory-Job-Verwaltung (reicht für lokalen Single-User-Betrieb)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# Separate, viel einfachere Verwaltung für den Splitter: kein Fortschritt,
# kein Abbruch-Mechanismus nötig, da Splitten (reines Seiten-/Kapitel-Kopieren
# ohne Übersetzung/OCR) praktisch sofort fertig ist statt Minuten zu dauern.
SPLITS: dict[str, dict] = {}
SPLITS_LOCK = threading.Lock()

# Mapping Argos-Sprachcode -> Tesseract-Sprachcode für OCR
OCR_LANG_MAP = {
    "de": "deu",
    "en": "eng",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "ja": "jpn_vert",  # vertikal geschriebenes Japanisch (Light Novels, Manga-Text etc.)
}


def _run_pdf_job(job_id: str, input_path: Path, output_path: Path, from_code: str, to_code: str, grammar_check: bool, cancel_event: threading.Event):
    def progress_callback(stats):
        with JOBS_LOCK:
            if job_id not in JOBS:
                return
            JOBS[job_id]["progress"] = {
                "pages_done": stats.pages_done,
                "pages_total": stats.pages_total,
                "pages_failed": stats.pages_failed,
                "text_blocks_translated": stats.text_blocks_translated,
                "images_with_text": stats.images_with_text,
                "images_untouched": stats.images_untouched,
            }

    ocr_lang = OCR_LANG_MAP.get(from_code, "eng")
    pages_dir = str(OUTPUT_DIR / f"{job_id}_pages")
    # Fortschritt sofort setzen. Sonst zeigt das Frontend während der
    # PDF-Analyse irreführend dauerhaft "Verarbeitung läuft", obwohl noch
    # keine Seite abgeschlossen wurde.
    try:
        total_pages = get_pdf_page_count(str(input_path))
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["progress"] = {
                    "pages_done": 0,
                    "pages_total": total_pages,
                    "pages_failed": 0,
                    "text_blocks_translated": 0,
                    "images_with_text": 0,
                    "images_untouched": 0,
                }
    except Exception as e:
        print(f"Seitenzahl konnte vorab nicht gelesen werden: {e}")

    process_pdf(
        str(input_path), str(output_path), from_code, to_code,
        ocr_lang=ocr_lang, grammar_check=grammar_check,
        pages_dir=pages_dir, progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


def _run_epub_job(job_id: str, input_path: Path, output_path: Path, from_code: str, to_code: str, grammar_check: bool, cancel_event: threading.Event):
    def progress_callback(stats):
        with JOBS_LOCK:
            if job_id not in JOBS:
                return
            JOBS[job_id]["progress"] = {
                # gleiche Schlüssel wie beim PDF-Fortschritt, damit das
                # Frontend die Fortschrittsanzeige unverändert wiederverwenden
                # kann (Dateien statt Seiten)
                "pages_done": stats.files_done,
                "pages_total": stats.files_total,
                "text_blocks_translated": stats.blocks_translated,
                "images_with_text": 0,
                "images_untouched": 0,
            }

    # EPUB-Verarbeitung läuft in einem Rutsch (kein OCR/Rendering, daher
    # deutlich schneller als PDF) - Zwischenspeichern pro Seite ist hier
    # nicht nötig, die Datei wird erst am Ende geschrieben.
    process_epub(
        str(input_path), str(output_path), from_code, to_code,
        grammar_check=grammar_check, progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


def _run_job(job_id: str, input_path: Path, output_path: Path, from_code: str, to_code: str, file_type: str, grammar_check: bool = False):
    with JOBS_LOCK:
        cancel_event = JOBS[job_id]["cancel_event"]

    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"

        if file_type == "epub":
            _run_epub_job(job_id, input_path, output_path, from_code, to_code, grammar_check, cancel_event)
        else:
            _run_pdf_job(job_id, input_path, output_path, from_code, to_code, grammar_check, cancel_event)

        with JOBS_LOCK:
            if job_id not in JOBS:
                return  # Job wurde inzwischen gelöscht
            if cancel_event.is_set():
                JOBS[job_id]["status"] = "cancelled"
            else:
                JOBS[job_id]["status"] = "done"
                JOBS[job_id]["output_path"] = str(output_path)

    except Exception as e:
        tb = traceback.format_exc()
        print("=" * 60)
        print(f"FEHLER BEI DER {file_type.upper()}-VERARBEITUNG:")
        print(tb)
        print("=" * 60)
        with JOBS_LOCK:
            if job_id not in JOBS:
                return
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["traceback"] = tb


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    from_lang: str = Form(...),
    to_lang: str = Form(...),
    grammar_check: str = Form("false"),
    project_name: str = Form(""),
):
    filename_lower = file.filename.lower()
    if filename_lower.endswith(".pdf"):
        file_type = "pdf"
    elif filename_lower.endswith(".epub"):
        file_type = "epub"
    else:
        raise HTTPException(status_code=400, detail="Nur PDF- oder EPUB-Dateien werden unterstützt.")

    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}.{file_type}"
    output_path = OUTPUT_DIR / f"{job_id}_translated.{file_type}"

    payload = await file.read()
    file_hash = hashlib.sha256(payload).hexdigest()
    with open(input_path, "wb") as f:
        f.write(payload)

    grammar_check_bool = grammar_check.lower() == "true"
    project_id = None
    if project_name.strip():
        project_id = str(uuid.uuid4())
        config = {"from_lang": from_lang, "to_lang": to_lang, "grammar_check": grammar_check_bool, "glossary": "automatic", "resume": True}
        upsert_project(project_id, project_name.strip(), json.dumps(config, ensure_ascii=False))

    # Kostenloser Resume-Mechanismus: dieselbe Quelldatei kann nach einem
    # App-/Server-Neustart erneut hochgeladen werden. Bereits fertige Seiten
    # werden übernommen und process_pdf überspringt sie.
    try:
        resume_index = json.loads(RESUME_INDEX.read_text(encoding="utf-8")) if RESUME_INDEX.exists() else {}
    except Exception:
        resume_index = {}
    previous_id = resume_index.get(file_hash)
    previous_pages = OUTPUT_DIR / f"{previous_id}_pages" if previous_id else None
    if previous_id and previous_pages and previous_pages.is_dir():
        current_pages = OUTPUT_DIR / f"{job_id}_pages"
        shutil.copytree(previous_pages, current_pages, dirs_exist_ok=True)

    resume_index[file_hash] = job_id
    try:
        RESUME_INDEX.write_text(json.dumps(resume_index, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "filename": file.filename,
            "file_type": file_type,
            "from_lang": from_lang,
            "to_lang": to_lang,
            "project_id": project_id,
            "source_hash": file_hash,
            "progress": {"pages_done": 0, "pages_total": 0},
            "cancel_event": threading.Event(),
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, input_path, output_path, from_lang, to_lang, file_type, grammar_check_bool),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nicht gefunden.")
    # cancel_event ist ein threading.Event und lässt sich nicht in JSON
    # umwandeln - daher gezielt ausschließen statt den ganzen Job zurückzugeben.
    return {k: v for k, v in job.items() if k != "cancel_event"}


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Bricht eine laufende Übersetzung ab und löscht Job + zugehörige
    Dateien (Upload, Ausgabedatei, Seiten-Zwischenstand). Wird vom "X"-Button
    im Frontend nach Bestätigung des Lösch-/Abbruch-Dialogs aufgerufen."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job nicht gefunden.")
        job["cancel_event"].set()
        file_type = job.get("file_type", "pdf")
        status = job.get("status")

    # Laufende Netzwerk-/OCR-Aufrufe für die aktuell in Bearbeitung
    # befindliche Seite lassen sich nicht hart unterbrechen - der
    # Hintergrund-Thread bemerkt das cancel_event beim nächsten Schleifen-
    # durchlauf und beendet sich dann selbst (siehe process_pdf/process_epub).
    # Bereits "queued" oder bereits "done"/"error" gestoppte Jobs können wir
    # sofort aufräumen.
    if status in ("queued", "done", "error", "cancelled"):
        _delete_job_files(job_id, file_type)
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        return {"status": "deleted"}

    return {"status": "cancelling"}


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Löscht einen bereits beendeten/abgebrochenen Job endgültig samt
    Dateien. Wird vom Frontend aufgerufen, nachdem cancel_job() dem laufenden
    Thread Zeit gegeben hat, sich zu beenden."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return {"status": "already_deleted"}
        file_type = job.get("file_type", "pdf")

    _delete_job_files(job_id, file_type)
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    return {"status": "deleted"}


def _delete_job_files(job_id: str, file_type: str):
    for path in [
        UPLOAD_DIR / f"{job_id}.{file_type}",
        OUTPUT_DIR / f"{job_id}_translated.{file_type}",
    ]:
        path.unlink(missing_ok=True)
    pages_dir = OUTPUT_DIR / f"{job_id}_pages"
    if pages_dir.is_dir():
        shutil.rmtree(pages_dir, ignore_errors=True)


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nicht gefunden.")

    file_type = job.get("file_type", "pdf")
    output_path = OUTPUT_DIR / f"{job_id}_translated.{file_type}"

    if file_type == "pdf":
        pages_dir = OUTPUT_DIR / f"{job_id}_pages"
        # Falls die fertige Gesamtdatei noch nicht existiert (Job läuft noch
        # oder wurde unterbrochen), aus den bisher fertigen Einzelseiten live
        # zusammenbauen - so ist ein Download JEDERZEIT möglich, nicht erst
        # nach Abschluss oder nur nach einer Unterbrechung.
        if not output_path.exists():
            num_assembled = assemble_pdf(str(pages_dir), str(output_path))
            if num_assembled == 0:
                raise HTTPException(status_code=404, detail="Noch keine Seite fertig übersetzt.")
        media_type = "application/pdf"
    else:
        # EPUB wird als Ganzes verarbeitet (kein OCR/Rendering pro Seite),
        # daher gibt es hier keinen sinnvollen Zwischenstand - die Datei
        # existiert erst, wenn die Übersetzung fertig ist.
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Die Übersetzung ist noch nicht fertig.")
        media_type = "application/epub+zip"

    prefix = "translated_" if job.get("status") == "done" else "teilweise_übersetzt_"
    return FileResponse(
        output_path,
        media_type=media_type,
        filename=f"{prefix}{job['filename']}",
    )


@app.post("/api/split/upload")
async def split_upload(file: UploadFile = File(...)):
    """Nimmt eine PDF oder EPUB entgegen und liefert die Gesamt-Seiten-
    bzw. Abschnittszahl zurück, damit das Frontend den Bereichs-Regler
    (von Seite X bis Y) mit dem richtigen Maximum anzeigen kann. Übersetzt
    nichts - reine Analyse."""
    filename_lower = file.filename.lower()
    if filename_lower.endswith(".pdf"):
        file_type = "pdf"
    elif filename_lower.endswith(".epub"):
        file_type = "epub"
    else:
        raise HTTPException(status_code=400, detail="Nur PDF- oder EPUB-Dateien werden unterstützt.")

    split_id = str(uuid.uuid4())
    input_path = SPLIT_DIR / f"{split_id}.{file_type}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        if file_type == "pdf":
            total_units = get_pdf_page_count(str(input_path))
        else:
            total_units = get_epub_section_count(str(input_path))
    except Exception as e:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Datei konnte nicht gelesen werden: {e}")

    with SPLITS_LOCK:
        SPLITS[split_id] = {
            "filename": file.filename,
            "file_type": file_type,
            "input_path": str(input_path),
            "created_at": datetime.now().isoformat(),
        }

    return {
        "split_id": split_id,
        "file_type": file_type,
        "total_units": total_units,
        # "unit_label" sagt dem Frontend, ob es von "Seite" oder "Abschnitt"
        # sprechen soll - EPUB hat anders als PDF keine festen Bildschirm-
        # seiten (Text fließt je nach Gerät neu), gezählt werden dort
        # Kapitel-/Abschnittsdateien in Lesereihenfolge.
        "unit_label": "Seite" if file_type == "pdf" else "Abschnitt",
    }


@app.post("/api/split/execute")
async def split_execute(
    split_id: str = Form(...),
    start: int = Form(...),
    end: int = Form(...),
):
    with SPLITS_LOCK:
        entry = SPLITS.get(split_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Datei nicht gefunden - bitte erneut hochladen.")

    input_path = entry["input_path"]
    file_type = entry["file_type"]
    output_path = SPLIT_DIR / f"{split_id}_split.{file_type}"

    try:
        if file_type == "pdf":
            split_pdf(input_path, str(output_path), start, end)
        else:
            split_epub(input_path, str(output_path), start, end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    base_name = Path(entry["filename"]).stem
    ext = Path(entry["filename"]).suffix
    download_name = f"{base_name}_S{start}-{end}{ext}"
    media_type = "application/pdf" if file_type == "pdf" else "application/epub+zip"

    return FileResponse(output_path, media_type=media_type, filename=download_name)


@app.post("/api/projects")
async def create_project(name: str = Form(...), from_lang: str = Form(...), to_lang: str = Form(...), grammar_check: str = Form("false")):
    pid = str(uuid.uuid4())
    config = {"from_lang": from_lang, "to_lang": to_lang, "grammar_check": grammar_check.lower()=="true", "glossary": "automatic", "resume": True}
    upsert_project(pid, name, json.dumps(config, ensure_ascii=False))
    return {"project_id": pid, "name": name, "config": config}

@app.get("/api/projects/{project_id}")
async def read_project(project_id: str):
    row=get_project(project_id)
    if not row: raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    return {"project_id":row[0],"name":row[1],"config":json.loads(row[2]),"created_at":row[3],"updated_at":row[4]}

# Frontend als statische Dateien ausliefern
app.mount("/", StaticFiles(directory=str(BASE_DIR / "frontend"), html=True), name="frontend")
