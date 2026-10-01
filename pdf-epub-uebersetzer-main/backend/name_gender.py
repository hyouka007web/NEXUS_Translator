"""Kostenlose lokale Namens-/Geschlechtserkennung.

Nutzen: gender-guesser enthält eine große, lokale Liste internationaler Namen.
Es gibt keine API und keine Online-Abfrage. Für Namen, die nicht in der
Datenbank vorkommen, bleibt das Ergebnis unknown. Die Datei data/names.json
kann jederzeit um eigene Namen erweitert werden.
"""
import json, re
from pathlib import Path
try:
    import gender_guesser.detector as gender_detector
except Exception:
    gender_detector = None

CUSTOM = Path(__file__).resolve().parent.parent / 'data' / 'names.json'
try:
    custom = json.loads(CUSTOM.read_text(encoding='utf-8')) if CUSTOM.exists() else {}
except Exception:
    custom = {}

detector = gender_detector.Detector(case_sensitive=False) if gender_detector else None

def gender(name):
    n = re.sub(r"[^\wÀ-ž'’-]", '', name.strip(), flags=re.UNICODE)
    if not n: return 'unknown'
    v = custom.get(n.lower())
    if v in ('male','female','unknown'): return v
    if detector:
        g = detector.get_gender(n)
        if g in ('male','mostly_male'): return 'male'
        if g in ('female','mostly_female'): return 'female'
    return 'unknown'

def extract_names(text):
    # Mehrfach vorkommende Eigennamen: 2+ Wörter oder einzelne Namen mit Großbuchstaben.
    tokens = re.findall(r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]{2,}\b", text)
    stop = {'The','This','That','And','But','Then','When','Chapter','Chapter','Der','Die','Das','Ein','Eine','Und','Aber','Dann','Wenn'}
    return sorted({t for t in tokens if t not in stop})
