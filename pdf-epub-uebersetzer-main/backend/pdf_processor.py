"""
PDF-Verarbeitung ohne PyMuPDF (das auf Termux/Android nicht installierbar ist).

Strategie:
  - pdfplumber liest jede Seite: Textabsätze (mit exakter Position + Schrift-
    größe) und Bilder (mit Position).
  - Für jeden Textabsatz wird eine "Overlay-Seite" mit reportlab gebaut: an
    der exakt gleichen Position ein deckendes Rechteck (verdeckt Original-
    text) plus der übersetzte Text. Der Text wird mit reportlab.platypus
    (Paragraph/ParagraphStyle) automatisch umgebrochen, statt ihn bei zu
    langer Übersetzung nur extrem klein zu quetschen.
  - Für Bilder wird per Tesseract geprüft, ob es sich um ein "Text-Bild"
    handelt (Scan/Foto eines Dokuments, inkl. vertikalem Japanisch) oder ein
    echtes Foto/Diagramm:
      - Text-Bild -> Hintergrundfarbe schätzen, Bereich im Overlay füllen,
        übersetzten Text einsetzen.
      - Echtes Bild -> unangetastet lassen.
  - Die Overlay-Seite wird mit pypdf auf die ORIGINALSEITE gelegt (merge_page).
    Dadurch bleiben alle Originalgrafiken, Vektorelemente und echten Bilder
    exakt erhalten - nur an den Text-Stellen liegt jetzt die Übersetzung.
  - Mehrere Seiten werden PARALLEL verarbeitet (Mehrkern-CPU des Handys wird
    genutzt), da OCR und Übersetzung pro Seite unabhängig voneinander sind.
    Jede Seite öffnet dafür ihre eigenen Datei-Handles, damit kein gemeinsamer
    Zustand zwischen den Threads geteilt wird (Thread-Sicherheit).
"""

import io
import os
import re
import gc
import concurrent.futures
from dataclasses import dataclass
from collections import Counter
from xml.sax.saxutils import escape

import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import black
from PIL import Image
import pytesseract

from translator import translator, correct_grammar, normalize_source_text, TranslationError
from name_gender import gender, extract_names

# ---- Schwellenwerte für die Text-Bild-Erkennung ----
MIN_WORDS_FOR_TEXT_IMAGE = 15
MIN_AVG_CONFIDENCE = 60  # Tesseract-Konfidenz 0-100

# ---- Zeilen-Gruppierung ----
LINE_TOLERANCE = 3  # Punkte Abweichung, innerhalb derer Wörter als eine Zeile gelten

# ---- Absatz-Gruppierung ----
# Zeilen werden zu Absätzen zusammengefasst, damit ganze Sätze (die sich über
# mehrere gedruckte Zeilen erstrecken) als zusammenhängender Text übersetzt
# werden. Das verbessert die Grammatik der Übersetzung erheblich gegenüber
# einer reinen Zeile-für-Zeile-Übersetzung.
PARAGRAPH_GAP_FACTOR = 1.6  # Lückenfaktor: größerer Abstand als üblich = neuer Absatz

# ---- Batch-Übersetzung ----
# Statt jeden Absatz einzeln zu übersetzen (1 Netzwerk-Anfrage pro Absatz),
# werden mehrere Absätze mit einem Trennmarker zusammengefasst und in EINER
# Anfrage übersetzt. Das reduziert die Anzahl der Anfragen drastisch.
BATCH_SEPARATOR = "\n[[LBRK]]\n"
MAX_BATCH_CHARS = 1400  # Sicherheitsgrenze pro Anfrage - klein gehalten, damit
# jede Anfrage schnell bleibt. Größere Anfragen sind bei Googles kostenloser
# Schnittstelle spürbar langsamer und brechen öfter ab (was durch die Retry-
# Logik dann zusätzliche Wartezeit kostet).




# ---- OCR-Konfiguration pro Sprache ----
# jpn_vert = vertikal geschriebenes Japanisch (Spalten von oben nach unten,
# Spaltenreihenfolge rechts nach links) - Tesseract löst die Lesereihenfolge
# bei diesem Modus intern korrekt auf. --psm 5 = "einheitlicher Block aus
# vertikal ausgerichtetem Text", die empfohlene Einstellung dafür.
VERTICAL_OCR_LANGS = {"jpn_vert"}

# ---- Parallele Seitenverarbeitung ----
# Mehrere Seiten werden gleichzeitig verarbeitet, da OCR (externer Tesseract-
# Prozess) und Übersetzung (Netzwerk-Anfrage) beide die Python-GIL freigeben,
# während sie auf ein Ergebnis warten - echte Parallelität auf Mehrkern-CPUs.
# 3-4 ist ein guter Kompromiss: nutzt mehrere Kerne, ohne die kostenlose
# Übersetzungs-API mit zu vielen gleichzeitigen Anfragen zu überlasten.
MAX_WORKERS = int(os.environ.get("PDF_MAX_WORKERS", "2"))
# Standard jetzt 2 (für ein Tablet mit mehr Kernen/RAM als ein Handy
# üblicherweise vertretbar) - über die Umgebungsvariable ohne Code-Änderung
# anpassbar:
#   PDF_MAX_WORKERS=1 bash run.sh   (sicherer, falls 2 zu viel für das Gerät ist)
#   PDF_MAX_WORKERS=3 bash run.sh   (mehr Parallelität, mehr RAM/CPU-Bedarf)
# Während eine Seite auf die Google-Antwort wartet (Netzwerk-Latenz), kann
# ein zweiter Worker parallel schon die nächste Seite rendern/per OCR lesen -
# das überlappt Warte- und Rechenzeit, statt beides strikt nacheinander
# abzuarbeiten.

FONT_NAME = "Helvetica"


@dataclass
class ProcessingStats:
    pages_total: int = 0
    pages_done: int = 0
    pages_failed: int = 0
    text_blocks_translated: int = 0
    images_with_text: int = 0
    images_untouched: int = 0

def _name_aware_pronouns(source_text, translated_text, target_lang):
    if target_lang != "de" or not source_text or not translated_text:
        return translated_text
    result = translated_text
    for name in extract_names(source_text):
        g = gender(name)
        if g not in ("male", "female") or name not in result:
            continue
        wanted = "er" if g == "male" else "sie"
        other = "sie" if g == "male" else "er"
        sentences = re.split(r"(?<=[.!?])\s+", result)
        for i, sent in enumerate(sentences):
            if name in sent and re.search(r"\b(er|sie)\b", sent, re.I):
                sentences[i] = re.sub(r"\b" + other + r"\b", wanted, sent, count=1, flags=re.I)
        result = " ".join(sentences)
    return result


def _translate_lines_batched(texts, from_code, to_code):
    """Übersetzt eine Liste von Absätzen möglichst in wenigen Anfragen,
    statt eine Netzwerk-Anfrage pro Absatz zu senden. Fällt bei Problemen
    automatisch auf Einzelübersetzung zurück."""
    if not texts:
        return []

    texts = [normalize_source_text(t) for t in texts]
    results = [None] * len(texts)
    batch_indices, batch_texts, batch_chars = [], [], 0

    def flush_batch():
        if not batch_indices:
            return
        combined = BATCH_SEPARATOR.join(batch_texts)
        try:
            translated_combined = translator.translate(combined, from_code, to_code)
            parts = re.split(r"\s*\[\[LBRK\]\]\s*", translated_combined)
        except Exception:
            parts = []

        if len(parts) == len(batch_texts):
            for idx, part in zip(batch_indices, parts):
                results[idx] = normalize_source_text(part)
        else:
            # Der Marker kann von einem Übersetzer verändert/entfernt werden.
            # Dann wird JEDER Absatz separat übersetzt. Ein Fehler wird nicht
            # als "erfolgreich" behandelt.
            for idx, original in zip(batch_indices, batch_texts):
                results[idx] = translator.translate(original, from_code, to_code)

    for i, text in enumerate(texts):
        if batch_chars + len(text) > MAX_BATCH_CHARS and batch_indices:
            flush_batch()
            batch_indices, batch_texts, batch_chars = [], [], 0
        batch_indices.append(i)
        batch_texts.append(text)
        batch_chars += len(text) + len(BATCH_SEPARATOR)

    flush_batch()
    return results


def _group_words_into_lines(words):
    """Gruppiert pdfplumber-Wörter (mit x0,x1,top,bottom,size) zu Zeilen."""
    if not words:
        return []

    words = sorted(words, key=lambda w: (round(w["top"] / LINE_TOLERANCE), w["x0"]))
    lines = []
    current_line = [words[0]]

    for w in words[1:]:
        if abs(w["top"] - current_line[-1]["top"]) <= LINE_TOLERANCE:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
    lines.append(current_line)

    result = []
    for line_words in lines:
        text = " ".join(w["text"] for w in line_words)
        x0 = min(w["x0"] for w in line_words)
        x1 = max(w["x1"] for w in line_words)
        top = min(w["top"] for w in line_words)
        bottom = max(w["bottom"] for w in line_words)
        sizes = [w.get("size", 11) for w in line_words if w.get("size")]
        size = Counter(sizes).most_common(1)[0][0] if sizes else 11
        result.append({"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom, "size": size})

    return result


def _group_lines_into_paragraphs(lines):
    """Fasst aufeinanderfolgende Zeilen zu Absätzen zusammen, damit ganze
    Sätze (die sich über mehrere Zeilen erstrecken) als zusammenhängender
    Text übersetzt werden - wichtig für korrekte Grammatik."""
    if not lines:
        return []

    lines = sorted(lines, key=lambda l: l["top"])
    paragraphs = []
    current = [lines[0]]

    for prev, line in zip(lines, lines[1:]):
        line_height = prev["bottom"] - prev["top"] or 10
        gap = line["top"] - prev["bottom"]
        x_shift = abs(line["x0"] - prev["x0"])
        if gap > line_height * PARAGRAPH_GAP_FACTOR or x_shift > line_height * 3:
            paragraphs.append(current)
            current = [line]
        else:
            current.append(line)
    paragraphs.append(current)

    result = []
    for para_lines in paragraphs:
        text = " ".join(l["text"] for l in para_lines)
        x0 = min(l["x0"] for l in para_lines)
        x1 = max(l["x1"] for l in para_lines)
        top = min(l["top"] for l in para_lines)
        bottom = max(l["bottom"] for l in para_lines)
        sizes = [l["size"] for l in para_lines]
        size = Counter(sizes).most_common(1)[0][0] if sizes else 11
        result.append({"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom, "size": size})

    return result


def _sample_background_color(pil_image: Image.Image) -> tuple:
    """Schätzt die Hintergrundfarbe über die Randpixel des Bildes."""
    img = pil_image.convert("RGB")
    w, h = img.size
    edge_pixels = []
    for x in range(0, w, max(1, w // 20)):
        edge_pixels.append(img.getpixel((x, 0)))
        edge_pixels.append(img.getpixel((x, h - 1)))
    for y in range(0, h, max(1, h // 20)):
        edge_pixels.append(img.getpixel((0, y)))
        edge_pixels.append(img.getpixel((w - 1, y)))

    r = sum(p[0] for p in edge_pixels) / len(edge_pixels)
    g = sum(p[1] for p in edge_pixels) / len(edge_pixels)
    b = sum(p[2] for p in edge_pixels) / len(edge_pixels)
    return (r / 255, g / 255, b / 255)


def _is_text_image(pil_image: Image.Image, ocr_lang: str):
    """Prüft per Tesseract, ob ein Bildbereich überwiegend Text enthält."""
    # --oem 1 = nur die (schnellere) LSTM-Engine statt der Standard-Kombi aus
    # altem + neuem Erkennungsmodell - spart auf einem Handy-Prozessor
    # spürbar Zeit pro Aufruf, ohne bei normalem gedrucktem/fotografiertem
    # Text merklich an Genauigkeit zu verlieren.
    config = "--oem 1 --psm 5" if ocr_lang in VERTICAL_OCR_LANGS else "--oem 1"
    data = pytesseract.image_to_data(
        pil_image, lang=ocr_lang, config=config, output_type=pytesseract.Output.DICT
    )

    words, confidences = [], []
    for text, conf in zip(data["text"], data["conf"]):
        conf = int(conf) if str(conf).lstrip("-").isdigit() else -1
        if text.strip() and conf > 0:
            words.append(text)
            confidences.append(conf)

    if not words:
        return False, ""

    avg_conf = sum(confidences) / len(confidences)
    # Bei Japanisch zählt Tesseract oft einzelne Zeichen als "Wörter" ->
    # niedrigere Mindestanzahl, da schon wenige Zeichen viel Textinhalt sind
    min_words = 8 if ocr_lang in VERTICAL_OCR_LANGS else MIN_WORDS_FOR_TEXT_IMAGE
    is_text = len(words) >= min_words and avg_conf >= MIN_AVG_CONFIDENCE
    return is_text, " ".join(words)


def _draw_box(c, page_height, x0, top, x1, bottom, fill_color, text, font_size, min_size=6):
    """Zeichnet ein deckendes Rechteck an der gegebenen pdfplumber-Position
    (top/bottom von oben gemessen) auf einen reportlab-Canvas (y von unten
    gemessen) und setzt den Text mithilfe von reportlab.platypus.Paragraph
    hinein - dieser bricht lange Übersetzungen automatisch in mehrere Zeilen
    um, statt die Schrift nur extrem klein zu quetschen. Falls der Text bei
    der Startgröße nicht in die Box passt, wird die Schrift schrittweise
    verkleinert (bis zur Mindestgröße), wobei weiterhin normal umgebrochen
    werden darf."""
    rl_y = page_height - bottom
    rl_height = bottom - top
    width = max(1, x1 - x0)

    c.setFillColorRGB(*fill_color)
    c.rect(x0, rl_y, width, rl_height, fill=1, stroke=0)

    text = (text or "").strip()
    if not text:
        return

    # Für Paragraph als "XML-ähnliches" Markup sicher machen (& < > escapen),
    # damit Sonderzeichen aus der Übersetzung nicht als Markup interpretiert
    # werden und den Vorgang zum Absturz bringen.
    safe_text = escape(text.replace("\n", " "))

    size = font_size
    paragraph = None
    measured_height = 0

    while size >= min_size:
        style = ParagraphStyle(
            name="cell",
            fontName=FONT_NAME,
            fontSize=size,
            leading=size * 1.18,
            textColor=black,
        )
        paragraph = Paragraph(safe_text, style)
        _, measured_height = paragraph.wrap(width, rl_height * 50)
        if measured_height <= rl_height or size <= min_size:
            break
        size -= 0.5

    # Text oben in der Box beginnen lassen (statt zentriert/unten)
    draw_y = rl_y + rl_height - measured_height
    paragraph.drawOn(c, x0, draw_y)


PAGE_OCR_RESOLUTION = 110  # dpi für das Rendern der ganzen Seite bei Japanisch


def _process_vertical_page(plumber_page, c, page_width, page_height, from_code, to_code, ocr_lang, grammar_check, stats_delta):
    """Rendert die GESAMTE Seite als ein Bild und erfasst sie in einem Rutsch
    per Tesseract (vertikaler Modus). Die Position für die Übersetzung wird
    aus den vom OCR erkannten Wort-Koordinaten selbst berechnet (nicht aus
    pdfplumber's Text-/Bild-Erkennung, die bei vertikalem Japanisch unzuver-
    lässig ist - siehe Kommentar bei der Verwendung dieser Funktion)."""
    try:
        rendered = plumber_page.to_image(resolution=PAGE_OCR_RESOLUTION)
        pil_image = rendered.original
    except Exception as e:
        print(f"Seite konnte nicht gerendert werden ({e}), bleibt unverändert.")
        return

    config = "--psm 5"
    data = pytesseract.image_to_data(
        pil_image, lang=ocr_lang, config=config, output_type=pytesseract.Output.DICT
    )

    words, confidences = [], []
    lefts, tops, rights, bottoms = [], [], [], []
    for i, text in enumerate(data["text"]):
        conf_raw = data["conf"][i]
        conf = int(conf_raw) if str(conf_raw).lstrip("-").isdigit() else -1
        if text.strip() and conf > 0:
            words.append(text)
            confidences.append(conf)
            l, t, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            lefts.append(l)
            tops.append(t)
            rights.append(l + w)
            bottoms.append(t + h)

    if not words:
        del pil_image, rendered
        return  # keine erkennbare Textmenge - Seite bleibt unverändert (z.B. reine Illustration)

    avg_conf = sum(confidences) / len(confidences)
    if len(words) < 8 or avg_conf < MIN_AVG_CONFIDENCE:
        del pil_image, rendered
        return  # zu unsicher - lieber nichts verändern als Unsinn erzeugen

    # Pixel-Koordinaten (aus dem gerenderten Bild) in PDF-Punkte umrechnen
    scale = 72.0 / PAGE_OCR_RESOLUTION
    union_x0 = min(lefts) * scale
    union_top = min(tops) * scale
    union_x1 = max(rights) * scale
    union_bottom = max(bottoms) * scale

    ocr_text = " ".join(words)
    translated = translator.translate(ocr_text, from_code, to_code)
    if grammar_check and translated:
        translated = correct_grammar(translated, to_code)

    try:
        bg_color = _sample_background_color(pil_image)
    except Exception:
        bg_color = (1, 1, 1)

    _draw_box(
        c, page_height,
        union_x0, union_top, union_x1, union_bottom,
        fill_color=bg_color,
        text=translated,
        font_size=11,
    )
    stats_delta["images_with_text"] += 1
    del pil_image, rendered


def _process_single_page_inner(input_path, page_index, from_code, to_code, ocr_lang, grammar_check, pages_dir, force_ocr=False):
    """Verarbeitet GENAU EINE Seite komplett eigenständig: eigene pdfplumber-
    und pypdf-Datei-Handles, damit mehrere Seiten sicher gleichzeitig in
    unterschiedlichen Threads verarbeitet werden können, ohne sich gemeinsamen
    Zustand zu teilen (Thread-Sicherheit).

    WICHTIG für den Speicherverbrauch: Die fertige Seite wird SOFORT als
    eigene kleine PDF-Datei auf die Festplatte geschrieben und NICHT als
    Objekt zurückgegeben. Ein einzelner, über die ganze Übersetzung hinweg
    wachsender PDF-Writer im Arbeitsspeicher war der wahrscheinlichste Grund
    für Abstürze bei längeren Dokumenten (Android beendet die App bei zu
    hohem Speicherverbrauch hart) - dieser Ansatz hält den Speicherbedarf
    während der gesamten Übersetzung durchgehend niedrig, unabhängig davon
    wie lang das Dokument ist.

    Wirft bei Fehlern ganz normal eine Exception - das Abfangen und der
    Original-Fallback passieren bewusst NICHT hier, sondern im äußeren
    Wrapper _process_single_page() weiter unten (siehe dort)."""
    stats_delta = {"text_blocks_translated": 0, "images_with_text": 0, "images_untouched": 0, "failed": False}

    with pdfplumber.open(input_path) as pdf_local:
        plumber_page = pdf_local.pages[page_index]
        page_width = float(plumber_page.width)
        page_height = float(plumber_page.height)

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(page_width, page_height))

        # ---- Native Textebene IMMER bevorzugen ----
        # Besonders wichtig für japanische PDFs: Wenn die PDF bereits eine
        # echte Textebene enthält, darf sie NICHT durch teures Vollseiten-OCR
        # ersetzt werden. OCR ist nur ein Fallback für echte Scan-/Bildseiten.
        words = plumber_page.extract_words(extra_attrs=["size", "fontname"])
        # Web-PDFs aus japanischen Buchseiten verwenden häufig CJK-
        # Kompatibilitätszeichen (z.B. ⾯ statt 面). Normalisieren, bevor
        # Zeilen/Absätze gebildet und Übersetzungsanfragen erstellt werden.
        for w in words:
            w["text"] = normalize_source_text(w.get("text", ""))
        words = [w for w in words if w["text"]]
        page_has_native_text = len(words) > 0

        if (ocr_lang in VERTICAL_OCR_LANGS) and not page_has_native_text:
            # Vertikales Japanisch als Scan/Bild: erst jetzt Vollseiten-OCR.
            # Eine vorhandene Textebene wird niemals unnötig gerendert.
            _process_vertical_page(
                plumber_page, c, page_width, page_height,
                from_code, to_code, ocr_lang, grammar_check, stats_delta,
            )
            words = []
            page_has_native_text = False

        # ---- Text übersetzen: zu Absätzen gruppiert (bessere Grammatik) ----
        if force_ocr and not page_has_native_text and ocr_lang not in VERTICAL_OCR_LANGS:
            try:
                rendered = plumber_page.to_image(resolution=110)
                pil = rendered.original
                is_text_image, ocr_text = _is_text_image(pil, ocr_lang)
                if is_text_image and ocr_text:
                    translated = translator.translate(ocr_text, from_code, to_code)
                    translated = _name_aware_pronouns(ocr_text, translated, to_code)
                    if grammar_check and translated:
                        translated = correct_grammar(translated, to_code)
                    _draw_box(c, page_height, 0, 0, page_width, page_height, (1,1,1), translated, 11)
                    stats_delta["text_blocks_translated"] += 1
                    del pil, rendered
                    words = []
                    page_has_native_text = True
            except Exception as e:
                print(f"Vollseiten-OCR fehlgeschlagen Seite {page_index+1}: {e}")

        lines = [l for l in _group_words_into_lines(words) if l["text"].strip()]
        paragraphs = _group_lines_into_paragraphs(lines)

        translated_texts = _translate_lines_batched(
            [p["text"].strip() for p in paragraphs], from_code, to_code
        )

        for para, translated in zip(paragraphs, translated_texts):
            final_text = _name_aware_pronouns(para["text"], translated or "", to_code)
            if grammar_check and final_text:
                final_text = correct_grammar(final_text, to_code)
            _draw_box(
                c, page_height,
                para["x0"], para["top"], para["x1"], para["bottom"],
                fill_color=(1, 1, 1),
                text=final_text,
                font_size=para["size"],
            )
            stats_delta["text_blocks_translated"] += 1

        # ---- Bilder prüfen: Text-Bild oder echtes Bild? ----
        for img in plumber_page.images:
            x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
            if x1 <= x0 or bottom <= top:
                continue

            # Winzige Zierelemente (Trennlinien, Icons, kleine Logos)
            # überspringen: ein Bereich unter ~25x25pt kann ohnehin kein
            # sinnvoll lesbares Wort enthalten, aber der Tesseract-Aufruf
            # dafür kostet auf einem Handy-Prozessor trotzdem volle Zeit.
            # Bei Seiten mit vielen kleinen Deko-Bildern spart das
            # spürbar Gesamtzeit, ohne echte Textbilder zu verpassen.
            MIN_TEXT_IMAGE_SIZE_PT = 25
            if (x1 - x0) < MIN_TEXT_IMAGE_SIZE_PT or (bottom - top) < MIN_TEXT_IMAGE_SIZE_PT:
                stats_delta["images_untouched"] += 1
                continue

            if page_has_native_text:
                # Seite hat schon "echten" Text - Bild NUR noch dann per
                # OCR prüfen, wenn es groß genug ist, um selbst eine
                # eigene fotografierte/eingescannte Textseite zu sein
                # (z.B. eine eingescannte Grafik/Tafel innerhalb eines
                # sonst digitalen Buchs). Kleinere Bilder auf einer
                # Text-Seite sind so gut wie nie zusätzlicher lesbarer
                # Fließtext, sondern Illustrationen/Fotos/Diagramme -
                # für die lohnt sich der teure OCR-Aufruf nicht.
                img_area_fraction = ((x1 - x0) * (bottom - top)) / (page_width * page_height)
                LARGE_IMAGE_AREA_THRESHOLD = 0.4  # >= 40% der Seitenfläche
                if img_area_fraction < LARGE_IMAGE_AREA_THRESHOLD:
                    stats_delta["images_untouched"] += 1
                    continue

            try:
                cropped = plumber_page.within_bbox((x0, top, x1, bottom)).to_image(resolution=110)
                pil_image = cropped.original
            except Exception:
                continue

            is_text_image, ocr_text = _is_text_image(pil_image, ocr_lang)

            if is_text_image:
                translated = translator.translate(ocr_text, from_code, to_code)
                translated = _name_aware_pronouns(ocr_text, translated, to_code)
                if grammar_check and translated:
                    translated = correct_grammar(translated, to_code)
                bg_color = _sample_background_color(pil_image)
                _draw_box(
                    c, page_height,
                    x0, top, x1, bottom,
                    fill_color=bg_color,
                    text=translated,
                    font_size=10,
                )
                stats_delta["images_with_text"] += 1
            else:
                stats_delta["images_untouched"] += 1
            del pil_image, cropped

        c.showPage()
        c.save()
        buf.seek(0)

    # Eigener PdfReader für diesen Thread (nicht mit anderen Threads geteilt)
    reader_local = PdfReader(input_path)
    original_page = reader_local.pages[page_index]
    try:
        overlay_reader = PdfReader(buf)
        if len(overlay_reader.pages) > 0:
            # WICHTIG: over=True explizit setzen. Ohne dieses Argument war
            # (je nach installierter pypdf-Version) nicht garantiert, dass
            # der Overlay (deckende Box + übersetzter Text) ÜBER der
            # Originalseite liegt - lag er darunter, blieb der alte,
            # unübersetzte Text sichtbar obwohl die Übersetzung technisch
            # längst passiert war. Das war Ursache für "Download fertig,
            # aber PDF sieht unübersetzt aus".
            original_page.merge_page(overlay_reader.pages[0], over=True)
    except Exception as overlay_error:
        print(
            f"Hinweis: Overlay für Seite {page_index + 1} übersprungen "
            f"({overlay_error}), Originalseite wird unverändert übernommen."
        )

    # Seite SOFORT als eigene kleine Datei sichern (nicht im Speicher behalten)
    page_writer = PdfWriter()
    written_page = page_writer.add_page(original_page)
    # compress_content_streams() muss NACH add_page() aufgerufen werden - die
    # Methode verlangt, dass die Seite bereits Teil eines PdfWriter ist (sonst
    # AttributeError). Das ist der entscheidende, bisher fehlende Schritt:
    # merge_page() dekomprimiert die Content-Streams beim Zusammenführen und
    # lässt sie unkomprimiert zurück. Ohne diesen Aufruf wächst die Datei über
    # viele Seiten hinweg dramatisch (in Tests: >11x größer) und das finale
    # Schreiben wird spürbar langsamer, je mehr Seiten bereits verarbeitet
    # wurden - das war die Ursache für den "Hänger am Ende" und die
    # aufgeblähte Dateigröße (80 MB statt 10-15 MB).
    try:
        written_page.compress_content_streams()
    except Exception as compress_error:
        print(f"Hinweis: Kompression für Seite {page_index + 1} übersprungen ({compress_error}).")

    page_file = os.path.join(pages_dir, f"page_{page_index:05d}.pdf")
    with open(page_file, "wb") as f:
        page_writer.write(f)

    # Alles Schwere explizit aus dem Speicher entlassen, statt auf den
    # (langsameren) automatischen Garbage-Collector zu warten
    del original_page, reader_local, page_writer, buf, c
    gc.collect()

    return page_index, stats_delta


def _process_single_page(input_path, page_index, from_code, to_code, ocr_lang, grammar_check, pages_dir, force_ocr=False):
    """Dünner Wrapper um _process_single_page_inner(): fängt JEDE Exception
    ab, die bei der Verarbeitung einer einzelnen Seite auftreten kann (z.B.
    ein beschädigtes Bild, ein Encoding-Sonderfall, ein OCR-Absturz auf
    genau dieser einen Seite). Eine fehlschlagende Seite darf niemals den
    gesamten Übersetzungsjob mitreißen - das war vorher der Fall, weil eine
    Exception aus einem Worker-Thread über future.result() bis nach oben
    durchgereicht wurde und dort den kompletten Job auf "error" gesetzt hat,
    selbst wenn nur eine einzige von z.B. 300 Seiten betroffen war.

    Im Fehlerfall wird die Originalseite UNVERÄNDERT (unübersetzt) ins
    Zwischenverzeichnis geschrieben, damit die finale PDF trotzdem
    vollständig bleibt - nur eben an dieser einen Stelle unübersetzt statt
    komplett zu fehlen."""
    try:
        return _process_single_page_inner(
            input_path, page_index, from_code, to_code, ocr_lang,
            grammar_check, pages_dir, force_ocr,
        )
    except Exception as e:
        print(
            f"FEHLER bei Seite {page_index + 1}: {e} - Seite wird unverändert "
            f"(unübersetzt) übernommen, der restliche Job läuft normal weiter."
        )
        try:
            reader_local = PdfReader(input_path)
            fallback_writer = PdfWriter()
            fallback_writer.add_page(reader_local.pages[page_index])
            page_file = os.path.join(pages_dir, f"page_{page_index:05d}.pdf")
            with open(page_file, "wb") as f:
                fallback_writer.write(f)
        except Exception as fallback_error:
            # Selbst das unveränderte Kopieren ist fehlgeschlagen (z.B. weil
            # die Originaldatei an dieser Stelle selbst beschädigt ist) -
            # dann fehlt diese eine Seite im Endergebnis, aber der Job läuft
            # für alle anderen Seiten trotzdem normal weiter statt
            # komplett abzubrechen.
            print(f"FEHLER: Auch Original-Kopie für Seite {page_index + 1} fehlgeschlagen: {fallback_error}")

        return page_index, {
            "text_blocks_translated": 0,
            "images_with_text": 0,
            "images_untouched": 0,
            "failed": True,
        }


def assemble_pdf(pages_dir: str, output_path: str) -> int:
    """Fügt alle bisher fertigen Einzelseiten-Dateien zu EINER PDF zusammen.
    Kann jederzeit aufgerufen werden - auch während die Übersetzung noch
    läuft (liefert dann den aktuellen Teilstand) oder nach einer Unter-
    brechung (liefert die bis dahin fertigen Seiten). Gibt die Anzahl der
    zusammengefügten Seiten zurück."""
    if not os.path.isdir(pages_dir):
        return 0

    page_files = sorted(
        f for f in os.listdir(pages_dir) if f.startswith("page_") and f.endswith(".pdf")
    )
    if not page_files:
        return 0

    writer = PdfWriter()
    for filename in page_files:
        reader = PdfReader(os.path.join(pages_dir, filename))
        if reader.pages:
            writer.add_page(reader.pages[0])

    # Jede Einzelseiten-Datei wurde unabhängig geschrieben und bringt daher
    # ihre eigene Kopie z.B. der im Original-PDF eingebetteten Schriftart(en)
    # mit - beim Zusammenfügen von z.B. 300 Seiten ergibt das 300 Kopien
    # derselben Schriftart. compress_identical_objects() erkennt inhaltlich
    # identische Objekte und führt sie zu einer einzigen Kopie zusammen. Das
    # ist neben der Content-Stream-Kompression (siehe _process_single_page)
    # der zweite große Hebel gegen die aufgeblähte Dateigröße.
    try:
        writer.compress_identical_objects(remove_orphans=True)
    except Exception as e:
        print(f"Hinweis: Objekt-Deduplizierung übersprungen ({e}).")

    with open(output_path, "wb") as f:
        writer.write(f)

    return len(page_files)


def _sample_document_for_ocr(input_path, ocr_lang, sample_pages=15):
    """Prüft bis zu 15 Seiten, bevor OCR als Dokumentmodus aktiviert wird.

    Wichtig: Zuerst wird auf ALLEN bis zu 15 Stichprobenseiten nach einer
    nativen Textebene gesucht. Das verhindert, dass ein normales japanisches
    Text-PDF unnötig 15x gerendert und per Tesseract analysiert wird.
    OCR wird nur als Fallback betrachtet, wenn die Stichprobe überwiegend
    keine extrahierbare Textebene besitzt.
    """
    try:
        with pdfplumber.open(input_path) as pdf:
            total = min(len(pdf.pages), sample_pages)
            if total == 0:
                return False

            native_text_pages = 0
            for page_number, page in enumerate(pdf.pages[:total], 1):
                try:
                    words = page.extract_words()
                    if len(words) >= 8:
                        native_text_pages += 1
                except Exception as e:
                    print(f"Textanalyse Seite {page_number}/{total} fehlgeschlagen: {e}")

            # Bereits nach der vollständigen 15-Seiten-Stichprobe entscheiden.
            # Bei nativen Text-PDFs KEIN OCR-Vorlauf.
            if native_text_pages >= max(1, int(total * 0.60)):
                print(
                    f"Textanalyse: {native_text_pages}/{total} Stichprobenseiten "
                    f"enthalten native Textebene -> OCR-Vorlauf deaktiviert."
                )
                return False

            # Nur wenn die Stichprobe überwiegend scan-/bildbasiert ist,
            # zusätzlich einige Seiten per OCR prüfen.
            ocr_text_pages = 0
            for page_number, page in enumerate(pdf.pages[:total], 1):
                try:
                    img = page.to_image(resolution=70).original
                    ok, txt = _is_text_image(img, ocr_lang)
                    if ok and txt:
                        ocr_text_pages += 1
                    del img
                except Exception as e:
                    print(f"OCR-Stichprobe Seite {page_number}/{total} fehlgeschlagen: {e}")

            force = ocr_text_pages >= max(3, int(total * 0.60))
            print(
                f"OCR-Stichprobe: {ocr_text_pages}/{total} Seiten mit erkanntem Text "
                f"-> Vollseiten-OCR={'an' if force else 'aus'}."
            )
            return force
    except Exception as e:
        print(f"OCR-Stichprobe fehlgeschlagen: {e}")
        return False

def process_pdf(
    input_path: str,
    output_path: str,
    from_code: str,
    to_code: str,
    ocr_lang: str = "deu",
    grammar_check: bool = False,
    pages_dir: str = None,
    progress_callback=None,
    cancel_event=None,
) -> ProcessingStats:
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    stats = ProcessingStats(pages_total=total_pages)

    force_ocr = _sample_document_for_ocr(input_path, ocr_lang, sample_pages=15)
    print(f"OCR-Modus nach 15-Seiten-Prüfung: {force_ocr}")

    if pages_dir is None:
        pages_dir = output_path + "_pages"
    os.makedirs(pages_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i in range(total_pages):
            existing_page = os.path.join(pages_dir, f"page_{i:05d}.pdf")
            if os.path.exists(existing_page):
                stats.pages_done += 1
                if progress_callback:
                    progress_callback(stats)
                continue
            # Vor jeder neuen Seite prüfen, ob der Nutzer inzwischen über den
            # "X"-Button abgebrochen hat - dann werden keine weiteren Seiten
            # mehr eingeplant. Bereits laufende Seiten werden noch fertig
            # (Netzwerk-/OCR-Aufrufe lassen sich nicht sauber mitten in der
            # Ausführung unterbrechen), aber es wird nichts Neues mehr
            # gestartet.
            if cancel_event is not None and cancel_event.is_set():
                break
            future = executor.submit(
                _process_single_page, input_path, i, from_code, to_code,
                ocr_lang, grammar_check, pages_dir, force_ocr,
            )
            futures[future] = i

        for future in concurrent.futures.as_completed(futures):
            try:
                page_index, stats_delta = future.result()
            except Exception as e:
                # Zusätzliches Sicherheitsnetz: _process_single_page() fängt
                # eigentlich schon alles ab, aber falls doch einmal etwas
                # unerwartet durchrutscht, darf das trotzdem nicht den
                # gesamten Job stoppen.
                print(f"FEHLER: Unerwarteter Fehler in einem Seiten-Worker: {e}")
                stats.pages_failed += 1
                stats.pages_done += 1
                if progress_callback:
                    progress_callback(stats)
                continue

            stats.text_blocks_translated += stats_delta["text_blocks_translated"]
            stats.images_with_text += stats_delta["images_with_text"]
            stats.images_untouched += stats_delta["images_untouched"]
            if stats_delta.get("failed"):
                stats.pages_failed += 1
            stats.pages_done += 1

            if progress_callback:
                progress_callback(stats)

    if cancel_event is not None and cancel_event.is_set():
        # Abgebrochen: trotzdem die bisher fertigen Seiten zu einer
        # (Teil-)PDF zusammenbauen, damit der Nutzer sie noch bekommen kann,
        # falls er sich dafür entscheidet.
        assemble_pdf(pages_dir, output_path)
        return stats

    # Erst jetzt, am Ende, alle Einzelseiten zu einer PDF zusammenfügen -
    # dieser Schritt braucht kurzzeitig mehr Speicher, aber nur EINMAL ganz
    # am Ende, nicht durchgehend über die gesamte (oft mehrere Minuten
    # dauernde) Übersetzung hinweg. Das ist der entscheidende Unterschied
    # zur vorherigen Version, die die Abstürze verursacht hat.
    assemble_pdf(pages_dir, output_path)

    return stats


def get_pdf_page_count(input_path: str) -> int:
    """Liest nur die Seitenzahl, ohne die Datei sonst zu verändern - für die
    Splitter-Vorschau im Frontend (weiß der Nutzer, bis zu welcher Seite er
    maximal wählen kann)."""
    return len(PdfReader(input_path).pages)


def split_pdf(input_path: str, output_path: str, start_page: int, end_page: int) -> int:
    """Erstellt eine neue PDF, die NUR die Seiten von start_page bis
    end_page (beide inklusive, 1-indexiert - also wie im Alltag gezählt,
    'Seite 3' ist wirklich die dritte sichtbare Seite) aus der Originaldatei
    enthält. Reine Kopie der Originalseiten (keine Übersetzung, kein
    Rendering) - deshalb schnell und verlustfrei, inklusive aller Bilder,
    Vektorgrafiken und eingebetteten Schriften."""
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)

    if start_page < 1 or end_page < start_page or start_page > total_pages:
        raise ValueError(
            f"Ungültiger Seitenbereich {start_page}-{end_page} "
            f"(Dokument hat {total_pages} Seiten)."
        )
    end_page = min(end_page, total_pages)

    writer = PdfWriter()
    for i in range(start_page - 1, end_page):  # intern 0-indexiert
        writer.add_page(reader.pages[i])

    with open(output_path, "wb") as f:
        writer.write(f)

    return end_page - start_page + 1
