"""
EPUB-Übersetzung. Nutzt ausschließlich die Python-Standardbibliothek
(zipfile, xml.etree.ElementTree) - bewusst KEIN neues pip-Paket, nach den
Erfahrungen mit Kompilier-Problemen bei anderen Bibliotheken auf Termux.

Ein EPUB ist im Kern ein ZIP-Archiv aus XHTML-Dateien + Metadaten. Anders als
bei PDFs ist hier KEIN OCR nötig - der Text ist bereits "echt" vorhanden,
in der korrekten logischen Lesereihenfolge (auch bei japanischen eBooks mit
vertikaler Schreibrichtung, die per CSS gesteuert wird - die zugrunde
liegende HTML-Reihenfolge stimmt trotzdem).

Strategie:
  - container.xml verweist auf die .opf-Datei (Inhaltsverzeichnis).
  - Aus der .opf-Datei wird die Liste aller XHTML-Inhaltsdateien gelesen.
  - Für jede Datei wird jedes Block-Element (Absatz, Überschrift, Listenpunkt
    etc.) als EIN zusammenhängender Text übersetzt - wichtig für korrekte
    Grammatik, genau wie bei der PDF-Übersetzung.
  - Furigana-Lesehilfen (<rt>-Tags bei japanischen eBooks) werden beim
    Extrahieren übersprungen, damit sie nicht sinnlos mitübersetzt werden.
  - Bilder, CSS, Schriften, Kapitelstruktur und alle Metadaten bleiben
    unverändert - nur der sichtbare Fließtext wird ersetzt.
"""

import os
import re
import zipfile
import concurrent.futures
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from translator import translator, correct_grammar

NAMESPACES = {
    "xhtml": "http://www.w3.org/1999/xhtml",
    "opf": "http://www.idpf.org/2007/opf",
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
}

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "td", "th", "figcaption", "dd", "dt",
}

# Tags, deren Textinhalt NICHT mitübersetzt werden soll
SKIP_TEXT_TAGS = {"rt", "rp", "script", "style"}

BATCH_SEPARATOR = "\n[[LBRK]]\n"
MAX_BATCH_CHARS = 1400
# Von 700 auf 2500 angehoben: weniger, dafür größere Anfragen an Google.
# Das SPART zwar Requests, macht aber jeden einzelnen davon größer und damit
# bei einem Fehlschlag teurer (mehr Text muss im Fallback einzeln erneut
# angefragt werden) - siehe Fallback-Logik in flush_batch() unten.

TRANSLATE_BATCH_WORKERS = 1
# Wie viele Batches gleichzeitig an translator.translate() übergeben werden.
# Wichtig: das ist NICHT dasselbe wie "gleichzeitige Google-Anfragen" - die
# eigentliche Drosselung auf maximal TRANSLATE_CONCURRENCY (=6, in
# translator.py) gleichzeitige echte Netzwerk-Anfragen passiert bereits
# GLOBAL über alle Dateien/Jobs hinweg per Semaphore. Diese 4 Worker hier
# sorgen nur dafür, dass innerhalb EINER EPUB-Datei mehrere Batches parallel
# an der Warteschlange der Semaphore anstehen, statt strikt nacheinander
# eine Netzwerk-Rundreise nach der anderen abzuwarten.


@dataclass
class EpubStats:
    files_total: int = 0
    files_done: int = 0
    blocks_translated: int = 0


def _translate_batched(texts, from_code, to_code):
    """Gleiche Batch-Logik wie bei der PDF-Übersetzung: mehrere Textblöcke
    pro Anfrage, um Netzwerk-Anfragen zu sparen. Die Batches selbst werden
    parallel (statt nacheinander) verarbeitet, um Netzwerk-Latenzen zu
    überlappen - die eigentliche Google-Anfragerate wird weiterhin zentral
    von der Semaphore in translator.py begrenzt, hier wird also nichts
    "überlastet", nur die Wartezeit besser genutzt."""
    if not texts:
        return []

    results = [None] * len(texts)

    # ---- Schritt 1: Texte in Batches aufteilen (rein lokal, kein Netzwerk) ----
    batches = []  # Liste von (batch_indices, batch_texts)
    batch_indices, batch_texts, batch_chars = [], [], 0
    for i, text in enumerate(texts):
        if batch_chars + len(text) > MAX_BATCH_CHARS and batch_indices:
            batches.append((batch_indices, batch_texts))
            batch_indices, batch_texts, batch_chars = [], [], 0
        batch_indices.append(i)
        batch_texts.append(text)
        batch_chars += len(text) + len(BATCH_SEPARATOR)
    if batch_indices:
        batches.append((batch_indices, batch_texts))

    def translate_one_batch(batch_indices, batch_texts):
        """Übersetzt GENAU EINEN Batch und schreibt die Ergebnisse direkt an
        die richtigen Original-Positionen in `results`. Läuft in einem
        Worker-Thread - `results` ist eine Liste mit fester Länge, in die
        jeder Batch garantiert nur an seinen EIGENEN (vorher fest zugeteilten)
        Indizes schreibt. Da sich verschiedene Batches nie Indizes teilen,
        ist kein Lock nötig - klassisches "jeder Thread hat seinen eigenen
        Bereich"-Muster, das schreibsicher ist, ohne dass wir es explizit
        absichern müssen."""
        try:
            combined = BATCH_SEPARATOR.join(batch_texts)
            translated_combined = translator.translate(combined, from_code, to_code)
            parts = re.split(r"\s*\[\[LBRK\]\]\s*", translated_combined)
            if len(parts) == len(batch_texts):
                for idx, part in zip(batch_indices, parts):
                    results[idx] = part.strip()
                return
            # Trenner ist beim Übersetzen verlorengegangen/verschoben worden
            # (z.B. weil Google ihn mitübersetzt oder Absätze zusammengelegt
            # hat) - Fallback: dieser eine Batch wird nochmal einzeln,
            # Text für Text angefragt statt gebündelt.
            raise ValueError(
                f"Batch-Trenner nicht wiederhergestellt "
                f"({len(parts)} Teile statt {len(batch_texts)} erwartet)"
            )
        except Exception as e:
            print(
                f"Hinweis: Batch-Übersetzung fehlgeschlagen "
                f"({len(batch_texts)} Blöcke, {e}) - falle auf Einzelanfragen "
                f"für diesen Batch zurück."
            )
            for idx, text in zip(batch_indices, batch_texts):
                try:
                    results[idx] = translator.translate(text, from_code, to_code)
                except Exception as single_error:
                    # Letzte Verteidigungslinie: lieber der unübersetzte
                    # Originaltext als ein abgestürzter Gesamtjob.
                    print(f"Hinweis: Einzelanfrage fehlgeschlagen ({single_error}) - Originaltext wird beibehalten.")
                    results[idx] = text

    # ---- Schritt 2: Batches parallel abarbeiten ----
    if len(batches) <= 1:
        # Bei nur einem Batch lohnt sich kein Thread-Pool-Overhead.
        for b_indices, b_texts in batches:
            translate_one_batch(b_indices, b_texts)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=TRANSLATE_BATCH_WORKERS) as executor:
        futures = [executor.submit(translate_one_batch, b_indices, b_texts) for b_indices, b_texts in batches]
        # .result() abrufen (auch wenn kein Rückgabewert erwartet wird): so
        # werden unerwartete Exceptions aus dem Worker-Thread hier im
        # Haupt-Thread sichtbar/geloggt statt lautlos verschluckt zu werden.
        for future in concurrent.futures.as_completed(futures):
            future.result()

    return results


def _local_tag(elem):
    """Tag-Name ohne Namespace-Präfix, z.B. '{http://...}p' -> 'p'"""
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def _extract_block_text(elem):
    """Sammelt den sichtbaren Text eines Block-Elements (ohne Furigana-
    Lesehilfen), rekursiv über alle Kind-Elemente hinweg."""
    parts = []

    def walk(e):
        if _local_tag(e) in SKIP_TEXT_TAGS:
            return
        if e.text:
            parts.append(e.text)
        for child in e:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    if elem.text:
        parts.append(elem.text)
    for child in elem:
        walk(child)
        if child.tail:
            parts.append(child.tail)

    return " ".join(p.strip() for p in parts if p and p.strip())


def _translate_block_preserve_markup(elem, from_code, to_code):
    """Übersetzt Textknoten innerhalb eines Blocks, ohne <strong>, <em>,
    <a>, <ruby> usw. zu löschen. Dadurch bleibt Inline-Formatierung erhalten."""
    nodes = []
    def walk(e):
        if _local_tag(e) in SKIP_TEXT_TAGS:
            return
        if e.text and e.text.strip():
            nodes.append((e, "text", e.text))
        for child in e:
            walk(child)
            if child.tail and child.tail.strip():
                nodes.append((child, "tail", child.tail))
    walk(elem)
    if not nodes:
        return 0
    texts=[n[2] for n in nodes]
    translated=_translate_batched(texts, from_code, to_code)
    for (node, kind, _), value in zip(nodes, translated):
        if kind == "text": node.text=value
        else: node.tail=value
    return len(nodes)


def _translate_xhtml_bytes(content_bytes, from_code, to_code, grammar_check):
    try:
        root = ET.fromstring(content_bytes)
    except ET.ParseError as e:
        print(f"XHTML-Datei konnte nicht geparst werden, bleibt unverändert: {e}")
        return content_bytes, 0

    ET.register_namespace("", NAMESPACES["xhtml"])

    block_elements = [elem for elem in root.iter() if _local_tag(elem) in BLOCK_TAGS]
    count = 0
    for elem in block_elements:
        try:
            count += _translate_block_preserve_markup(elem, from_code, to_code)
        except Exception as e:
            print(f"EPUB-Block konnte nicht übersetzt werden: {e}")
    if grammar_check:
        # Grammatik-Politur bleibt optional; die Struktur/Tags werden dabei
        # bewusst nicht neu aufgebaut.
        pass
    output = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return output, count


def _find_opf_path(zin):
    container_xml = zin.read("META-INF/container.xml")
    root = ET.fromstring(container_xml)
    rootfile = root.find(".//container:rootfile", NAMESPACES)
    return rootfile.attrib["full-path"]


def _get_content_files(zin, opf_path):
    """Liste aller XHTML-Inhaltsdateien aus dem Manifest (Pfade relativ zum
    EPUB-Root), in der Reihenfolge, in der sie im Manifest stehen."""
    opf_dir = os.path.dirname(opf_path)
    opf_data = zin.read(opf_path)
    root = ET.fromstring(opf_data)
    manifest = root.find("opf:manifest", NAMESPACES)

    files = []
    for item in manifest.findall("opf:item", NAMESPACES):
        media_type = item.attrib.get("media-type", "")
        if media_type in ("application/xhtml+xml", "text/html"):
            href = item.attrib["href"]
            full_path = os.path.normpath(os.path.join(opf_dir, href)).replace("\\", "/")
            files.append(full_path)
    return files


def process_epub(
    input_path: str,
    output_path: str,
    from_code: str,
    to_code: str,
    grammar_check: bool = False,
    progress_callback=None,
    cancel_event=None,
) -> EpubStats:
    with zipfile.ZipFile(input_path, "r") as zin:
        opf_path = _find_opf_path(zin)
        content_files = _get_content_files(zin, opf_path)
        stats = EpubStats(files_total=len(content_files))

        all_names = zin.namelist()
        translated_content = {}

        for filename in content_files:
            if cancel_event is not None and cancel_event.is_set():
                # Abgebrochen: EPUB kann (anders als PDF) nicht sinnvoll
                # teilweise zusammengebaut werden, da die ZIP-Datei erst
                # ganz am Ende in einem Rutsch geschrieben wird. Wir brechen
                # hier einfach ab, der Aufrufer (main.py) markiert den Job
                # dann als abgebrochen.
                return stats
            try:
                original_bytes = zin.read(filename)
                new_bytes, block_count = _translate_xhtml_bytes(
                    original_bytes, from_code, to_code, grammar_check
                )
                translated_content[filename] = new_bytes
                stats.blocks_translated += block_count
            except Exception as e:
                print(f"Datei {filename} konnte nicht übersetzt werden, bleibt unverändert: {e}")

            stats.files_done += 1
            if progress_callback:
                progress_callback(stats)

        with zipfile.ZipFile(output_path, "w") as zout:
            # mimetype MUSS als allererste Datei und UNKOMPRIMIERT gespeichert
            # werden - das schreibt der EPUB-Standard so vor
            if "mimetype" in all_names:
                zout.writestr("mimetype", zin.read("mimetype"), zipfile.ZIP_STORED)

            for name in all_names:
                if name == "mimetype":
                    continue
                data = translated_content.get(name, zin.read(name))
                zout.writestr(name, data, zipfile.ZIP_DEFLATED)

    return stats


def _get_spine_files(zin, opf_path):
    """Liste der Inhaltsdateien in der TATSÄCHLICHEN Lesereihenfolge (Spine),
    nicht der (oft anderen) Reihenfolge im Manifest. Für den Splitter ist das
    wichtig - 'Abschnitt 3 bis 15' muss sich auf die Reihenfolge beziehen, in
    der ein E-Reader die Kapitel tatsächlich anzeigt.
    Rückgabe: Liste von (idref, Dateipfad)-Paaren."""
    opf_dir = os.path.dirname(opf_path)
    opf_data = zin.read(opf_path)
    root = ET.fromstring(opf_data)

    manifest = root.find("opf:manifest", NAMESPACES)
    id_to_href = {item.attrib["id"]: item.attrib["href"] for item in manifest.findall("opf:item", NAMESPACES)}

    spine = root.find("opf:spine", NAMESPACES)
    files = []
    for itemref in spine.findall("opf:itemref", NAMESPACES):
        idref = itemref.attrib.get("idref")
        href = id_to_href.get(idref)
        if href:
            full_path = os.path.normpath(os.path.join(opf_dir, href)).replace("\\", "/")
            files.append((idref, full_path))
    return files


def get_epub_section_count(input_path: str) -> int:
    """Anzahl der Kapitel-/Abschnittsdateien in Lesereihenfolge (Spine) - für
    die Splitter-Vorschau im Frontend. EPUB kennt anders als PDF keine festen
    Bildschirmseiten (der Text fließt je nach Gerät/Schriftgröße neu), daher
    zählen wir hier Abschnitte statt Seiten."""
    with zipfile.ZipFile(input_path, "r") as zin:
        opf_path = _find_opf_path(zin)
        return len(_get_spine_files(zin, opf_path))


def split_epub(input_path: str, output_path: str, start_index: int, end_index: int) -> int:
    """Erstellt ein neues EPUB, das nur die Abschnitte (Kapitel-Dateien) von
    start_index bis end_index (1-indexiert, inklusive, in Spine-Lesereihen-
    folge) enthält. Alle Ressourcen (Bilder, CSS, Schriften, Metadaten)
    bleiben unangetastet im Archiv erhalten - nur die Spine (das eigentliche
    "Inhaltsverzeichnis" der Lesereihenfolge) wird auf die gewählten
    Abschnitte eingegrenzt, sodass ein E-Reader nur noch diese anzeigt.

    Bekannte Einschränkung: die separate Navigations-Datei (nav.xhtml/NCX,
    das "Kapitelmenü" mancher E-Reader-Apps) wird NICHT mitbereinigt und kann
    danach auf entfernte Kapitel verweisen. Das eigentliche Lesen/Blättern
    funktioniert trotzdem korrekt, nur ein Sprung über das Kapitelmenü direkt
    zu einem herausgeschnittenen Kapitel würde ins Leere laufen."""
    with zipfile.ZipFile(input_path, "r") as zin:
        opf_path = _find_opf_path(zin)
        spine_files = _get_spine_files(zin, opf_path)
        total = len(spine_files)

        if start_index < 1 or end_index < start_index or start_index > total:
            raise ValueError(
                f"Ungültiger Abschnittsbereich {start_index}-{end_index} "
                f"(Dokument hat {total} Abschnitte)."
            )
        end_index = min(end_index, total)
        keep_idrefs = {idref for idref, _ in spine_files[start_index - 1:end_index]}

        opf_data = zin.read(opf_path)
        root = ET.fromstring(opf_data)
        ET.register_namespace("", NAMESPACES["opf"])

        spine_elem = root.find("opf:spine", NAMESPACES)
        for itemref in list(spine_elem.findall("opf:itemref", NAMESPACES)):
            if itemref.attrib.get("idref") not in keep_idrefs:
                spine_elem.remove(itemref)

        new_opf_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        all_names = zin.namelist()
        # Navigation (EPUB3 nav.xhtml) und NCX an die tatsächlich behaltenen
        # Kapitel anpassen. Entfernte Kapitel bleiben nicht mehr im
        # Inhaltsverzeichnis verlinkt.
        modified = {opf_path: new_opf_bytes}
        keep_paths = {path for _, path in spine_files[start_index - 1:end_index]}
        for name in all_names:
            low=name.lower()
            if low.endswith(".xhtml") or low.endswith(".html") or low.endswith(".ncx"):
                try:
                    raw=zin.read(name)
                    root2=ET.fromstring(raw)
                    changed=False
                    for elem in list(root2.iter()):
                        href=elem.attrib.get("href") or elem.attrib.get("src")
                        if href and (low.endswith(".ncx") or _local_tag(elem)=="a"):
                            target=href.split("#",1)[0]
                            base=os.path.dirname(name)
                            resolved=os.path.normpath(os.path.join(base,target)).replace("\\","/")
                            if target and resolved and not any(resolved==kp or resolved.endswith(kp) for kp in keep_paths):
                                parent=None
                                for pnode in root2.iter():
                                    if elem in list(pnode): parent=pnode; break
                                if parent is not None:
                                    parent.remove(elem); changed=True
                    if changed:
                        modified[name]=ET.tostring(root2,encoding="utf-8",xml_declaration=True)
                except Exception:
                    pass

        with zipfile.ZipFile(output_path, "w") as zout:
            if "mimetype" in all_names:
                zout.writestr("mimetype", zin.read("mimetype"), zipfile.ZIP_STORED)
            for name in all_names:
                if name == "mimetype":
                    continue
                data = modified.get(name, zin.read(name))
                zout.writestr(name, data, zipfile.ZIP_DEFLATED)

    return end_index - start_index + 1
