# NEXUS Translation Pipeline Fix

## Behoben

1. **Japanische Web-PDFs**
   - PDF-Texte werden vor der Übersetzung mit Unicode NFKC normalisiert.
   - Dadurch werden CJK-Kompatibilitätszeichen wie `⾯` zu `面`.
   - Eine native Textebene wird weiterhin vor OCR bevorzugt.

2. **Fertig gemeldet, aber Originalsprache**
   - Übersetzungsfehler werden nicht mehr als erfolgreiche Übersetzung behandelt.
   - Es gibt mehrere Retries und anschließend einen zweiten kostenlosen Google-Web-Fallback.
   - Wenn keine echte Übersetzung erzeugt werden kann, wird der Fehler sichtbar protokolliert.

3. **Batch-Übersetzung**
   - Wenn der Trennmarker von Google verändert wird, werden die betroffenen Absätze einzeln übersetzt.
   - Ein fehlgeschlagener Batch bleibt dadurch nicht unbemerkt.

4. **Fortschritt**
   - Bei PDF-Jobs wird die Seitenzahl sofort nach dem Upload gesetzt.
   - Die UI kann deshalb schon während der Analyse `0 von N Seiten` anzeigen.

5. **Französisch**
   - `fr` ist als Quell- und Zielsprache im Frontend vorhanden.
   - Das Backend unterstützt `fra` für OCR.
