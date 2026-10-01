# Japanese PDF translation fix

## Fixed

- Japanese PDFs with an existing native text layer no longer trigger full-page OCR.
- The processor checks the complete available sample (up to 15 pages) for native text before deciding whether OCR is needed.
- Vertical-Japanese OCR remains available as a fallback for scanned/image-only pages.
- The normal paragraph/batch translation path is now used for native Japanese text, including `ja -> de`.
- This prevents the preparation phase from spending the startup time rendering/OCRing normal text PDFs.
