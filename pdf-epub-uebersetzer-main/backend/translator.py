import time, threading, concurrent.futures, re, json, unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen
from deep_translator import GoogleTranslator
from memory import get_translation, put_translation, get_glossary, put_glossary

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
GOOGLE_TIMEOUT_SECONDS = 20
TRANSLATE_CONCURRENCY = 1
_translate_semaphore = threading.Semaphore(TRANSLATE_CONCURRENCY)
# Ein globaler Takt für ALLE Übersetzungsanfragen. Mehrere PDF-Seiten dürfen
# dadurch nicht gleichzeitig denselben kostenlosen Dienst fluten.
MIN_SECONDS_BETWEEN_REQUESTS = 0.28  # ca. 3.5 Requests/s, unter dem 5/s-Limit
_request_lock = threading.Lock()
_last_request_at = 0.0

def _wait_for_translation_slot():
    global _last_request_at
    with _request_lock:
        now = time.monotonic()
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (now - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()

_network_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
POLISH_PIVOT_LANG = "en"
POLISH_PIVOT_FALLBACK = "fr"


class TranslationError(RuntimeError):
    """Raised when a translation could not actually be produced.

    Returning the source text on failure is deliberately forbidden: doing that
    makes the application report a finished translation while the PDF still
    contains the original language.
    """


def normalize_source_text(text):
    """Normalize PDF Unicode before translation.

    Some Japanese web PDFs encode normal kanji as Unicode compatibility
    characters such as U+2FA3 (⾯) instead of U+9762 (面). NFKC converts these
    to ordinary Japanese characters understood much more reliably by
    translation engines.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[\u0001-\u0008\u000b\u000c\u000e-\u001f]", " ", text)
    # PDF extraction can leave isolated control-like replacement glyphs.
    text = text.replace("\uFFFD", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _looks_translated(source, result, from_code, to_code):
    if not result or not result.strip():
        return False
    s = normalize_source_text(source).strip()
    r = normalize_source_text(result).strip()
    if not s:
        return True
    if r == s:
        # Identical output is valid for a tiny number of names/symbols, but
        # not for a normal sentence. Let the caller use the source-language
        # character test to decide whether it is suspicious.
        letters = re.sub(r"[\W\d_]+", "", s, flags=re.UNICODE)
        return len(letters) < 4
    return True


def _direct_google_gtx(text, from_code, to_code):
    """Small dependency-free fallback using Google's public web translation
    endpoint. This is the same free web service family used by many clients;
    no API key is required. It is a fallback, not a paid API.
    """
    q = quote(text, safe="")
    url = (
        "https://translate.googleapis.com/translate_a/single"
        f"?client=gtx&sl={quote(from_code)}&tl={quote(to_code)}"
        f"&dt=t&q={q}"
    )
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 NEXUS-Translator/1.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=GOOGLE_TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    parts = []
    for item in data[0] if isinstance(data, list) and data else []:
        if isinstance(item, list) and item and item[0]:
            parts.append(str(item[0]))
    result = "".join(parts).strip()
    if not result:
        raise TranslationError("Google-Webübersetzer lieferte eine leere Antwort.")
    return result


def _google_translate_pair(text, from_code, to_code):
    text = normalize_source_text(text)
    if not text:
        return text

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _translate_semaphore:
                _wait_for_translation_slot()
                f = _network_executor.submit(
                    lambda: GoogleTranslator(source=from_code, target=to_code).translate(text)
                )
                result = f.result(timeout=GOOGLE_TIMEOUT_SECONDS)
            if _looks_translated(text, result, from_code, to_code):
                return result
            last_error = TranslationError("Übersetzer gab den unveränderten Quelltext zurück.")
        except Exception as e:
            last_error = e
            print(
                f"Übersetzungsversuch {attempt}/{MAX_RETRIES} "
                f"({from_code}->{to_code}) fehlgeschlagen: {e}"
            )
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * (2 ** (attempt - 1)))

    # Unofficial free web endpoint as a second path. This is important on
    # systems where deep-translator's request is blocked while normal HTTPS
    # requests still work.
    try:
        with _translate_semaphore:
            _wait_for_translation_slot()
            f = _network_executor.submit(
                lambda: _direct_google_gtx(text, from_code, to_code)
            )
            result = f.result(timeout=GOOGLE_TIMEOUT_SECONDS)
        if _looks_translated(text, result, from_code, to_code):
            return result
        last_error = TranslationError("Google-Webübersetzer gab erneut den Quelltext zurück.")
    except Exception as e:
        last_error = e

    raise TranslationError(
        f"Übersetzung {from_code}->{to_code} fehlgeschlagen: {last_error}"
    )


def _protect_glossary(text, glossary):
    if not glossary:
        return text, {}
    replacements = {}
    for i, (src, target) in enumerate(
        sorted(glossary.items(), key=lambda x: -len(x[0]))
    ):
        if not src.strip():
            continue
        token = f"ZXGLOSS{i}Q"
        pattern = r"(?<!\w)" + re.escape(src) + r"(?!\w)"
        if re.search(pattern, text, re.I):
            text = re.sub(pattern, token, text, flags=re.I)
            replacements[token] = target
    return text, replacements


def _restore(text, replacements):
    for token, target in replacements.items():
        text = text.replace(token, target)
    return text


class Translator:
    def translate(self, text, from_code, to_code, use_memory=True, auto_glossary=True):
        text = normalize_source_text(text)
        if not text or not text.strip():
            return text
        if from_code == to_code:
            return text

        if use_memory:
            cached = get_translation(text, from_code, to_code)
            if cached:
                # Old versions could have cached a failed/source-identical
                # result. Never reuse such a value.
                if _looks_translated(text, cached, from_code, to_code):
                    return cached

        glossary = get_glossary(from_code, to_code) if auto_glossary else {}
        protected, replacements = _protect_glossary(text, glossary)
        result = _google_translate_pair(protected, from_code, to_code)
        result = _restore(result, replacements)

        if result and result.strip():
            put_translation(text, from_code, to_code, result)
            # Automatic glossary is intentionally conservative. It avoids
            # spawning one extra network request for every proper noun.
        return result


translator = Translator()


def correct_grammar(text, lang_code):
    if not text or not text.strip():
        return text
    pivot = POLISH_PIVOT_FALLBACK if lang_code == POLISH_PIVOT_LANG else POLISH_PIVOT_LANG
    intermediate = _google_translate_pair(text, lang_code, pivot)
    polished = _google_translate_pair(intermediate, pivot, lang_code)
    return polished or text
