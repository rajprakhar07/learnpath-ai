"""Figure extraction from PDFs (pypdf). Only real embedded images are returned; captions come from the page's own
"Figure N: ..." sentence and are paired only when the pairing is unambiguous (same count of images and captions)."""
import hashlib
import io

from pypdf import PdfReader

from .textutil import split_sentences

MIN_SIDE = 80          # px: skip icons / rules / decorations
MAX_FIGURES = 60
MAX_BYTES = 5 * 1024 * 1024
MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "tiff": "image/tiff", "bmp": "image/bmp"}


def _captions(page_text: str):
    out = []
    for s in split_sentences(page_text):
        low = s.lower()
        if low.startswith(("figure ", "fig. ", "fig ")) and any(ch.isdigit() for ch in s[:12]):
            out.append(s)
    return out


def extract_pdf_figures(path, page_texts):
    """page_texts: {page_number: text}. -> list of figure dicts (loc_kind='page')."""
    out, seen = [], set()
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return []
        for pno, page in enumerate(reader.pages, start=1):
            try:
                imgs = list(page.images)
            except Exception:
                continue
            good = []
            for im in imgs:
                try:
                    data = im.data
                    if not data or len(data) > MAX_BYTES:
                        continue
                    pil = im.image
                    w, h = pil.size
                    if min(w, h) < MIN_SIDE:
                        continue
                    sha = hashlib.sha256(data).hexdigest()
                    if (pno, sha) in seen:
                        continue
                    seen.add((pno, sha))
                    ext = (im.name.rsplit(".", 1)[-1] if "." in im.name else "png").lower()
                    if ext not in MIME:
                        buf = io.BytesIO()
                        pil.convert("RGB").save(buf, "PNG")
                        data, ext = buf.getvalue(), "png"
                    good.append({"blob": data, "ext": ext, "mime": MIME[ext], "width": w, "height": h, "alt_text": "", "caption": "",
                                 "loc_kind": "page", "loc_start": pno})
                except Exception:
                    continue
            caps = _captions(page_texts.get(pno, ""))
            if good and len(caps) == len(good):
                for g, c in zip(good, caps):
                    g["caption"] = c
            out.extend(good)
            if len(out) >= MAX_FIGURES:
                break
    except Exception:
        return out
    return out[:MAX_FIGURES]
