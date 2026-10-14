"""One place that formats source locations for pages, slides and transcript time ranges."""


def fmt_time(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def loc_label(kind: str, start: float, end: float | None = None) -> str:
    if kind == "page":
        return f"p.{int(start)}"
    if kind == "slide":
        return f"slide {int(start)}"
    if kind == "time":
        if end is not None and int(end) > int(start):
            return f"{fmt_time(start)}–{fmt_time(end)}"
        return fmt_time(start)
    return str(start)
