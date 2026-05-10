"""XMP sidecar writer.

Two kinds of tags coexist in the sidecar's keyword list:

1. **User tags** — what the photographer added in the UI (Keep, Portfolio,
   "Eagle", "Yellowstone 2026", etc.). These come through as-is.

2. **AI tags** — auto-derived from the current state of the AI fields,
   prefixed with `WC:` so they don't collide with user tags. Recomputed
   on every sidecar write, so they always reflect the current truth —
   if you correct the AI (or re-analyze, or run Claude scoring), the
   next sidecar write strips the stale AI tags and writes fresh ones.

The hierarchical version uses `Wildlife AI|<label>` so Lightroom shows
them grouped under a single `Wildlife AI` parent keyword in its keyword
panel, rather than scattered across the flat list.

User corrections override AI values: if the user said `keep=yes` but the
AI said `no`, the sidecar carries `WC:Keep`."""

import json
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import sqlite3

XMP_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="wildlife-cull">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
    xmp:Rating="{rating}">
{label_block}{keywords_block}{notes_block}  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


AI_TAG_PREFIX = "WC:"
AI_HIERARCHY_ROOT = "Wildlife AI"


def _pretty(v: str) -> str:
    """soft_focus → Soft Focus; in_motion → In Motion."""
    return " ".join(part.capitalize() for part in str(v).replace("-", "_").split("_"))


def _ai_label_for(field: str, value: str) -> Optional[str]:
    """Map an AI field's value to a tag suffix. Returns None to skip
    the tag entirely (e.g. n/a, unknown, neutral cases that aren't
    useful as a Lightroom filter)."""
    if value in (None, "", "n/a", "Unknown", "unknown"):
        return None
    if field == "keep":
        return "Keep" if value == "yes" else "Cull"
    if field == "in_focus":
        return "InFocus" if value == "yes" else "OOF"
    if field == "eye_focus":
        return f"Eye-{_pretty(value)}"
    if field == "motion":
        return f"Motion-{_pretty(value)}"
    if field == "composition":
        return f"Comp-{_pretty(value)}"
    if field == "lighting":
        return f"Light-{_pretty(value)}"
    if field == "animal_type":
        return _pretty(value)
    if field == "technical_issue":
        return f"Issue-{_pretty(value)}"
    return None


def compute_ai_tags(row, feedback: Optional[dict] = None) -> list[str]:
    """Return the prefixed AI tag list reflecting the row's current state.
    The user's feedback (corrections) wins over the AI's stored value for
    every overlapping field — the cull-decision tag follows the user's
    truth, not the AI's first guess."""
    fb = feedback or {}

    def effective(field: str) -> Optional[object]:
        if field in fb and fb[field] not in (None, ""):
            return fb[field]
        return row.get(f"ai_{field}") if hasattr(row, "get") else row[f"ai_{field}"]

    tags: list[str] = []

    for field in ("keep", "in_focus", "eye_focus", "motion", "composition",
                  "lighting", "animal_type"):
        v = effective(field)
        if isinstance(v, str):
            label = _ai_label_for(field, v)
            if label:
                tags.append(AI_TAG_PREFIX + label)

    # Silhouette: boolean. Only emit a tag when it's true (silhouettes are
    # the interesting case).
    sil_fb = fb.get("is_silhouette")
    sil = sil_fb if sil_fb is not None else _get(row, "ai_is_silhouette")
    if sil:
        tags.append(AI_TAG_PREFIX + "Silhouette")

    # Technical issues — one tag per issue.
    issues_raw = fb.get("technical_issues")
    if issues_raw is None:
        issues_raw = _get(row, "ai_technical_issues")
    issues: list[str] = []
    if isinstance(issues_raw, list):
        issues = [str(x) for x in issues_raw]
    elif isinstance(issues_raw, str) and issues_raw:
        try:
            parsed = json.loads(issues_raw)
            if isinstance(parsed, list):
                issues = [str(x) for x in parsed]
        except Exception:
            pass
    for iss in issues:
        label = _ai_label_for("technical_issue", iss)
        if label:
            tags.append(AI_TAG_PREFIX + label)

    # Claude scores (Phase 2) — rounded to integer for filtering bands.
    tech = _get(row, "claude_technical_score")
    if isinstance(tech, (int, float)):
        tags.append(AI_TAG_PREFIX + f"Tech-{int(round(tech))}")
    aest = _get(row, "claude_aesthetic_score")
    if isinstance(aest, (int, float)):
        tags.append(AI_TAG_PREFIX + f"Aest-{int(round(aest))}")

    # Dedupe while preserving order.
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _get(row, key):
    if hasattr(row, "get"):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _hierarchical(tag: str) -> str:
    """`WC:Eye-Sharp` → `Wildlife AI|Eye-Sharp`. Lightroom uses `|` as the
    hierarchy separator, so this lets all auto-tags nest under one parent
    in its keyword tree."""
    if tag.startswith(AI_TAG_PREFIX):
        return f"{AI_HIERARCHY_ROOT}|{tag[len(AI_TAG_PREFIX):]}"
    return tag


def _strip_ai_tags(tags: list[str]) -> list[str]:
    return [t for t in tags if not t.startswith(AI_TAG_PREFIX)]


def _keyword_block(flat_tags: list[str], hierarchical_tags: list[str]) -> str:
    if not flat_tags:
        return ""
    flat = "\n".join(f"     <rdf:li>{escape(t)}</rdf:li>" for t in flat_tags)
    hier = "\n".join(f"     <rdf:li>{escape(t)}</rdf:li>" for t in hierarchical_tags)
    return (
        "   <dc:subject>\n"
        "    <rdf:Bag>\n"
        f"{flat}\n"
        "    </rdf:Bag>\n"
        "   </dc:subject>\n"
        "   <lr:hierarchicalSubject>\n"
        "    <rdf:Bag>\n"
        f"{hier}\n"
        "    </rdf:Bag>\n"
        "   </lr:hierarchicalSubject>\n"
    )


def _label_block(tags: list[str]) -> str:
    """Lightroom color label, picked from user tags. AI tags don't
    influence the color — only the photographer's explicit Portfolio /
    Keep / Reject tags do."""
    label = None
    if "Portfolio" in tags:
        label = "Purple"
    elif "Keep" in tags:
        label = "Green"
    elif "Reject" in tags:
        label = "Red"
    if label:
        return f"   <xmp:Label>{label}</xmp:Label>\n"
    return ""


def _notes_block(notes: Optional[str]) -> str:
    if not notes:
        return ""
    return (
        "   <dc:description>\n"
        "    <rdf:Alt>\n"
        f"     <rdf:li xml:lang=\"x-default\">{escape(notes)}</rdf:li>\n"
        "    </rdf:Alt>\n"
        "   </dc:description>\n"
    )


def write_sidecar(
    image_path: str,
    rating: Optional[int],
    tags: list[str],
    notes: Optional[str],
    ai_tags: Optional[list[str]] = None,
) -> Path:
    """Low-level sidecar writer. Callers that want the AI tags
    auto-attached should use `refresh_sidecar` instead — it pulls the
    current AI state from the DB and computes ai_tags fresh."""
    target = Path(image_path + ".xmp")
    rating_val = rating if rating is not None else -1
    user_tags = _strip_ai_tags(tags or [])
    ai_list = list(ai_tags or [])
    flat_tags = user_tags + ai_list
    hierarchical_tags = user_tags + [_hierarchical(t) for t in ai_list]
    content = XMP_TEMPLATE.format(
        rating=rating_val,
        label_block=_label_block(user_tags),
        keywords_block=_keyword_block(flat_tags, hierarchical_tags),
        notes_block=_notes_block(notes),
    )
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)
    return target


_SIDECAR_COLUMNS = (
    "path, user_rating, user_tags, user_notes, "
    "ai_keep, ai_in_focus, ai_eye_focus, ai_motion, ai_composition, "
    "ai_lighting, ai_is_silhouette, ai_animal_type, ai_technical_issues, "
    "claude_technical_score, claude_aesthetic_score, ai_feedback_json"
)


def refresh_sidecar(conn: sqlite3.Connection, image_id: int) -> Optional[Path]:
    """Single entry point used everywhere that needs to write the sidecar.
    Pulls the current row, strips any stale AI tags from the user tag
    list, recomputes the AI tag set from current fields + corrections,
    writes the XMP. Returns the path written (or None if the row is gone
    or has no path on disk)."""
    row = conn.execute(
        f"SELECT {_SIDECAR_COLUMNS} FROM images WHERE id=?", (image_id,)
    ).fetchone()
    if not row or not row["path"]:
        return None

    try:
        user_tags = json.loads(row["user_tags"]) if row["user_tags"] else []
    except Exception:
        user_tags = []
    if not isinstance(user_tags, list):
        user_tags = []
    user_tags = _strip_ai_tags(user_tags)

    feedback = None
    if row["ai_feedback_json"]:
        try:
            feedback = json.loads(row["ai_feedback_json"])
        except Exception:
            feedback = None

    ai_tags = compute_ai_tags(row, feedback)
    return write_sidecar(
        row["path"], row["user_rating"], user_tags, row["user_notes"],
        ai_tags=ai_tags,
    )
