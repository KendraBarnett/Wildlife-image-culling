from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

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


def _keyword_block(tags: list[str]) -> str:
    if not tags:
        return ""
    items = "\n".join(f"     <rdf:li>{escape(t)}</rdf:li>" for t in tags)
    return (
        "   <dc:subject>\n"
        "    <rdf:Bag>\n"
        f"{items}\n"
        "    </rdf:Bag>\n"
        "   </dc:subject>\n"
        "   <lr:hierarchicalSubject>\n"
        "    <rdf:Bag>\n"
        f"{items}\n"
        "    </rdf:Bag>\n"
        "   </lr:hierarchicalSubject>\n"
    )


def _label_block(tags: list[str]) -> str:
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


def write_sidecar(image_path: str, rating: Optional[int], tags: list[str], notes: Optional[str]) -> Path:
    target = Path(image_path + ".xmp")
    rating_val = rating if rating is not None else -1
    content = XMP_TEMPLATE.format(
        rating=rating_val,
        label_block=_label_block(tags),
        keywords_block=_keyword_block(tags),
        notes_block=_notes_block(notes),
    )
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)
    return target
