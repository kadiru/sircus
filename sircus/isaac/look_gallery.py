"""Lay out native Isaac appearance previews without altering their scene content."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def font(size):
    for filename in (
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def label(value):
    if isinstance(value, dict):
        return str(value.get("label", value.get("name", value.get("type", ""))))
    return str(value or "")


def descriptions(variant):
    return [
        f"Table · {label(variant.get('table'))}",
        f"Floor · {label(variant.get('floor'))}",
        f"Background · {label(variant.get('background'))}",
    ]


def fit_line(draw, text, selected_font, width):
    while draw.textlength(text, font=selected_font) > width:
        text = text[:-2].rstrip("…") + "…"
    return text


def sheet(directory, variants, filename, columns, width):
    image_height = round(width * 9 / 16)
    label_height, gutter, top = 119, 18, 76
    rows = (len(variants) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            columns * (width + gutter) + gutter,
            top + rows * (image_height + label_height + gutter),
        ),
        "#171c22",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gutter, 18), "SIRCUS · APPEARANCE PREVIEWS", font=font(26), fill="#ffffff"
    )
    draw.text(
        (gutter, 49),
        "Native Isaac Sim renders · Choose a numbered combination",
        font=font(16),
        fill="#b7c1ce",
    )
    title_font, detail_font = font(21), font(16)
    for i, variant in enumerate(variants):
        x = gutter + (i % columns) * (width + gutter)
        y = top + (i // columns) * (image_height + label_height + gutter)
        image = directory / f"look-{int(variant['id']):02}.jpg"
        with Image.open(image) as source:
            preview = source.convert("RGB")
            preview.thumbnail((width, image_height), LANCZOS)
            canvas.paste(
                preview,
                (
                    x + (width - preview.width) // 2,
                    y + (image_height - preview.height) // 2,
                ),
            )
        title = f"{int(variant['id']):02}  {variant.get('name', 'Combination')}"
        draw.text(
            (x + 8, y + image_height + 9),
            fit_line(draw, title, title_font, width - 16),
            font=title_font,
            fill="#ffffff",
        )
        for line, description in enumerate(descriptions(variant)):
            draw.text(
                (x + 8, y + image_height + 39 + 23 * line),
                fit_line(draw, description, detail_font, width - 16),
                font=detail_font,
                fill="#c1cad6",
            )
    canvas.save(directory / filename, quality=94, subsampling=0)


def page(directory, variants):
    cards = []
    entries = []
    for variant in variants:
        number = int(variant["id"])
        title = f"{number:02} · {variant.get('name', 'Combination')}"
        filename = f"look-{number:02}.jpg"
        overview = f"overview-{number:02}.jpg"
        if not (directory / overview).is_file():
            overview = None
        detail = "\n".join(descriptions(variant))
        entries.append(
            dict(title=title, file=filename, overview=overview, detail=detail)
        )
        overview_thumb = (
            f'<img class="overview-thumb" loading="lazy" src="{overview}" '
            f'alt="Full farm view of {html.escape(title, quote=True)}" title="Inspect full farm">'
            if overview
            else ""
        )
        cards.append(
            f'<a class="card" href="{filename}" data-index="{len(entries)-1}">'
            f'<img loading="lazy" src="{filename}" alt="{html.escape(title, quote=True)}">'
            f'<div class="caption"><h2>{html.escape(title)}</h2>'
            + "".join(f"<p>{html.escape(line)}</p>" for line in descriptions(variant))
            + overview_thumb
            + "</div></a>"
        )
    data = json.dumps(entries, ensure_ascii=False).replace("<", "\\u003c")
    content = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sircus · Choose a look</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#14191f;color:#edf2f7}
body{max-width:1680px;margin:auto;padding:30px}header{margin-bottom:24px}h1{font-size:32px;margin:0 0 10px}
header p{color:#b8c5d3;margin:7px 0}header a{color:#86d9d4}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:20px}
.card{display:block;border:1px solid #35414d;border-radius:12px;overflow:hidden;background:#1c242d;color:inherit;text-decoration:none;transition:border-color .15s}
.card:hover,.card:focus-visible{border-color:#83d5cf;outline:2px solid #83d5cf}.card img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}
.caption{padding:15px}h2{font-size:19px;margin:0 0 12px}.caption p{font-size:13px;line-height:1.5;margin:4px 0;color:#bdcbd9}
.card img.overview-thumb{width:42%;margin-top:12px;border:1px solid #647686;border-radius:5px}
dialog{padding:0;border:1px solid #647686;border-radius:12px;max-width:96vw;max-height:96vh;background:#17212b;color:inherit}
dialog::backdrop{background:#000c}.toolbar{display:flex;justify-content:space-between;align-items:center;gap:15px;padding:12px 16px}.toolbar strong{flex:1}
button{padding:9px 13px;border:1px solid #667c8d;border-radius:6px;background:#263a48;color:white;font:inherit;cursor:pointer}button:hover{background:#385562}
#large{display:block;max-width:94vw;width:1440px;max-height:75vh;object-fit:contain;background:#101010}#details{white-space:pre-line;color:#c3d0dc;margin:14px 20px 20px;line-height:1.6}
@media(max-width:600px){body{padding:16px}h1{font-size:27px}.toolbar{flex-wrap:wrap}.toolbar strong{width:100%;flex:auto}}
</style>
<header><h1>Choose the Sircus farm’s look</h1><p>20 native Isaac Sim previews. Select a card to inspect the full render, then give the combination’s number.</p>
<p><a href="summary.jpg">Open all 20 comparisons</a> · <a href="group-01-05.jpg">01–05</a> · <a href="group-06-10.jpg">06–10</a> · <a href="group-11-15.jpg">11–15</a> · <a href="group-16-20.jpg">16–20</a></p></header>
<main>CARDS</main>
<dialog id="viewer"><div class="toolbar"><button id="prev" aria-label="Previous render">←</button><strong id="title"></strong><button id="view" aria-pressed="false">Full farm</button><button id="next" aria-label="Next render">→</button><button id="close">Close</button></div><img id="large" alt=""><p id="details"></p></dialog>
<script>
const entries = ENTRIES;
const viewer = document.getElementById('viewer');
let selected = 0;
let overview = false;
function show(index) {
  selected = (index + entries.length) % entries.length;
  const entry = entries[selected];
  const wide = overview && Boolean(entry.overview);
  document.getElementById('large').src = wide ? entry.overview : entry.file;
  document.getElementById('large').alt = entry.title + (wide ? ' · Full farm' : ' · Close view');
  document.getElementById('title').textContent = entry.title + (wide ? ' · Full farm' : ' · Close view');
  document.getElementById('details').textContent = entry.detail;
  const viewButton = document.getElementById('view');
  viewButton.hidden = !entry.overview;
  viewButton.textContent = wide ? 'Close view' : 'Full farm';
  viewButton.setAttribute('aria-pressed', String(wide));
}
document.querySelectorAll('.card').forEach(card => card.addEventListener('click', event => {
  if (!viewer.showModal) return;
  event.preventDefault(); overview = event.target.classList.contains('overview-thumb');
  show(Number(card.dataset.index)); viewer.showModal();
}));
document.getElementById('view').addEventListener('click', () => { overview = !overview; show(selected); });
document.getElementById('prev').addEventListener('click', () => show(selected - 1));
document.getElementById('next').addEventListener('click', () => show(selected + 1));
document.getElementById('close').addEventListener('click', () => viewer.close());
document.addEventListener('keydown', event => {
  if (!viewer.open) return;
  if (event.key === 'ArrowRight') show(selected + 1);
  if (event.key === 'ArrowLeft') show(selected - 1);
});
</script></html>
"""
    (directory / "index.html").write_text(
        content.replace("CARDS", "\n".join(cards)).replace("ENTRIES", data)
    )


def generate(directory):
    variants = json.loads((directory / "manifest.json").read_text())["variants"]
    if len(variants) != 20 or sorted(int(v["id"]) for v in variants) != list(
        range(1, 21)
    ):
        raise ValueError("Expected numbered variants 1 through 20")
    variants = sorted(variants, key=lambda item: int(item["id"]))
    for variant in variants:
        with Image.open(directory / f"look-{int(variant['id']):02}.jpg") as image:
            image.verify()
        overview = directory / f"overview-{int(variant['id']):02}.jpg"
        if overview.is_file():
            with Image.open(overview) as image:
                image.verify()
    sheet(directory, variants, "summary.jpg", columns=5, width=480)
    for first in range(0, 20, 5):
        sheet(
            directory,
            variants[first : first + 5],
            f"group-{first+1:02}-{first+5:02}.jpg",
            columns=2,
            width=640,
        )
    page(directory, variants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    generate(parser.parse_args().directory)
