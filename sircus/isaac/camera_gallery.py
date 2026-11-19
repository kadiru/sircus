"""Lay out native Isaac camera-composition stills for numbered view selection."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from look_gallery import LANCZOS, fit_line, font


def descriptions(variant):
    return [
        f"Camera height {float(variant['eye_m'][2]):.2f} m above tabletop Z=0",
        f"Aim Z {float(variant['target_m'][2]):+.2f} m (tabletop Z=0)",
    ]


def comparison_sheet(directory, variants, filename, width=640):
    height = round(width * 9 / 16)
    gap, caption, heading = 20, 90, 86
    rows = (len(variants) + 1) // 2
    canvas = Image.new(
        "RGB",
        (2 * width + 3 * gap, heading + rows * (height + caption + gap)),
        "#e5e8eb",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 17), "SIRCUS · CAMERA OPTIONS", font=font(28), fill="#233440")
    draw.text(
        (gap, 53),
        "Native Isaac Sim stills · Same silver-gray scene · Choose a numbered view",
        font=font(17),
        fill="#526371",
    )
    title_font, detail_font = font(22), font(17)
    for index, variant in enumerate(variants):
        x = gap + (index % 2) * (width + gap)
        y = heading + (index // 2) * (height + caption + gap)
        with Image.open(directory / variant["file"]) as original:
            image = original.convert("RGB")
            image.thumbnail((width, height), LANCZOS)
            canvas.paste(
                image,
                (x + (width - image.width) // 2, y + (height - image.height) // 2),
            )
        title = f"{int(variant['id']):02}  {variant['name']}"
        draw.text(
            (x + 8, y + height + 9),
            fit_line(draw, title, title_font, width - 16),
            font=title_font,
            fill="#233440",
        )
        for line, detail in enumerate(descriptions(variant)):
            draw.text(
                (x + 8, y + height + 40 + 22 * line),
                fit_line(draw, detail, detail_font, width - 16),
                font=detail_font,
                fill="#526371",
            )
    canvas.save(directory / filename, quality=94, subsampling=0)


def page(directory, variants):
    cards, entries = [], []
    for index, variant in enumerate(variants):
        title = f"{int(variant['id']):02} · {variant['name']}"
        detail = "\n".join(descriptions(variant))
        entries.append(dict(title=title, file=variant["file"], detail=detail))
        cards.append(
            f'<a class="card" href="{html.escape(variant["file"],quote=True)}" data-index="{index}">'
            f'<img src="{html.escape(variant["file"],quote=True)}" loading="lazy" alt="{html.escape(title,quote=True)}">'
            '<div class="caption">'
            f"<h2>{html.escape(title)}</h2>"
            + "".join(f"<p>{html.escape(line)}</p>" for line in descriptions(variant))
            + "</div></a>"
        )
    content = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sircus · Choose a camera view</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,sans-serif;background:#e5e8eb;color:#263642}
body{margin:auto;padding:28px;max-width:1540px}header{margin-bottom:24px}h1{margin:0 0 10px;font-size:32px}
header p{color:#536573;margin:8px 0;line-height:1.5}a{color:#245c73}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:20px}
.card{border:1px solid #c1cbd3;border-radius:12px;overflow:hidden;background:#f5f7f8;text-decoration:none;color:inherit}
.card:hover,.card:focus-visible{outline:2px solid #4d8da6}.card img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#bdc5cb}
.caption{padding:14px}h2{font-size:20px;margin:0 0 10px}.caption p{font-size:13px;line-height:1.4;margin:5px 0;color:#536573}
dialog{padding:0;border:1px solid #9baab5;border-radius:12px;max-width:96vw;max-height:96vh;background:#eef2f4;color:#263642}
dialog::backdrop{background:#25333fbb}.toolbar{display:flex;gap:14px;align-items:center;padding:12px 16px}.toolbar strong{flex:1}
button{padding:9px 13px;border:1px solid #9daeb9;border-radius:6px;background:#e3eaf0;color:#263642;font:inherit;cursor:pointer}
button:hover{background:#d4e1e9}#large{display:block;width:1440px;max-width:94vw;max-height:75vh;object-fit:contain}
#details{white-space:pre-line;margin:15px 20px;line-height:1.6;color:#536573}
@media(max-width:600px){body{padding:15px}h1{font-size:27px}.toolbar{flex-wrap:wrap}.toolbar strong{flex-basis:100%}main{grid-template-columns:1fr}}
</style>
<header><h1>Choose the Sircus camera view</h1>
<p>10 native Isaac Sim composition previews. The silver-gray surfaces, lighting and scene state are fixed while the camera changes.</p>
<p>These are still previews for choosing the next recording angle. Heights are measured above the tabletop, whose Z is zero.</p>
<p><a href="summary.jpg">All 10 views</a> · <a href="group-01-05.jpg">01–05, larger</a> · <a href="group-06-10.jpg">06–10, larger</a></p>
</header><main>CARDS</main>
<dialog id="viewer"><div class="toolbar"><button id="prev" aria-label="Previous view">←</button><strong id="title"></strong><button id="next" aria-label="Next view">→</button><button id="close">Close</button></div><img id="large" alt=""><p id="details"></p></dialog>
<script>
const entries = ENTRIES;
const viewer = document.getElementById('viewer');
let selected = 0;
function show(index) {
  selected = (index + entries.length) % entries.length;
  const entry = entries[selected];
  document.getElementById('large').src = entry.file;
  document.getElementById('large').alt = entry.title;
  document.getElementById('title').textContent = entry.title;
  document.getElementById('details').textContent = entry.detail;
}
document.querySelectorAll('.card').forEach(card => card.addEventListener('click', event => {
  if (!viewer.showModal) return;
  event.preventDefault(); show(Number(card.dataset.index)); viewer.showModal();
}));
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
        content.replace("CARDS", "\n".join(cards)).replace(
            "ENTRIES", json.dumps(entries, ensure_ascii=False).replace("<", "\\u003c")
        )
    )


def generate(directory):
    manifest = json.loads((directory / "camera-options.json").read_text())
    variants = sorted(manifest["variants"], key=lambda item: int(item["id"]))
    if [int(v["id"]) for v in variants] != list(range(1, 11)):
        raise ValueError("Expected ten consecutively numbered camera options")
    for variant in variants:
        expected = f"camera-{int(variant['id']):02}.jpg"
        if variant["file"] != expected:
            raise ValueError("Camera preview filename disagrees with its option number")
        with Image.open(directory / expected) as image:
            image.verify()
    comparison_sheet(directory, variants, "summary.jpg")
    comparison_sheet(directory, variants[:5], "group-01-05.jpg", width=760)
    comparison_sheet(directory, variants[5:], "group-06-10.jpg", width=760)
    page(directory, variants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    generate(parser.parse_args().directory)
