# Choose the farm appearance

`look_preview.py` renders 20 material combinations through the native Isaac RTX camera. Each combination has a close view and a full-farm view. All images use the same frozen 35-table/70-arm state, camera poses and lighting. The manifest records the state digest and verifies that physics did not advance while the looks changed. These are appearance previews.

The five columns are cool gray, pale ash, honey oak, walnut and charcoal tabletops. The four rows are mid-gray floor, warm concrete floor, an off-white square with charcoal surroundings, and a cool gray square with off-white surroundings. The square is 22.8 × 22.8 m, centered at (0.67, −0.25) m. It extends at least 3 m beyond every outside tabletop. The three wood materials use seeded procedural grain with explicit texture coordinates on a visual tabletop mesh.

A 250-intensity neutral dome provides fill, while a 1500-intensity distant light with 2° angular size creates visible shadows. Camera shadows and ambient occlusion are enabled. Existing collision geometry and source scene dimensions supply the arrangement.

Use the pinned [Isaac setup](README.md) with this directory as `SIRCUS_ISAAC_SOURCE`. Inside the existing isolated container, run with a fresh output directory.

```bash
python.sh look_preview.py --headless --enable_cameras --device cuda:0 \
  --source /workspace/output/dark-farm-20261102-v3 \
  --output /workspace/output/look-previews-new
python.sh look_gallery.py /workspace/output/look-previews-new
```

The camera saves `look-01.jpg` through `look-20.jpg`, with matching `overview-*.jpg` files. The gallery adds `index.html`, `summary.jpg` and four larger comparison sheets. Open the HTML, select a numbered card and toggle Close view / Full farm. The generated manifest lists material colors, camera poses, padding and SHA-256 checksums for all 40 native images. Images, textures and HTML stay outside Git.

The completed November 2 previews are in `output/look-previews-20261102-v3` on the configured Ubuntu host. [Render manifest](../../baselines/isaac-look-previews-20261102/manifest.json). The chosen look awaits the user's selection before a new movie is recorded.
