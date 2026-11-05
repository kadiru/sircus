"""A separate read-only loopback comparison viewer; old viewers stay untouched."""

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from continuous_pilot_server import handler as pilot_handler

PANEL = '''<section id="comparison"><h2>Broad versus feasibility-guided PPO</h2>
<p id="comparisonStatus" role="status">Loading verified comparison…</p>
<p id="comparisonSummary"></p><div id="comparisonChecks" class="scroll"></div>
<svg id="comparisonCurve" viewBox="0 0 700 350" role="img" aria-label="Matched-budget exploratory capture-rate curves"></svg>
<p>Gray: previous broad-domain pilot · mint: new map-guided pilot. These are rolling exploratory training rates; frozen held-out checks are in the table.</p>
<p id="comparisonActionScope" class="note"></p><p id="comparisonScope"></p>
<p id="comparisonPreparation"></p><h3>Geometry-conditioned local action support</h3><p>These ranges are continuous residual boxes around a geometry-only anchor. Sampled probe acceptance does not certify every point. Empty supports remain unresolved.</p><div id="comparisonSupports" class="scroll"></div><details><summary>Geometry-only support definitions and preparation provenance</summary><pre id="comparisonSupport"></pre></details>
</section>'''


def index_html():
    source = Path(__file__).with_name("continuous_pilot.html").read_text()
    source = source.replace("Sircus · Continuous shot pilot", "Sircus · Feasibility-guided PPO comparison")
    source = source.replace("A wider shot space. A continuous policy.", "Geometry first. Continuous shot learning.")
    source = source.replace("Two articulated arms choose a bounce point, board height and tilt, and ball-release height. Reachability sampling, learning and frozen evaluation stay separate.",
        "A geometry-only map narrows continuous shot proposals. The previous broad-domain pilot remains frozen for comparison; native scoring and release guards stay unchanged.")
    source = source.replace('Allowed domain', 'Global outer domain')
    source = source.replace('<div class="metrics" id="metrics">', PANEL+'\n<div class="metrics" id="metrics">')
    return source.replace('</body>', '<script type="module" src="/feasibility_guided_panel.js"></script></body>').encode()


def handler(root):
    class GuidedHandler(pilot_handler(root)):
        def send_content(self, head=False):
            path = self.path.split("?",1)[0]
            if path == "/":
                data, mime = index_html(), "text/html; charset=utf-8"
            elif path == "/feasibility_guided_panel.js":
                data, mime = Path(__file__).with_name("feasibility_guided_panel.js").read_bytes(), "application/javascript"
            else:
                return super().send_content(head=head)
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if not head:
                try:self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
    return GuidedHandler


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",type=int,default=18795)
    args=parser.parse_args()
    if not (args.output/"report.json").is_file():parser.error("Explicit guided report required")
    ThreadingHTTPServer(("127.0.0.1",args.port),handler(args.output)).serve_forever()
