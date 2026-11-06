"""Separate read-only loopback viewer; never restarts an existing dashboard."""

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from continuous_pilot_server import handler as pilot_handler

PANEL = '''<section id="comparison"><h2>Broad, local and diverse-region PPO</h2>
<p id="comparisonStatus" role="status">Loading verified three-method comparison…</p>
<p id="comparisonSummary"></p><div id="comparisonChecks" class="scroll"></div>
<svg id="comparisonCurve" viewBox="0 0 700 350" role="img" aria-label="Matched-budget exploratory capture-rate curves for three methods"></svg>
<p>Gray: broad pilot · mint: one local region · blue: diverse regions with changed settling. Exploratory training curves are separate from frozen checks above.</p>
<p id="comparisonActionScope" class="note"></p><p id="comparisonScope"></p>
<h3>Admission and settling failures</h3><div id="comparisonFailures" class="scroll"></div>
<p class="note">Counts include rejected active attempts. Improved admission alone is not improved capture; a changed setup controller also changes the native experiment.</p>
<div id="setupCheck" hidden><h3>Fixed-request setup check</h3><p id="setupCheckSummary"></p><p id="setupCheckScope" class="note"></p><details><summary>Compact setup-check receipt</summary><pre id="setupCheckReceipt"></pre></details></div>
<details><summary>Exact declared setup and settling protocol</summary><pre id="comparisonSettling"></pre></details>
<h3>Geometry-region diversity</h3><p id="comparisonPreparation"></p><div id="comparisonSupports" class="scroll"></div>
<p>Ranges below are envelopes over separate local regions. Gaps between regions are not permitted actions, and finite qualification probes do not certify a feasible interior. Sampled requests are not executed motion or successful-shot coverage.</p>
<details><summary>Individual geometry regions and provenance</summary><div id="comparisonRegions" class="scroll"></div><pre id="comparisonSupport"></pre></details>
</section>'''


def index_html():
    text=Path(__file__).with_name("continuous_pilot.html").read_text()
    text=text.replace("Sircus · Continuous shot pilot","Sircus · Diverse-region PPO comparison")
    text=text.replace("A wider shot space. A continuous policy.","More feasible choices. Measured settling.")
    text=text.replace("Two articulated arms choose a bounce point, board height and tilt, and ball-release height. Reachability sampling, learning and frozen evaluation stay separate.",
        "A hybrid policy chooses a geometry region and continuous shot parameters. Both earlier pilots stay frozen; measured release gates and capture scoring remain unchanged.")
    text=text.replace('<div class="metrics" id="metrics">',PANEL+'\n<div class="metrics" id="metrics">')
    return text.replace('</body>','<script type="module" src="/diverse_pilot_panel.js"></script></body>').encode()


def base_script():
    # Label only this served copy; historical readers keep their original bytes.
    text=Path(__file__).with_name("continuous_pilot.js").read_text()
    text=text.replace("report.study?.domain?.decoder==='geometry-conditioned local affine residual-v1'?",
        "['geometry-conditioned local affine residual-v1','geometry-only region choice + local affine residual-v1'].includes(report.study?.domain?.decoder)?")
    return text.encode()


def handler(root):
    class DiverseHandler(pilot_handler(root)):
        def send_content(self,head=False):
            path=self.path.split("?",1)[0]
            if path=="/":data,mime=index_html(),"text/html; charset=utf-8"
            elif path=="/diverse_pilot_panel.js":data,mime=Path(__file__).with_name("diverse_pilot_panel.js").read_bytes(),"application/javascript"
            elif path=="/continuous_pilot.js":data,mime=base_script(),"application/javascript"
            else:return super().send_content(head=head)
            self.send_response(200)
            self.send_header("Content-Type",mime)
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            if not head:
                try:self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
    return DiverseHandler


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",type=int,default=18796)
    args=parser.parse_args()
    if not (args.output/"report.json").is_file():parser.error("Explicit diverse-region report required")
    ThreadingHTTPServer(("127.0.0.1",args.port),handler(args.output)).serve_forever()
