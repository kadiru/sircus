# Completed case 01 clearance-window comparison

The longer prediction window refused the known problematic shot. It did **not**
find a better executed shot or improve capture. The independent native audit
passed **44,237 checks**, covering all 18 requested slots.

| Case 01 method | Requested | Released | Strict captures | Robot-contact releases | Abstentions |
|---|---:|---:|---:|---:|---:|
| Short rim window | 3 | 3 | 0 | 3/3 | 0 |
| Longer table window | 3 | 0 | 0 | Not observed | 3 |

The remaining 12 slots were matched pre/post controls. All behaved as expected:
six positive controls captured cleanly and six negative controls missed. There
were no invalid-physics results. The six captures in the overall run were
**controls**, not case 01 improvements. Unexecuted long-window shots have unknown
native contact and retention outcomes; their three requests stay in the capture
denominator.

Only the forecast window changed. The comparison retained candidate 14, its
exact held joint endpoints, five learned flight predictions, refined support
certificate, 2 mm margin, maximum 2 ms certificate sampling, native 64 substeps
and existing strict scorer. The original short-window certificate must also
pass. Missing, shortened or overlong horizons reject rather than clamp.

| Predicted quantity | Short window | Longer window |
|---|---:|---:|
| Five forecast durations | 321.488–326.923 ms | 397.344–402.265 ms |
| Minimum clearance lower bound | +12.052 mm | −36.068 mm |
| Clearance decision | Pass | Refuse |

These bounds concern conditional free-flight predictions, not measured contact
depth. The longer endpoint is the first descending ball-center crossing of
tabletop height plus ball radius. Bucket/table deflections remain outside this
free-flight model. No training, alternative-candidate search, noise test or
selector promotion occurred. This previously observed layout is development
evidence, not a fresh held-out result. Held endpoints do not validate driven
arm transit or physical hardware.

Worker wall time was **127.696331 s**; the complete pipeline took **142.651 s**.
Neither value is isolated GPU compute time. The 4090 was released after the run;
the existing 4070 process remained PID 2016772 at the completion check. Isaac
and existing services were not restarted.

Next: clarify whether table-assisted captures belong in the task objective,
then explore alternative admitted shots for case 01 on development data. Freeze
any resulting rule before a finite, fresh held-out comparison. Keep the current
selector unchanged meanwhile. A [historical example](table-assisted-example.json)
does confirm board → table → bucket/rim → one-second retention during a PPO
training attempt; its original strict score remains unchanged. See the
[explanation](../../docs/MJLAB_TABLE_ASSISTED_EXAMPLE_20261108.md).

## Evidence and provenance

- [Sanitized audited results](results.json)
- [Frozen protocol](../../sircus/mjlab/CLEARANCE_HORIZON.md)
- Read-only interface section: `#clearanceHorizon`
- Frozen producer: `ccfe3d9063693078ea1658359a207c42a8eda970`
- [Draft PR #88](https://github.com/kadiru/sircus/pull/88), stacked on
  [PR #87](https://github.com/kadiru/sircus/pull/87); no merge or policy deployment.

Hashes below bind the published results to the completed native evidence:

| Artifact | SHA-256 / fingerprint |
|---|---|
| Native report | `526df056c959160581ea8c64d29285a16c835d4c93d3bbde298eaeefd41d61dd` |
| Manifest | `423cb9a56056ed57e84153f7f2053af71e3c27a8525e2eb9b5fc2f57b52a7f98` |
| Request plan | `6bf91a80841b54e4027bc0d536b01b2d7dea5773c456344736e22666bfc1f247` |
| Contract fingerprint | `91a60c1aa6911249295a87dd23161b4586c13c1d024005dcfdaa5699da839d8f` |
| Audit source | `820ea2f5a5085ac1b0a89141638633032af174ef5ae806d930cb355c1064895d` |

Raw native observations, runtime paths, logs and host configuration remain
outside Git. Historical scoring definitions and evidence are preserved.
