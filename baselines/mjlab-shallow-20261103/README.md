# Exact shallow-board candidate — November 3, 2026

Source-freeze status: **native shot results pending**. This is a new, separately
versioned DEVELOPMENT experiment; the completed lower-board0/208 far-flight
result is preserved at its original fingerprint.

The exact request is board-center height **380 mm**, tilt **28° from horizontal**,
and drop gap **386.8 mm** at650/700/750/770 mm horizontal board-to-bucket distances.
With radius37.5 mm, released ball-center height is804.3 mm above tabletop.

[CPU preflight](cpu-preflight.json) passed all four paired native endpoint
checks with minimum clearance lower bound5.969446 mm and maximum full-pose
position error0.666 nm. CUDA was hidden; **zero ball flights were run**. This
exact386.8 mm input is24.338µm lower than the previous CPU-only386.824338 mm
proposal, and was solved and checked independently.

[Protocol/reproduction](../../sircus/mjlab/SHALLOW_STUDY.md):24 exact parent
controls at64 substeps, then four exact discovery shots. Only actual captures
receive3 nominal+30 fresh jointly perturbed H/tilt/gap requests, paired at
32/64/128. Maximum424 requests. No refinement or parameters changed in response
to an outcome. All rejections remain failures and no reliability is inferred
from an untested engine or one discovery shot.

The4090 task waits for the existing cooperative GPU lock; another Sircus
benchmark already owns it. No process is interrupted. The4070, Isaac setup,
active policies and original baselines remain untouched. The existing viewer
will receive only a separate readonly`#shallowBoardStudy` via static/derived-file
updates, without restart or solver activation. Raw arrays/joints/logs/credentials
stay outside Git. The bucket's reported175 mm height interpretation and
provisional interior-depth model remain unchanged.
