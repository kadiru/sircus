# Credential provisioning

Local fake-only simulator8083 and numerical training need no cloud or robot credentials.

For authorized Mac hardware work, copy sircus/physical_trials/credentials.example.json to an ignored owner-readable .private path and provision sfhack10 machine address/API key ID/API key privately. The current adapter defaults to sircus/perception/.private/sfhack10.json. chmod600 that file. Do not send it to Ubuntu or commit it. Teach poses and limits on the actual setup; calibration.example.json is only a schema.

A future cloud simulation part needs its own Viam connection ID/secret and a deliberate binding/guard/bridge update. Never duplicate an existing live part identity across hosts. Github access uses each contributor's own GitHub identity, SSH key or credential helper; never embed a token in the remote URL.
