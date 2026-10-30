# Mac Mini audio work

Own `sircus/audio/` and audio-specific documentation/tests in this Sircus checkout.
Read the repository AGENTS/CONTRIBUTING and this directory's README/STATUS first.
Ghost Booth (`booth/`) is the main UI. The old `viam-ball-hackathon/sound/` is a
preserved historical snapshot, and `odd-balls-music/live/` contains separate
musical Logic sessions. Do not maintain a second active audio implementation there.

Start/restart Ghost Booth with FX off. Playback is an explicit operator action;
do not automatically audition sounds while doing development checks. Prefer
fake-backend tests, compile checks and silent native readiness for agent validation.
Keep profiles, recordings, tokens, compiler products and evidence in ignored runtime
locations. Only the original generated catalog assets are intentionally versioned.

Use the shared AudioEvent contract and preserve simulation/sensor/operator evidence,
clock domains, unknown outcomes and cancellation. ODD MIDI alone cannot prove
pickup, board/bucket contact or capture/miss. No robot control, Viam configuration,
physical dashboard deployment or simulation training is authorized by audio work.
Do not send messages to other chats unless the human explicitly requests it.
