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

At each safe task boundary, fetch origin and review newer main changes. Keep
`codex/mac-mini-oddball-sfx` while its audio PR remains open; merge newer main
without rewriting published history. After it is merged, create a new audio task
branch from updated main. Preserve unrelated work in place and use a separate
worktree when needed; never auto-stash, hard-reset, clean or force-push.

At coherent checkpoints, explicitly stage the reviewed audio files, run relevant
tests and repository file/credential checks, commit, push and update the existing
audio PR. Keep it draft while listening or integration review remains outstanding;
do not merge it. MacBook Codex owns integration and shared dashboard/controller
edits. Synchronization never authorizes service restarts or playback. Report the
pushed SHA, fetched main SHA, ancestry, PR state, tests and remaining blockers.
