/* Sircus SFX receiver: MIDI FX > Scripter before stock Retro Synth.
 * Derived from the Mac Mini's tested logging-free ODD note ownership/reset path.
 * Choose ONLY Sircus SFX as this track's MIDI input; never All/ODD.
 * Incoming cue notes/pitch bends are channel 16; instrument output is channel 1.
 * Retro Synth pitch bend range must be +/-12 semitones. No samples required.
 * No Trace, no sendAfterMilliseconds: cancellation belongs to the adapter.
 */
var PluginParameters = [
    {name:"Input channel", type:"lin", minValue:1, maxValue:16, numberOfSteps:15, defaultValue:16},
    {name:"Output channel", type:"lin", minValue:1, maxValue:16, numberOfSteps:15, defaultValue:1},
    {name:"Mute and release", type:"menu", valueStrings:["Off","On"], defaultValue:0}
];
var active = {};
function release(key) {
    var voice = active[key];
    if (!voice) return;
    var off = new NoteOff();
    off.pitch = voice.pitch; off.channel = voice.channel; off.velocity = 0;
    off.send(); delete active[key];
}
function Reset() {
    Object.keys(active).forEach(release);
    var channel = GetParameter("Output channel");
    [120,123].forEach(function (number) {
        var panic = new ControlChange();
        panic.channel = channel; panic.number = number; panic.value = 0; panic.send();
    });
    var bend = new PitchBend(); bend.channel = channel; bend.value = 0; bend.send();
}
function ParameterChanged(index, value) { Reset(); }
function HandleMIDI(event) {
    if (event.channel !== GetParameter("Input channel")) return;
    if (event instanceof ControlChange && (event.number === 120 || event.number === 123)) {
        Reset(); return;
    }
    if (GetParameter("Mute and release")) return;
    if (event instanceof NoteOn || event instanceof NoteOff) {
        var key = event.channel + ":" + event.pitch;
        if (event instanceof NoteOff || event.velocity === 0) { release(key); return; }
        if (event.pitch < 24 || event.pitch > 96) return;
        release(key);
        var note = new NoteOn();
        note.channel = GetParameter("Output channel"); note.pitch = event.pitch;
        note.velocity = Math.max(1, Math.min(90, Math.round(event.velocity)));
        note.send();
        active[key] = {channel:note.channel, pitch:note.pitch, expires:Date.now()+2500};
    } else if (event instanceof PitchBend) {
        var bend = new PitchBend(); bend.channel = GetParameter("Output channel");
        bend.value = Math.max(-8192, Math.min(8191, event.value)); bend.send();
    }
    // Consume all other controllers/notes; never expose raw ODD CCs to the synth.
}
function ProcessMIDI() {
    var now = Date.now();
    Object.keys(active).forEach(function (key) {
        if (now >= active[key].expires) release(key);
    });
}
