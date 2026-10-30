// Optional Scripter before a manually commissioned Sampler. Select ONLY Sircus FX.
// No Trace calls, raw CC routing or robot actions. Native audio is the main route.
var enabled=false;
var notes=[36,37,38,39,40,41,42,43,44,45,48,49,50,51,52,53];
function mapped(n){return notes.indexOf(n)!==-1;}
function cc(number,value){var e=new ControlChange();e.channel=16;e.number=number;e.value=value;e.send();}
function kill(){enabled=false;cc(11,0);cc(64,0);for(var i=0;i<notes.length;i++){var e=new NoteOff();e.channel=16;e.pitch=notes[i];e.send();}cc(120,0);cc(123,0);}
function HandleMIDI(e){
  if(e.channel!==16)return;
  if(e instanceof ControlChange){
    if(e.number===11){enabled=e.value>0;e.send();if(!enabled)kill();}
    else if(e.number===120||e.number===123)kill();
    else if(e.number===64&&e.value===0)e.send();
    return;
  }
  if(e instanceof NoteOff){if(mapped(e.pitch))e.send();return;}
  if(e instanceof NoteOn&&enabled&&mapped(e.pitch))e.send();
}
function Reset(){kill();}
