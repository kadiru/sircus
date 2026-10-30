/** Backend-only IPC adapter. stdout is sanitized status; env holds Mini credentials. */
import {createInterface} from 'node:readline';
import {pathToFileURL} from 'node:url';
import {SircusAudio} from './sircus-audio-client.mjs';

export class DashboardAudio {
  constructor({audio, publish=()=>{}, now=Date.now}) {
    this.audio=audio;this.publish=publish;this.now=now;this.revision=0;
    this.enabled=false;this.requested=false;this.volume=.35;this.connection='connecting';
    this.pending=0;this.error=null;this.muteConfirmed=null;this.retired=new Set();
    this.controlQueue=[];this.controlling=false;this.enableInFlight=false;this.ambiguousEnable=false;
    this.latestFX=-1;this.latestVolume=-1;this.activeControl=null;
    this.audio.onError=()=>{this.error='audio_event_failed';this.send();};
  }
  send(){this.publish({revision:this.revision,status:{enabled:this.enabled,requested_enabled:this.requested,pending:this.pending>0,
    connection:this.connection,error:(!this.requested&&this.muteConfirmed===false&&!this.pending)?
      'remote_mute_unconfirmed':this.error,mute_confirmed:this.muteConfirmed}});}
  async poll(){
    const revision=this.revision;
    const controlWasPending=this.controlling||this.pending>0;
    try{
      const status=await this.audio.status();
      // connected in Ghost Booth means MIDI, not HTTP. Successful HTTP is reachability.
      if(revision===this.revision){
        this.connection='connected';
        // A response sampled before an ON/OFF acknowledgment must not undo it.
        // Only reconcile an idle observation; controls own their pending intent.
        if(!controlWasPending&&!this.controlling&&this.pending===0){
          if(this.requested && status.fx_enabled===false){
            this.enabled=false;this.requested=false;
            this.muteConfirmed=!this.ambiguousEnable;
            this.error='audio_disabled_remotely';
            this.audio.cancel(); // Retire queued cues; never automatically re-enable.
          }else if(!this.requested && status.fx_enabled===true){
            // Another Mini client can enable playback after our earlier OFF.
            // Keep local opt-in off while withdrawing the obsolete mute claim.
            this.enabled=false;this.muteConfirmed=false;this.error='remote_mute_unconfirmed';
          }
        }
      }
    }catch{if(revision===this.revision){this.connection='error';this.error='audio_unavailable';}}
    this.send();
  }
  cancelCues(){
    // The supplied client cancels every active request, including controls. Reassert
    // an interrupted explicit OFF after a scoped stop; never let cancellation eat mute.
    const item=this.activeControl;
    if(this.enableInFlight)this.ambiguousEnable=true;
    if(item?.op==='fx'&&!item.enabled&&item.revision===this.latestFX&&
       !this.controlQueue.some(row=>row.op==='fx'&&row.revision===item.revision)){
      this.controlQueue.unshift(item);
    }
    this.audio.cancel();
  }
  handle(message){
    if(!message || typeof message!=='object')return;
    const revision=message.revision;
    if(!Number.isSafeInteger(revision) || revision<0)return;
    if(message.op==='emit'){
      const event=message.event;
      if(!this.enabled || !this.requested || revision<this.revision || !event ||
         event.schema!=='sircus.audio.v1' || this.retired.has(event.trial_id) ||
         !Number.isFinite(message.queued_unix_ms) || this.now()-message.queued_unix_ms>1000)return;
      if(event.event_type==='outcome' && (event.outcome==='unknown' ||
         (event.source!=='simulation'&&!event.confirmed)))return;
      this.audio.emit(event); // Required immediate, immutable, bounded client enqueue.
      return;
    }
    if(!['fx','stop','volume'].includes(message.op))return;
    if(message.op==='fx'){
      if(typeof message.enabled!=='boolean'||revision<=this.latestFX)return;
      this.latestFX=revision;
      this.requested=message.enabled;this.enabled=false;
      this.muteConfirmed=message.enabled?null:false;
      // Abort wait/queued cues now. A timed-out/aborted enable can still execute remotely.
      this.cancelCues();
      this.controlQueue=this.controlQueue.filter(item=>item.op!=='fx');
    } else if(message.op==='stop'){
      if(typeof message.trial_id!=='string'||!message.trial_id||message.trial_id.length>256)return;
      if(this.retired.size>=4096&&!this.retired.has(message.trial_id)){
        this.forceMute(revision);return;
      }
      this.retired.add(message.trial_id);
      this.cancelCues();
    } else {
      if(typeof message.volume!=='number'||!Number.isFinite(message.volume)||message.volume<0||message.volume>1||
         revision<=this.latestVolume)return;
      this.latestVolume=revision;
      this.volume=message.volume;
      this.controlQueue=this.controlQueue.filter(item=>item.op!=='volume');
    }
    this.revision=Math.max(this.revision,revision);
    if(this.controlQueue.length>=32){this.forceMute(revision);return;}
    this.controlQueue.push(message);this.pending=this.controlQueue.length+(this.controlling?1:0);
    this.send();void this.controls();
  }
  forceMute(revision){
    this.revision=Math.max(this.revision,revision);this.latestFX=this.revision;
    this.requested=false;this.enabled=false;this.muteConfirmed=false;
    this.cancelCues();
    this.controlQueue=[{op:'fx',enabled:false,revision:this.latestFX}];
    this.pending=this.controlQueue.length+(this.controlling?1:0);
    this.error='audio_control_capacity';this.send();void this.controls();
  }
  current(item){
    return item.op==='stop'||item.revision===(item.op==='fx'?this.latestFX:this.latestVolume);
  }
  async controls(){
    if(this.controlling)return;
    this.controlling=true;
    try{
      while(this.controlQueue.length){
        const item=this.controlQueue.shift();
        if(!this.current(item))continue;
        this.activeControl=item;
        this.pending=this.controlQueue.length+1;this.send();
        try{
          if(item.op==='fx'){
            // Apply chosen volume before opt-in; all controls serialize.
            if(item.enabled)await this.audio.setVolume(this.volume);
            if(!this.current(item))continue;
            this.enableInFlight=item.enabled;
            await this.audio.setFX(item.enabled);
          }else if(item.op==='stop')await this.audio.stop(item.trial_id);
          else await this.audio.setVolume(item.volume);
          if(this.current(item)){
            this.connection='connected';
            if(item.op==='fx'){
              this.error=null;
              this.enabled=item.enabled&&this.requested;
              if(!item.enabled){
                this.muteConfirmed=!this.ambiguousEnable;
                if(this.ambiguousEnable)this.error='remote_mute_unconfirmed';
              }
            }
          }
        }catch{
          if(this.enableInFlight)this.ambiguousEnable=true;
          if(this.current(item)){
            this.connection='error';this.error='audio_control_failed';
            if(item.op==='fx'){this.enabled=false;if(!item.enabled)this.muteConfirmed=false;}
          }
        }finally{this.enableInFlight=false;this.activeControl=null;}
      }
    }finally{this.controlling=false;this.pending=0;this.send();}
  }
}

export function runWorker(){
  const audio=new SircusAudio({baseURL:process.env.SIRCUS_AUDIO_URL,
    token:process.env.SIRCUS_AUDIO_TOKEN||''});
  const worker=new DashboardAudio({audio,publish:value=>process.stdout.write(JSON.stringify(value)+'\n')});
  const input=createInterface({input:process.stdin,crlfDelay:Infinity});
  input.on('line',line=>{try{if(line.length<=65536)worker.handle(JSON.parse(line));}catch{/* Never echo inputs/secrets. */}});
  const timer=setInterval(()=>void worker.poll(),3000);timer.unref();
  input.on('close',()=>{clearInterval(timer);audio.cancel();});
  void worker.poll();
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)runWorker();
