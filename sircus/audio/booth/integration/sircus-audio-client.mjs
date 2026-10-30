/** Optional server-side audio observer. Never await a cue on a robot control path. */
import {randomUUID} from 'node:crypto';
export class SircusAudio {
  constructor({baseURL,token,fetchImpl=fetch,onError=()=>{},maxQueue=32,timeoutMs=350}) {
    this.baseURL=baseURL.replace(/\/$/,'');this.token=token;this.fetch=fetchImpl;this.onError=onError;
    this.maxQueue=maxQueue;this.timeoutMs=timeoutMs;this.queue=[];this.running=false;this.active=new Set();this.generation=0;
  }
  report(error){try{this.onError(error);}catch{/* Audio diagnostics must not break the producer. */}}
  async request(path,body){
    const controller=new AbortController();this.active.add(controller);const timer=setTimeout(()=>controller.abort(),this.timeoutMs);
    try {
      const response=await this.fetch(this.baseURL+'/api/'+path,{method:body===undefined?'GET':'POST',headers:{Authorization:'Bearer '+this.token,...(body===undefined?{}:{'Content-Type':'application/json'})},...(body===undefined?{}:{body:JSON.stringify(body)}),signal:controller.signal});
      if(!response.ok)throw new Error(`Audio HTTP ${response.status}`);
      const result=await response.json();if(result.accepted===false)throw new Error('Audio cue rejected: '+result.reason);return result;
    } finally {clearTimeout(timer);this.active.delete(controller);}
  }
  cancel(){this.generation++;this.queue=[];for(const request of this.active)request.abort();}
  // Operator/UI controls may await their own response; cue producers never do.
  setFX(enabled){if(!enabled)this.cancel();return this.request('fx',{enabled});}
  stop(trialId){this.cancel();return this.request('stop',trialId?{trial_id:trialId}:{});}
  setVolume(volume){return this.request('volume',{volume});}
  status(){return this.request('status');}
  emit(event){
    try {
      if(this.queue.length>=this.maxQueue)throw new Error('Audio queue full; cue dropped');
      // Snapshot evidence now; later producer mutation must not change the queued observation.
      const body=JSON.parse(JSON.stringify(event));
      this.queue.push({created:Date.now(),generation:this.generation,body});void this.pump();
    }catch(error){this.report(error);}
  }
  // Compatibility helper for explicit operator actions only. Prefer emit(AudioEvent).
  event(type,{trialId,eventId,velocity=85}={}){this.emit({type,trial_id:trialId,event_id:eventId??randomUUID(),velocity});}
  async pump(){
    if(this.running)return;this.running=true;
    try {
      while(this.queue.length){
        const item=this.queue.shift();if(item.generation!==this.generation)continue;
        if(Date.now()-item.created>1000){this.report(new Error('Stale audio cue dropped'));continue;}
        try{await this.request('event',item.body);}catch(error){this.report(error);}
      }
    }finally{this.running=false;}
  }
}
