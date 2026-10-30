/** Ordered, explicit audio-only controls shared by the local panel and backend client. */
export class FXControls {
  constructor({request, clientId}) {
    this.request=request;this.clientId=clientId;this.sequence=0;
  }
  invalidate(){this.sequence++;}
  async setFX(enabled){
    if(typeof enabled!=='boolean')throw new Error('enabled must be boolean');
    return this.control('fx',{enabled});
  }
  panic(){return this.control('panic',{});}
  async control(path,body){
    const sequence=++this.sequence;
    if(!Number.isSafeInteger(sequence))throw new Error('Audio control sequence exhausted');
    const metadata={control_client_id:this.clientId,control_sequence:sequence};
    if(path==='fx'&&body.enabled){
      // Only explicit ON waits for status. OFF/panic remain immediately available.
      const status=await this.request('status');
      if(sequence!==this.sequence)throw new Error('Audio enable superseded by a newer control');
      if(typeof status.control_epoch!=='string'||!status.control_epoch)throw new Error('Audio service does not support ordered FX controls');
      metadata.control_epoch=status.control_epoch;
    }
    return this.request(path,{...body,...metadata});
  }
}
