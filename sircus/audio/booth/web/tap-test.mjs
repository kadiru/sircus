/** Audio-only preflight. Countdown/expiry are owned by the server, not this tab. */
export class TapTestControls {
  constructor({request, clientId}) {
    this.request=request;this.clientId=clientId;this.sequence=0;this.pending=false;
  }
  async start(duration=15) {
    if (!Number.isInteger(duration)||duration<1||duration>120) throw new Error('Choose 1–120 whole seconds');
    if (this.pending) throw new Error('A tap-test request is already pending');
    const seq=++this.sequence;this.pending=true;
    try {
      const s=await this.request('status');
      if(seq!==this.sequence)throw new Error('Tap test cancelled');
      if(!s.tap_test?.supported)throw new Error('Tap tester is unavailable in this service');
      return await this.request('tap-test',{action:'start',duration_s:duration,
        control_client_id:this.clientId,control_sequence:seq,control_epoch:s.control_epoch,
        show_config_digest:s.show_audio?.config_digest});
    } catch(error) {
      // A lost HTTP response does not imply the server rejected Start. Mute on
      // ambiguity, never retry Start; server expiry remains the final bound.
      if(seq===this.sequence)await this.stop().catch(()=>{});
      throw error;
    } finally {if(seq===this.sequence)this.pending=false;}
  }
  async stop() {
    const seq=++this.sequence;this.pending=false;
    return this.request('tap-test',{action:'stop',control_client_id:this.clientId,control_sequence:seq});
  }
}

export function testView(s) {
  const t=s?.tap_test, r=s?.show_audio;
  const ready=t?.supported&&s.audio_ready&&s.route==='native'&&r?.selection_matches&&r.connected_ball_count>0;
  return {canStart:!!ready&&!t.active,active:!!t?.active,
    text:t?.active?`Tap all four balls · ${Math.ceil(t.remaining_s)} seconds left`:
      t?.test_id?`${t.reason==='tap_test_completed'?'Test finished':`Test ended: ${(t.reason||'stopped').replaceAll('_',' ')}`} · ${t.balls.filter(b=>b.taps>0).length}/${t.balls.length} responded`:
      ready?'Ready · Start test enables sound for the selected duration.':'Waiting for mapped balls and native audio.',
    balls:(t?.balls||[]).map(b=>({...b,detail:!b.bound?'Not mapped':!b.connected?'Disconnected':b.taps>0?`${b.taps} tap${b.taps===1?'':'s'} · last ${b.last_tap_age_s}s ago`:t.active?'Waiting for tap':t.test_id?'No tap detected':'Connected · not tested',
      tone:!b.connected?'offline':b.taps>0?'responding':'waiting'}))};
}
