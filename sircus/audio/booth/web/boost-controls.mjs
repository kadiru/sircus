/** Explicit native-only gain control. No playback, FX or profile mutations. */
export const BOOST_DB = 9.5424250944;

export class BoostControls {
  constructor({request, onChange = () => {}}) {
    this.request = request; this.onChange = onChange; this.sequence = 0;
    this.localPending = false; this.disconnected = false; this.error = '';
    this.state = null; this.requested = 0;
  }

  observe(status) {
    this.state = status;
    this.disconnected = false;
    if (!this.localPending && Number.isFinite(status.boost_requested_db)) {
      this.requested = status.boost_requested_db;
    }
    this.onChange();
  }

  disconnect() { this.disconnected = true; this.onChange(); }

  view() {
    const s = this.state;
    const supported = Number.isFinite(s?.boost_db) && typeof s?.peak_limiter_ready === 'boolean';
    const pending = this.localPending || s?.boost_pending === true;
    const native = s?.route === 'native', ready = s?.audio_ready === true;
    const applied = supported ? s.boost_db : 0;
    const checked = native && (pending ? this.requested > 0 : applied > 0);
    let text = 'Checking native output…';
    if (this.disconnected) text = 'Boost status unavailable';
    else if (s && !native) text = 'Native output only · Logic route unchanged';
    else if (s && !supported) text = 'Boost unavailable in this service';
    else if (pending) text = this.requested > 0 ? 'Boost pending · waiting for native confirmation' : 'Removing boost · waiting for native confirmation';
    else if (supported && !ready) text = 'Native audio unavailable';
    else if (supported && applied > 0) text = s.peak_limiter_ready
      ? (Math.abs(applied - BOOST_DB) < 0.001 ? '3× gain enabled' : `Boost enabled (+${applied.toFixed(2)} dB)`) + ' · peak limiter ready'
      : 'Boost status uncertain · peak limiter unavailable';
    else if (supported) text = s.peak_limiter_ready ? 'Boost off · peak limiter ready' : 'Boost off · peak limiter unavailable';
    return {checked, pending, text, error: this.error,
      // Every gain change requires verified native readiness, including normal
      // gain. FX mute/Stop are independent and are never disabled here.
      disabled: this.disconnected || !supported || !native || !ready || pending ||
        s.peak_limiter_ready !== true};
  }

  async setBoost(enabled) {
    if (typeof enabled !== 'boolean') throw new Error('Boost must be on or off');
    const s = this.state;
    if (this.localPending || s?.boost_pending === true) throw new Error('Wait for the pending boost change');
    if (this.disconnected || s?.route !== 'native' || s?.audio_ready !== true ||
        !Number.isFinite(s.boost_db) || s.peak_limiter_ready !== true) {
      throw new Error('Native audio with a ready peak limiter is required for boost');
    }
    ++this.sequence;
    this.localPending = true; this.requested = enabled ? BOOST_DB : 0; this.error = '';
    this.onChange();
    try {
      const status = await this.request('boost', {boost_db: this.requested});
      // A successful HTTP request is not proof that native output applied gain.
      // Display the returned applied value/pending state until native readback.
      if (!status || !Number.isFinite(status.boost_db) || typeof status.boost_pending !== 'boolean') {
        throw new Error('Native boost confirmation is unavailable');
      }
      this.state = status;
      this.disconnected = false;
      this.requested = Number.isFinite(status.boost_requested_db) ? status.boost_requested_db : this.requested;
    } catch (error) {
      this.disconnected = true;
      this.error = 'Boost change was not confirmed. Check audio status before retrying.';
      throw error;
    } finally {
      this.localPending = false;
      this.onChange();
    }
  }
}
