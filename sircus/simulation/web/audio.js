// Audio is an optional observer. This module never reads or changes robot controls.
export class AudioPanel {
  constructor({document, fetchImpl = globalThis.fetch.bind(globalThis),
    setTimer = (callback, delay) => globalThis.setTimeout(callback, delay),
    clearTimer = timer => globalThis.clearTimeout(timer),
    clientId = globalThis.crypto.randomUUID()}) {
    this.document = document; this.fetch = fetchImpl;
    this.setTimer = setTimer; this.clearTimer = clearTimer;
    this.elements = Object.fromEntries(['audioPanel', 'audioFX', 'audioFXLabel',
      'audioVolume', 'audioVolumeValue', 'audioConnection', 'audioMessage',
      'audioMuteWarning'].map(id => [id, document.getElementById(id)]));
    this.state = {configured: null, enabled: false, requested_enabled: false,
      volume: 0.35, connection: 'checking', pending: false, error: null};
    this.token = null; this.revision = 0; this.requests = 0; this.destroyed = false;
    this.clientId = clientId; this.fxIntent = null;
    this.polling = false; this.timer = null; this.muteUnconfirmed = false;
    this.listeners = [];
    this.listen('audioFX', 'change', () => {
      const enabled = this.elements.audioFX.checked;
      void this.control({action: 'fx', enabled});
    });
    this.listen('audioVolume', 'input', () => {
      this.elements.audioVolumeValue.textContent = `${Math.round(Number(this.elements.audioVolume.value) * 100)}%`;
    });
    this.listen('audioVolume', 'change', () => {
      const volume = Number(this.elements.audioVolume.value);
      if (Number.isFinite(volume) && volume >= 0 && volume <= 1) {
        this.state.volume = volume;
        void this.control({action: 'volume', volume});
      }
    });
    this.render();
  }

  listen(id, name, listener) {
    this.elements[id].addEventListener(name, listener);
    this.listeners.push([id, name, listener]);
  }

  async request(path, body) {
    const controller = new AbortController();
    const timer = this.setTimer(() => controller.abort(), 2500);
    try {
      const response = await this.fetch(path, {method: body ? 'POST' : 'GET',
        cache: 'no-store', signal: controller.signal,
        ...(body ? {headers: {'Content-Type': 'application/json',
          'X-Sircus-Audio-Token': this.token}, body: JSON.stringify(body)} : {})});
      if (!response.ok) throw new Error('Audio request unavailable');
      const value = await response.json();
      if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid audio status');
      return value;
    } finally { this.clearTimer(timer); }
  }

  accept(value) {
    this.state = {...this.state, ...value};
    if (typeof value.action_token === 'string') this.token = value.action_token;
    // A transport failure cannot prove remote playback stopped. Keep this warning
    // until the backend positively acknowledges mute; a later connected poll alone
    // must not erase the uncertainty.
    if (value.mute_confirmed === false) this.muteUnconfirmed = true;
    if (value.mute_confirmed === true && value.enabled === false) this.muteUnconfirmed = false;
    if (this.fxIntent === false) {
      // An ambiguous OFF request remains the operator's intent. A later poll
      // showing playback still ON is a mute warning, never a fresh ON action.
      this.state.requested_enabled = false; this.state.enabled = false;
      if (value.enabled === true) this.muteUnconfirmed = true;
    } else if (value.enabled === true) this.muteUnconfirmed = false;
    this.render();
  }

  render() {
    const s = this.state, e = this.elements;
    const requested = s.requested_enabled === true;
    const pending = this.requests > 0 || s.pending === true;
    e.audioFX.checked = requested;
    // OFF remains available during pending requests and transport loss. It only
    // affects audio; nothing in this panel can disable or wait on robot Stop.
    e.audioFX.disabled = !s.configured || !this.token;
    e.audioFXLabel.textContent = s.enabled === true ? 'on' : requested ? (pending ? 'pending' : 'unavailable') : 'off';
    e.audioVolume.disabled = !s.configured || !this.token || pending;
    if (this.document.activeElement !== e.audioVolume && Number.isFinite(s.volume) && s.volume >= 0 && s.volume <= 1) {
      e.audioVolume.value = String(s.volume);
      e.audioVolumeValue.textContent = `${Math.round(s.volume * 100)}%`;
    }
    const names = {checking: 'Checking audio…', connected: 'Audio service connected',
      ready: 'Audio service connected', disconnected: 'Audio unavailable', error: 'Audio error',
      connecting: 'Connecting audio…', unconfigured: 'Audio not configured', not_configured: 'Audio not configured',
      disabled: 'Audio not configured', unavailable: 'Audio unavailable'};
    e.audioConnection.textContent = s.configured === false && s.connection !== 'checking'
      ? 'Audio not configured' : names[s.connection] || 'Audio status unknown';
    e.audioMessage.textContent = s.error ? String(s.error)
      : pending ? 'Audio request pending. Robot controls remain available.'
      : !s.configured ? 'Backend audio setup is pending. FX stays off.'
      : s.enabled ? 'FX enabled for physical trial cues. Unknown outcomes stay silent.'
      : 'FX off. Robot controls work independently.';
    e.audioPanel.classList.toggle('audio-error', Boolean(s.error) || this.muteUnconfirmed);
    e.audioMuteWarning.hidden = !this.muteUnconfirmed;
  }

  async control(payload) {
    if (!this.token || !this.state.configured || this.destroyed) { this.render(); return; }
    const revision = ++this.revision;
    if (payload.action === 'fx') {
      this.fxIntent = payload.enabled;
      this.state.requested_enabled = payload.enabled;
      if (!payload.enabled) {this.state.enabled = false; this.muteUnconfirmed = true;}
    }
    this.requests++; this.state.error = null; this.render();
    try {
      const value = await this.request('/api/audio/control', {
        ...payload, client_id: this.clientId, sequence: revision});
      if (!this.destroyed && revision === this.revision) this.accept(value);
    } catch {
      if (!this.destroyed && revision === this.revision) {
        if (payload.action === 'fx') {
          this.fxIntent = false;
          this.state.enabled = false; this.state.requested_enabled = false;
          this.muteUnconfirmed = true;
        }
        this.state.connection = 'disconnected';
        this.state.error = 'Audio request failed. Robot controls are unaffected.';
      }
    } finally {
      this.requests--;
      if (!this.destroyed) this.render();
    }
  }

  async poll() {
    if (this.destroyed || this.polling || this.requests) return;
    this.polling = true;
    const revision = this.revision;
    try {
      const value = await this.request('/api/audio/status');
      if (!this.destroyed && revision === this.revision && !this.requests) this.accept(value);
    } catch {
      if (!this.destroyed && revision === this.revision && !this.requests) {
        this.state.connection = 'disconnected';
        this.state.error = 'Audio status unavailable. Robot controls are unaffected.';
        this.render();
      }
    } finally { this.polling = false; }
  }

  start() {
    const tick = async () => {
      await this.poll();
      if (!this.destroyed) this.timer = this.setTimer(tick, 1500);
    };
    void tick();
    return this;
  }

  destroy() {
    this.destroyed = true; this.clearTimer(this.timer);
    for (const [id, name, listener] of this.listeners) this.elements[id].removeEventListener(name, listener);
  }
}

if (typeof document !== 'undefined' && document.getElementById('audioPanel') && !document.documentElement.hasAttribute('data-show-shell')) {
  new AudioPanel({document}).start();
}
