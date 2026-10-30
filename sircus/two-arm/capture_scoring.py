"""Versioned whole-sphere capture events. Inputs: world mm and simulated seconds.

No I/O, robot clients or training. Contact fixtures never become shot successes.
The caller supplies an observed board contact and its outgoing-bounce signal.
"""
import math

from physics import bucket_contains_sphere, bucket_profile

SCORING_VERSION = 'sircus-board-bounce-capture-1s-v1'
PHYSICS_VERSION = 'sircus-compliant-board-bucket-impulse-v1'
RETENTION_S = 1.0
ROUND_OFF_MM = 1e-6
SUCCESS_DEFINITION = ('observed board bounce before whole-sphere entry through the open rim; '
                      'continuous whole-sphere containment >=1 simulated second; no model limit')


class CaptureTracker:
    def __init__(self, config, *, episode_kind='complete_shot', entry_deadline_s=4.):
        if episode_kind not in ('complete_shot', 'contact_fixture'):
            raise ValueError('Unknown episode_kind')
        if not math.isfinite(entry_deadline_s) or entry_deadline_s <= 0:
            raise ValueError('Positive finite entry deadline required')
        self.config = config
        self.profile = bucket_profile(config)
        self.radius = config['ball']['diameter_mm']/2
        self.episode_kind = episode_kind
        self.entry_deadline_s = entry_deadline_s
        self.previous = None
        self.events = []
        self.crossings = []
        self.inside_since = None
        self.max_retention = 0.
        self.entry = False
        self.valid_opening = False
        self.bounce_outs = 0
        self.full_exits = 0
        self.was_fully_above = True
        self.board_bounce_t = None
        self.first_contact_xyz = None
        self.entry_after_bounce = False
        self.rim_contact_t = None
        self.entry_after_rim = False
        self.success = False
        self.invalid = False

    def note_contact(self, name, t):
        if name == 'bucket-rim' and self.rim_contact_t is None:
            self.rim_contact_t = t

    def update(self, t, xyz, *, board_bounce=False, board_contact_xyz=None, invalid=False):
        if not math.isfinite(t) or len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise ValueError('Finite time and world position required')
        if self.previous and t < self.previous[0]:
            raise ValueError('Physics samples must be time ordered')
        self.invalid |= invalid
        if self.first_contact_xyz is None and board_contact_xyz is not None:
            self.first_contact_xyz = list(board_contact_xyz)
        if board_bounce and self.first_contact_xyz is not None and self.board_bounce_t is None:
            self.board_bounce_t = t
            self.events.append({'event':'board_bounce', 't':t})
        top = self.profile['rim_z_mm']
        if self.previous:
            pt, prev = self.previous
            if prev[2] > top >= xyz[2]:
                fraction = (prev[2]-top)/(prev[2]-xyz[2])
                point = [prev[i]+fraction*(xyz[i]-prev[i]) for i in range(3)]
                delta = [point[i]-self.profile['center_xy_mm'][i] for i in (0,1)]
                clearance = self.profile['inner_top_radius_mm']-self.radius-math.hypot(*delta)
                crossing = {'t':pt+fraction*(t-pt), 'xyz_mm':point, 'signed_miss_xy_mm':delta,
                            'clearance_mm':clearance, 'through_opening':clearance >= -ROUND_OFF_MM}
                self.crossings.append(crossing)
                self.events.append({'event':'downward_opening_plane_crossing', **crossing})
                self.valid_opening = crossing['through_opening']
        inside = bucket_contains_sphere(self.config, xyz, self.radius, tolerance_mm=ROUND_OFF_MM)
        if inside and self.valid_opening:
            if self.inside_since is None and t <= self.entry_deadline_s+1e-9:
                self.inside_since = t
                self.entry = True
                self.entry_after_bounce = self.board_bounce_t is not None and self.board_bounce_t <= t
                self.entry_after_rim = self.rim_contact_t is not None and self.rim_contact_t <= t
                self.events.append({'event':'whole_ball_entry', 't':t, 'xyz_mm':list(xyz)})
            if self.inside_since is not None:
                self.max_retention = max(self.max_retention, t-self.inside_since)
                if (t-self.inside_since >= RETENTION_S-1e-9 and self.entry_after_bounce
                        and self.episode_kind == 'complete_shot' and not self.invalid):
                    self.success = True
        elif self.inside_since is not None:
            self.events.append({'event':'containment_lost', 't':t, 'xyz_mm':list(xyz)})
            if xyz[2]+self.radius > top+ROUND_OFF_MM:
                self.bounce_outs += 1
                self.events.append({'event':'bounce_out', 't':t})
            self.inside_since = None
        fully_above = xyz[2]-self.radius > top+ROUND_OFF_MM
        if fully_above:
            if self.entry and not self.was_fully_above:
                self.full_exits += 1
                self.events.append({'event':'full_exit', 't':t})
            self.valid_opening = False
        # Partial protrusion resets dwell but retains opening eligibility: the
        # center may never rise above the plane and hence cannot cross it again.
        self.was_fully_above = fully_above
        self.previous = (t, list(xyz))

    def current_retention(self, t):
        return max(0., t-self.inside_since) if self.inside_since is not None else 0.

    def done(self, t):
        return (self.current_retention(t) >= RETENTION_S-1e-9 or
                (t >= self.entry_deadline_s and self.inside_since is None))

    def result(self, t, *, termination='horizon', model_limit=None):
        held = self.current_retention(t) >= RETENTION_S-1e-9 and not self.invalid and not model_limit
        success = self.success and not self.invalid and not model_limit
        if model_limit:
            outcome = 'model_limit'
        elif self.invalid:
            outcome = 'invalid_contact'
        elif success:
            outcome = 'captured'
        elif held:
            outcome = 'fixture_retained' if self.episode_kind == 'contact_fixture' else 'no_board_bounce'
        elif self.inside_since is not None:
            outcome = 'timeout'  # Hard horizon cut a current retention interval short.
        elif self.entry and self.bounce_outs:
            outcome = 'bounce_out'
        elif self.entry:
            outcome = 'containment_lost'
        elif termination == 'escaped' or t >= self.entry_deadline_s:
            outcome = 'no_entry'
        else:
            outcome = 'timeout'
        crossing = max(self.crossings, key=lambda x:x['clearance_mm']) if self.crossings else None
        return {'scoring_version':SCORING_VERSION, 'physics_version':PHYSICS_VERSION,
                'episode_kind':self.episode_kind, 'success':bool(success), 'outcome':outcome,
                'capture_class':('rim_assisted' if self.entry_after_rim else 'clean') if success else None,
                'bucket_retained_1s':bool(held), 'whole_ball_entry':self.entry,
                'valid_board_bounce':self.board_bounce_t is not None,
                'board_bounce_t_s':self.board_bounce_t, 'bounce_outs':self.bounce_outs,
                'full_exits':self.full_exits, 'max_continuous_containment_s':self.max_retention,
                'current_containment_s':self.current_retention(t),
                'opening_crossings':self.crossings, 'task_events':self.events,
                'opening_clearance_mm':crossing['clearance_mm'] if crossing else None,
                'signed_miss_xy_mm':crossing['signed_miss_xy_mm'] if crossing else None,
                'first_contact_ball_center_mm':self.first_contact_xyz,
                'termination_reason':termination, 'truncated':outcome == 'timeout',
                'success_definition':SUCCESS_DEFINITION}
