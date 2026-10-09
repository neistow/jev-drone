"""Jev tactical layer: the drone's programmable common sense.

Everything a human should review lives at the top of this file: the questions,
the option rubrics, and the thresholds. Nothing else in the project hard-codes a
number that changes how the aircraft reacts to a judgment.

Runs off the control loop in a worker thread. The flight code never blocks on it
and never *needs* it -- it reads whatever judgment is currently cached, and the
reflex layer in run.py owns safety regardless of what comes back.
"""
import os, threading, queue, time
import numpy as np
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

MODEL = "jev-latest"

# --- how the rover reacts to a judgment -------------------------------
THRESHOLDS = {
    "stale_after_s": 1.5,     # ignore a judgment older than this
    "risk_slow_down": 1.45,   # score above which we bleed speed
    "call_hz": 3.0,           # upper bound on how often we ask (API SAFETY)
    "call_budget": 160,       # hard cap per episode, never burn credits (API SAFETY)
    "consult_within_m": 4.0,  # only ask when something is actually in the way...
    "consult_lost_s": 1.0,    # ...or the target has been missing this long
    "commit_steps": 6,        # ~0.5s at 12.5Hz: commit to a maneuver
    "override_risk": 1.7,     # ...unless things get this dangerous
    "really_lost": 0.5,       # Noul above which we stop trusting the remembered bearing
}

# The rover's own capabilities.
ROVER = {
    "type": "differential drive rover (car), camera-only, no map and no GPS",
    "top_speed_mps": 3.6,
    "height_m": 0.35,
    "note": "All distances are from a forward camera. 'sector_range_m' is distance to the closest object in each sector. 'sector_clearance_m' is the height of that object's lowest point above the ground. If an object has a clearance > 0.4m (e.g. a high ceiling), the rover can safely drive UNDER it. If the nearest obstacle is 'cardboard', the rover is strong enough to safely smash through it.",
}

MISSION = ("Follow the target rover and keep it in view. Do not hit anything. "
           "You are a ground vehicle and cannot fly over obstacles.")

MANEUVERS = {
    "hold_course": (
        "Either the path ahead is completely clear, or the objects ahead have high clearance (>0.4m), or the obstacle is 'cardboard' and can be smashed. Keep driving straight at the target. "
        "Also use this when waiting for a moving obstacle to clear your path."),
    "steer_left": (
        "A low-clearance obstacle blocks the way ahead, and the left sectors show clearly more free "
        "space (or higher clearance) than the right. Steer around it to the left."),
    "steer_right": (
        "A low-clearance obstacle blocks the way ahead, and the right sectors show clearly more free "
        "space (or higher clearance) than the left. Steer around it to the right."),
    "brake": (
        "Close to a low-clearance obstacle on several sides and no option is clearly better, or waiting "
        "for a dynamic/moving obstacle (like a sliding wall) to move out of the way. Bleed off "
        "speed and hold until the picture improves."),
    "reverse": (
        "Blocked in front by a low-clearance obstacle and too close to turn safely. Back up to create space."
    ),
    "reacquire": (
        "The target has been out of sight long enough that the remembered bearing is "
        "stale. Stop chasing it and sweep to find the target again."),
}

QUESTIONS = {
    "maneuver": Choice(
        instructions={
            "role": "You are the tactical decision layer of an autonomous ground rover.",
            "mission": MISSION,
            "ask": "Which single maneuver should the rover commit to right now?",
        },
        criteria=MANEUVERS,
    ),
    "risk": Score(
        instructions="How dangerous is the rover's immediate situation? NOTE: Objects with >0.4m clearance pose NO RISK because the rover will pass underneath them.",
        criteria=["clear and open", "tight but manageable", "about to hit something"],
    ),
    "target_truly_lost": Noul(
        instructions=(
            "Has the chaser genuinely lost the target? Judge from unseen_for_s: a fraction "
            "of a second behind a pillar is a normal occlusion, several seconds of nothing "
            "in an open scene means the chase line is stale."),
        criteria={"true": "Give up the remembered bearing and sweep to search.",
                  "false": "Keep flying the last known bearing; it should reappear."},
    ),
    "override_safety_reflex": Noul(
        instructions="Should the rover suppress its low-level safety braking reflex? Only do this if you know the obstacle is harmless (like cardboard) and you intend to smash through it.",
        criteria={"true": "Suppress the safety reflex and ram the obstacle.",
                  "false": "Leave safety reflex enabled. Braking will occur if getting too close."},
    ),
}


def decision_needed(scene):
    """Code decides WHEN there is a judgment worth paying for. On an empty corridor
    with the target in view there is nothing to decide, so we do not ask."""
    return (scene["nearest_obstacle_m"] < THRESHOLDS["consult_within_m"]
            or (scene["target"]["unseen_for_s"] or 0.0) >= THRESHOLDS["consult_lost_s"])


def build_state(scene):
    """Everything the model is allowed to know, and nothing it cannot observe."""
    return {"mission": MISSION, "rover": ROVER, "observed": scene}


DEFAULT = {"maneuver": "hold_course", "risk": 0.0, "confidence": 0.0,
           "target_truly_lost": 0.0, "override_safety_reflex": 0.0, 
           "source": "default", "age_s": 0.0,
           "probabilities": {}}


class Tactician:
    """Asks Jev for a judgment at most `hz` times a second, and only when the
    scene has actually changed enough to be worth a call."""

    def __init__(self, hz=THRESHOLDS["call_hz"], budget=THRESHOLDS["call_budget"], model=MODEL):
        key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not key:
            raise RuntimeError("set TYPESAFE_API_KEY (see .env.example)")
        self.client = TypeSafeClient(api_key=key)
        self.min_dt = 1.0 / hz
        self.budget = budget
        self.model = model
        self.calls = 0
        self.attempts = 0
        self.skipped = 0
        self.errors = 0
        self.n_offer = 0
        self.n_ratelimited = 0
        self.n_full = 0
        self.last_error = None
        self.tokens = 0
        self.latency = []
        self._q = queue.Queue(maxsize=1)
        self._latest = dict(DEFAULT)
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._last_sent = float("-inf")
        self._last_key = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @staticmethod
    def _key(scene):
        """Coarse fingerprint: only re-ask when the situation is materially new."""
        t = scene["target"]
        return (
            scene["sectors_blocked"],
            tuple(min(int(v / 1.5), 6) for v in scene["sector_range_m"]),
            tuple(min(int(v / 0.5), 4) for v in scene.get("sector_clearance_m", [])),
            min(int(scene["nearest_obstacle_m"] / 1.5), 6),
            t["visible"],
            None if not t["visible"] else int(np.rad2deg(t["bearing_rad"]) / 15),
            (t.get("unseen_for_s") or 0) > 2.0,
        )

    def offer(self, scene, now):
        """Non-blocking. Hand the latest scene over if it's worth a call."""
        self.n_offer += 1
        if self.attempts >= self.budget or now - self._last_sent < self.min_dt:
            self.n_ratelimited += 1
            return
        key = self._key(scene)
        if key == self._last_key:
            self.skipped += 1
            return
        try:
            self._q.put_nowait((scene, now))
            self._last_sent, self._last_key = now, key
        except queue.Full:
            self.n_full += 1

    def read(self, now):
        with self._lock:
            out = dict(self._latest)
        out["age_s"] = round(now - self._stamp, 2)
        return out

    def _worker(self):
        while not self._stop.is_set():
            try:
                scene, now = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            self.attempts += 1  # counts against the budget whether or not the call succeeds
            try:
                r = self.client.system_one(state=build_state(scene), model=self.model, questions=QUESTIONS)
                a = r.answers
                judgment = {
                    "maneuver": a["maneuver"].choice,
                    "confidence": round(a["maneuver"].confidence, 3),
                    "probabilities": {k: round(v, 3) for k, v in a["maneuver"].probabilities.items()},
                    "risk": round(a["risk"].score, 2),
                    "target_truly_lost": round(a["target_truly_lost"].noul, 3),
                    "override_safety_reflex": round(a["override_safety_reflex"].noul, 3),
                    "source": "jev",
                }
                self.tokens += r.usage.input_tokens + r.usage.output_tokens
                self.calls += 1
                self.latency.append(time.time() - t0)
                with self._lock:
                    self._latest, self._stamp = judgment, now
            except Exception as e:                    # degrade, never crash the flight
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"[:160]
                # A failed call replaces the cached judgment with an error placeholder, so the
                # next offer() for this same scene must not be skipped as "unchanged" - otherwise
                # a static scene never gets re-asked and never recovers a real judgment.
                self._last_key = None
                with self._lock:
                    self._latest = dict(DEFAULT, source=f"error:{type(e).__name__}")

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.client.close()
        except Exception:
            pass

    def stats(self):
        lat = sorted(self.latency)
        return {"calls": self.calls, "attempts": self.attempts, "skipped_unchanged": self.skipped, "errors": self.errors,
                "last_error": self.last_error, "tokens": self.tokens,
                "offers": self.n_offer, "rate_limited": self.n_ratelimited, "queue_full": self.n_full,
                "median_latency_s": round(lat[len(lat) // 2], 3) if lat else None,
                "p90_latency_s": round(lat[int(len(lat) * 0.9)], 3) if lat else None}
