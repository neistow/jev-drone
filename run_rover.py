import os, sys, time, argparse
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass
import numpy as np
import mujoco
from drive import Pilot, Eye
from rover_tactics import decision_needed, THRESHOLDS

ROVER_SPEED = 0.85
SEARCH_SPEED = 2.4
TARGET_MAX_SPEED = 1.6
STANDOFF = 3.5

def rover_pose(t):
    return np.array([6.0 + ROVER_SPEED * t, 2.0 * np.sin(0.26 * t), 0.2])

def drive_course(m, d, t):
    d.mocap_pos[m.body("rover").mocapid[0]] = rover_pose(t)
    c = 2.0 * np.sin(0.26 * t)
    for name, off in (("gateL", 31.6), ("gateR", -31.6)):
        mid = m.body(name).mocapid[0]
        p = d.mocap_pos[mid].copy()
        p[1] = c + off
        d.mocap_pos[mid] = p


def baseline_heuristic(scene):
    """A reactive 2D heuristic that uses vertical clearance. 
    It will navigate pillars and go under the overpass, but struggles with the moving wall."""
    # Filter out obstacles that we can drive under (clearance > 0.4m)
    sec = [r if c < 0.4 else 25.0 for r, c in zip(scene["sector_range_m"], scene["sector_clearance_m"])]
    
    # Always prioritize immediate low-clearance obstacle evasion over blind target reacquisition
    if min(sec) < 4.0:
        left_room = min(sec[0], sec[1])
        right_room = min(sec[3], sec[4])
        
        if max(sec) < 2.5:
            return "brake"
            
        if sec[2] < 3.5:
            return "steer_left" if left_room > right_room else "steer_right"
        else:
            if left_room < 1.5: return "steer_right"
            if right_room < 1.5: return "steer_left"

    if scene["target"]["unseen_for_s"] > 1.0:
        return "reacquire"
            
    return "hold_course"

GUIDANCE_MODES = ("layered", "model-only")


class Guidance:
    """Turns a scene (+ the latest cached model judgment) into a velocity command.

    Two tactical arbitration modes (selected with ``--guidance``):

    * ``layered`` (default): ``baseline_heuristic(scene)`` runs continuously as the
      reactive layer underneath the model (matching the drone's cooperative layering
      in ``run.py``). A fresh model judgment overrides with explicit tactical maneuvers
      (``steer_left``, ``steer_right``, ``brake``, ``reverse``) or cardboard smashing
      (``override_safety_reflex``), while passive ``hold_course`` / ``reacquire`` without
      reflex override cooperates with the reactive obstacle-avoidance layer when a solid
      low-clearance obstacle blocks the path.
    * ``model-only`` (legacy): only the model decides. A stale/missing judgment
      makes the rover brake.

    In both modes the low-level safety reflex sits underneath everything and only
    the model's ``override_safety_reflex`` can suppress it.
    """

    def __init__(self, eye, mode="layered"):
        if mode not in GUIDANCE_MODES:
            raise ValueError(f"unknown guidance mode {mode!r}, expected one of {GUIDANCE_MODES}")
        self.eye = eye
        self.mode = mode
        self.yaw_sp = None
        self.sweep = 0.0
        self.sweep_dir = 1.0
        self.tgt_w = None
        self.tgt_v = np.zeros(2)
        self.tgt_t = 0.0
        self.commit = None
        self.commit_left = 0
        self.commit_src = None       # "model" or "heuristic": who owns the current commitment
        self.override_left = 0
        # last tactical arbitration, exposed for logs / HUD
        self.last_mv = None
        self.last_layer = None
        self.last_heuristic_mv = None

    def _search_heading(self, yaw, t, pos):
        if self.tgt_w is None or pos is None:
            return yaw
        lead = float(np.clip(t - self.tgt_t, 0.0, 5.0))
        aim = self.tgt_w + self.tgt_v * lead
        d = aim - pos[:2]
        if np.linalg.norm(d) < 0.5:
            return yaw
        return float(np.arctan2(d[1], d[0]))

    def _tactical_model_only(self, scene, judg):
        """Legacy arbitration: the model alone decides; stale/missing -> brake."""
        acted = 0
        layer = "model"
        if not decision_needed(scene):
            self.override_left = max(0, self.override_left - 1)
            if self.commit in ("steer_left", "steer_right", "reverse") and self.commit_left > 0:
                self.commit_left -= 1
                return self.commit, 1, "model"
            self.commit = None
            self.commit_left = 0
            mv = "hold_course"
            layer = "clear"
        elif judg and judg.get("maneuver"):
            if judg.get("age_s", 0.0) <= THRESHOLDS["stale_after_s"]:
                mv = judg["maneuver"]
                if self.commit is None or self.commit_left <= 0:
                    self.commit = mv
                    self.commit_left = THRESHOLDS["commit_steps"]
                else:
                    mv = self.commit
                    self.commit_left -= 1
                    if judg.get("maneuver") != self.commit and judg.get("risk", 0) > THRESHOLDS.get("override_risk", 1.7):
                        self.commit = judg["maneuver"]
                        mv = self.commit
                
                # Latch the reflex override for ~1.3 seconds (20 ticks) to carry momentum through the obstacle
                if judg.get("override_safety_reflex", 0) > 0.5:
                    self.override_left = 20
                else:
                    self.override_left = max(0, self.override_left - 1)
                    
                acted = 1
            else:
                self.commit = None
                self.commit_left = 0
                self.override_left = max(0, self.override_left - 1)
                mv = "brake"
                layer = "stale-brake"
        else:
            self.override_left = max(0, self.override_left - 1)
            mv = "brake"
            layer = "stale-brake"
        self.commit_src = "model" if self.commit is not None else None
        return mv, acted, layer

    def _commit_to(self, mv, src, risk=0.0):
        """Shared commitment logic: hold a maneuver for commit_steps ticks unless
        risk exceeds override_risk. If the incoming owner agrees on the same maneuver,
        preserve continuity rather than resetting the commitment counter."""
        if self.commit is None or self.commit_left <= 0:
            self.commit, self.commit_left, self.commit_src = mv, THRESHOLDS["commit_steps"], src
            return mv
        if self.commit_src != src:
            if mv == self.commit:
                self.commit_src = src
                self.commit_left = max(1, self.commit_left - 1)
                return self.commit
            self.commit, self.commit_left, self.commit_src = mv, THRESHOLDS["commit_steps"], src
            return mv
        self.commit_left -= 1
        if mv != self.commit and risk > THRESHOLDS.get("override_risk", 1.7):
            self.commit = mv
            self.commit_left = THRESHOLDS["commit_steps"]
        return self.commit

    def _tactical_layered(self, scene, judg, h_mv):
        """Cooperative layered arbitration (matching drone run.py):

        - when a non-holonomic turn temporarily rotates a close pillar out of the camera
          FOV (``not decision_needed(scene)``), hold any remaining ``steer_left`` /
          ``steer_right`` / ``reverse`` commitment ticks so the rover clears the pillar
          instead of whipping its nose back into it;
        - when the model is fresh and commands an explicit tactical maneuver or suppresses
          the safety reflex (e.g. smashing cardboard), the model leads;
        - when the model outputs passive ``hold_course`` or ``reacquire`` without reflex
          override while a solid obstacle blocks the path, cooperate with the reactive
          evasion layer (just as ``hold_course`` in ``run.py`` preserves reactive slide);
        - if a model refresh is briefly in-flight while the model holds an active
          commitment or reflex override latch, finish that commitment before falling back."""
        model_fresh = (bool(judg) and judg.get("source") == "jev" and bool(judg.get("maneuver"))
                       and judg.get("age_s", float("inf")) <= THRESHOLDS["stale_after_s"])

        if not decision_needed(scene):
            self.override_left = max(0, self.override_left - 1)
            if self.commit in ("steer_left", "steer_right", "reverse") and self.commit_left > 0:
                self.commit_left -= 1
                src = self.commit_src or "heuristic"
                return self.commit, (1 if src == "model" else 0), src
            self.commit, self.commit_left, self.commit_src = None, 0, None
            return h_mv, 0, "heuristic"

        if model_fresh:
            # only the model may suppress the safety reflex (e.g. to smash cardboard)
            if judg.get("override_safety_reflex", 0) > 0.5:
                self.override_left = 20
            else:
                self.override_left = max(0, self.override_left - 1)

            j_mv = judg["maneuver"]
            if (judg.get("target_truly_lost", 0.0) >= THRESHOLDS["really_lost"]
                    and j_mv == "hold_course" and not scene["target"]["visible"]
                    and self.override_left == 0):
                j_mv = "reacquire"

            is_cardboard = scene.get("nearest_obstacle_type") == "cardboard"
            # Cooperative obstacle deflection underneath passive hold_course / reacquire:
            # never let a blind search sweep (reacquire) drive into a solid obstacle, and
            # if hold_course has no reflex override while a narrow pillar is close (<2.2m),
            # let the reactive layer deflect around it.
            low_sec = [r if c < 0.4 else 25.0 for r, c in zip(scene["sector_range_m"], scene["sector_clearance_m"])]
            if not is_cardboard and self.override_left == 0 and h_mv in ("steer_left", "steer_right", "brake"):
                if j_mv == "reacquire" or (j_mv == "hold_course" and min(low_sec) < 2.2 and scene.get("sectors_blocked", 0) <= 2):
                    j_mv = h_mv

            mv = self._commit_to(j_mv, "model", judg.get("risk", 0.0))
            if mv in ("steer_left", "steer_right") and min(low_sec) < 2.0:
                # Ensure enough commitment remains to clear the obstacle after it exits the camera FOV
                self.commit_left = max(self.commit_left, 8)
            return mv, 1, "model"

        self.override_left = max(0, self.override_left - 1)
        if self.override_left > 0 and judg and judg.get("maneuver"):
            return judg["maneuver"], 1, "model"
        if self.commit_src == "model" and self.commit_left > 0 and self.commit is not None:
            self.commit_left -= 1
            return self.commit, 1, "model"
        mv = self._commit_to(h_mv, "heuristic")
        return mv, 0, "heuristic"

    def __call__(self, scene, judg, yaw, use_jev, fresh=False, t=0.0, pos=None):
        if self.yaw_sp is None:
            self.yaw_sp = yaw
        sec = scene["sector_range_m"]
        tgt = scene["target"]
        acted = 0
        reflex = 0

        # Target tracking (world frame)
        if tgt["visible"]:
            self.sweep = 0.0
            self.sweep_dir = 1.0

        if tgt["visible"] and fresh and pos is not None:
            c, s = np.cos(yaw + tgt["bearing_rad"]), np.sin(yaw + tgt["bearing_rad"])
            pos_cam_2d = pos[:2] + 0.35 * np.array([np.cos(yaw), np.sin(yaw)])
            w = pos_cam_2d + tgt["range_m"] * np.array([c, s])
            if self.tgt_w is not None and t > self.tgt_t:
                dt_ = t - self.tgt_t
                v = (w - self.tgt_w) / dt_
                spd = np.linalg.norm(v)
                if spd > TARGET_MAX_SPEED:
                    v *= TARGET_MAX_SPEED / spd
                self.tgt_v = 0.8 * self.tgt_v + 0.2 * v
            self.tgt_w = w
            self.tgt_t = t

        # Evasion / Guidance
        turn_bias = 0.0
        fwd = SEARCH_SPEED
        absolute_yaw = None

        # The reactive layer is evaluated every tick so it is always ready underneath.
        h_mv = baseline_heuristic(scene)
        if use_jev and self.mode == "layered":
            mv, acted, layer = self._tactical_layered(scene, judg, h_mv)
        else:
            # legacy path; with --no-jev, judg already carries the heuristic maneuver
            mv, acted, layer = self._tactical_model_only(scene, judg)
            if not use_jev and layer == "model":
                layer = "heuristic"

        if mv == "reacquire" and tgt["visible"]:
            mv = "hold_course"

        self.last_mv, self.last_layer, self.last_heuristic_mv = mv, layer, h_mv

        base_yaw = (yaw + tgt["bearing_rad"]) if tgt["visible"] else self._search_heading(yaw, t, pos)
        base_rel = float((base_yaw - yaw + np.pi) % (2 * np.pi) - np.pi)

        if mv == "hold_course":
            rng = STANDOFF + 1.5
            if tgt["visible"]:
                absolute_yaw = yaw + tgt["bearing_rad"]
                if tgt["range_m"] < STANDOFF:
                    fwd = ROVER_SPEED * (tgt["range_m"] / STANDOFF)
                elif tgt["range_m"] < rng:
                    fwd = np.interp(tgt["range_m"], [STANDOFF, rng], [ROVER_SPEED, SEARCH_SPEED])
            else:
                absolute_yaw = self._search_heading(yaw, t, pos)
        elif mv == "steer_left":
            if sec[2] < 3.0 and scene["sector_clearance_m"][2] < 0.4:
                turn_bias = 0.4
            else:
                # Never let a right-side target bearing (base_rel < -0.35) pull steer_left to the right!
                absolute_yaw = yaw + max(base_rel + 0.35, 0.25)
        elif mv == "steer_right":
            if sec[2] < 3.0 and scene["sector_clearance_m"][2] < 0.4:
                turn_bias = -0.4
            else:
                # Never let a left-side target bearing (base_rel > +0.35) pull steer_right to the left!
                absolute_yaw = yaw + min(base_rel - 0.35, -0.25)
        elif mv == "brake":
            fwd = 0.0
        elif mv == "reverse":
            fwd = -0.8
        elif mv == "reacquire":
            if fresh:
                if self.sweep == 0.0:
                    delta = (self._search_heading(yaw, t, pos) - yaw + np.pi) % (2 * np.pi) - np.pi
                    self.sweep_dir = -1.0 if delta < 0 else 1.0
                self.sweep += 0.35
                self.yaw_sp = self._search_heading(yaw, t, pos) + self.sweep_dir * 0.45 * np.sin(self.sweep)
            absolute_yaw = self.yaw_sp
            fwd = SEARCH_SPEED
        else:
            if fresh:
                if self.sweep == 0.0:
                    delta = (self._search_heading(yaw, t, pos) - yaw + np.pi) % (2 * np.pi) - np.pi
                    self.sweep_dir = -1.0 if delta < 0 else 1.0
                self.sweep += 0.3
                self.yaw_sp = self._search_heading(yaw, t, pos) + self.sweep_dir * 0.35 * np.sin(self.sweep)
            absolute_yaw = self.yaw_sp

        # Bleed speed when the model judges risk high (matching drone run.py line 206),
        # except when actively overriding the reflex to smash cardboard.
        if (use_jev and judg and judg.get("source") == "jev" and self.override_left == 0
                and judg.get("risk", 0.0) > THRESHOLDS["risk_slow_down"] and mv not in ("brake", "reverse")):
            fwd *= 0.65

        # Reflex
        fatal_ranges = [sec[i] if scene["sector_clearance_m"][i] < 0.4 else 25.0 for i in range(5)]
        near_fatal = min(fatal_ranges)
        
        # JEV can explicitly override the safety reflex for things like cardboard
        if self.override_left > 0:
            near_fatal = 25.0
            fatal_ranges = [25.0] * 5

        if near_fatal < 1.3:
            reflex = 1
            fwd = min(fwd, 0.25)
            if fatal_ranges[2] < 1.3:
                absolute_yaw = None  # Reflex overrides guidance
                fwd = -0.8
                turn_bias = 0.5 if fatal_ranges[1] > fatal_ranges[3] else -0.5
            else:
                left_fatal = min(fatal_ranges[0], fatal_ranges[1])
                right_fatal = min(fatal_ranges[3], fatal_ranges[4])
                if left_fatal < 1.3 and left_fatal <= right_fatal:
                    absolute_yaw = None
                    turn_bias = -0.5
                elif right_fatal < 1.3 and right_fatal < left_fatal:
                    absolute_yaw = None
                    turn_bias = 0.5

        if fresh:
            self.yaw_sp = absolute_yaw if absolute_yaw is not None else yaw + turn_bias
        
        return fwd, self.yaw_sp, acted, reflex

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--no-jev", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    parser.add_argument("--video", default=None)
    parser.add_argument("--log", type=str, default=None, help="Export JSONL logs to this file")
    parser.add_argument("--model", type=str, default=os.environ.get("JEV_MODEL", "jev-latest"),
                        help="Model to use (e.g. jev-latest for native, or ~typesafe/jev-latest / provider/model for OpenRouter)")
    parser.add_argument("--provider", choices=["auto", "typesafe", "native", "openrouter"], default="auto",
                        help="Explicitly force API provider (auto detects from --model)")
    parser.add_argument("--fast", action="store_true", help="run faster than real time (unfair to Jev)")
    parser.add_argument("--guidance", choices=GUIDANCE_MODES, default=os.environ.get("ROVER_GUIDANCE", "layered"),
                        help="Tactical arbitration: 'layered' (default) keeps baseline_heuristic running underneath "
                             "the model as the continuous reactive/fallback layer, with fresh model judgments "
                             "overriding it; 'model-only' is the legacy behaviour (model alone decides, "
                             "stale/missing judgment -> brake). Ignored with --no-jev.")
    args = parser.parse_args()
    if args.provider == "openrouter":
        os.environ["USE_OPENROUTER"] = "1"
    elif args.provider in ("typesafe", "native"):
        os.environ["USE_OPENROUTER"] = "0"

    m = mujoco.MjModel.from_xml_path("rover_world.xml")
    d = mujoco.MjData(m)
    
    pilot = Pilot(m)
    eye = Eye(m)
    guide = Guidance(eye, mode=args.guidance)

    use_jev = not args.no_jev
    if use_jev:
        from rover_tactics import Tactician
        tactician = Tactician(model=args.model)
    
    mujoco.mj_step(m, d)
    
    viewer = None
    if not args.headless and os.environ.get("MUJOCO_GL") != "egl" and not args.video:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)
        viewer.cam.trackbodyid = m.body("chaser").id
        viewer.cam.distance = 7.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        
    writer = cam = big = hud = tape = None
    if args.video:
        import imageio.v2 as imageio
        from hud import Hud
        big = mujoco.Renderer(m, 880, 1180)
        hud = Hud(1180, 880)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation = 4.0, -15.0
        writer = imageio.get_writer(args.video, fps=30, quality=8, macro_block_size=1)
        tape = []
    
    import json
    t = 0.0
    dt = 0.002
    steps = int(args.seconds / dt)
    
    v_des, yaw_cmd = 0.0, 0.0
    scene, fresh = None, False
    judg = {"maneuver": "hold_course", "risk": 0.0, "confidence": 0.0,
            "target_truly_lost": 0.0, "source": "off", "age_s": 0.0, "probabilities": {}}
    hits = 0
    hit_steps = set()
    layer_ticks = {}
    chaser_geoms = [m.geom(name).id for name in ["chaser_collision", "fl_geom", "bl_geom", "fr_geom", "br_geom"] if name in [m.geom(i).name for i in range(m.ngeom)]]
    cardboard_geoms = [m.geom(i).id for i in range(m.ngeom) if "cardboard" in m.geom(i).name]
    start_pos = d.xpos[m.body("chaser").id].copy()
    max_x = float(start_pos[0])
    
    realtime = not args.fast
    wall0 = time.time()
    for i in range(steps):
        if realtime:
            lag = t - (time.time() - wall0)
            if lag > 0.0005:
                time.sleep(lag)
        drive_course(m, d, t)
        
        rot = d.xmat[m.body("chaser").id].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        pos = d.xpos[m.body("chaser").id]
        if pos[0] > max_x:
            max_x = float(pos[0])
        
        for j in range(d.ncon):
            g1, g2 = d.contact[j].geom1, d.contact[j].geom2
            if (g1 in chaser_geoms) != (g2 in chaser_geoms):
                other = g2 if g1 in chaser_geoms else g1
                # Ignore floor and cardboard collisions for hits count
                if m.geom(other).name != "floor" and other not in cardboard_geoms:
                    if other not in hit_steps:
                        hit_steps.add(other)
                        hits += 1
        
        if i % 33 == 0:
            scene = eye(d, t)
            fresh = True
        
        if i % 40 == 0 and scene:
            if use_jev:
                if decision_needed(scene):
                    tactician.offer(scene, t)
                j = tactician.read(t)
                if j:
                    judg = j
                    if not decision_needed(scene):
                        judg["age_s"] = 0.0
            else:
                judg = {
                    "maneuver": baseline_heuristic(scene),
                    "risk": 1.0,
                    "confidence": 1.0,
                    "target_truly_lost": 0.0,
                    "override_safety_reflex": 0.0,
                    "source": "heuristic",
                    "age_s": 0.0,
                    "probabilities": {}
                }
            v_des, yaw_cmd, acted, reflex = guide(scene, judg, yaw, use_jev, fresh, t, pos)
            layer_ticks[guide.last_layer] = layer_ticks.get(guide.last_layer, 0) + 1
            
            if args.log:
                import json
                log_entry = {
                    "time": t,
                    "pos": pos.tolist(),
                    "yaw": float(yaw),
                    "scene": scene,
                    "judg": judg,
                    "guide_action": {"v_des": float(v_des), "yaw_cmd": float(yaw_cmd), "acted": acted, "reflex": reflex,
                                     "commit": guide.commit, "commit_left": guide.commit_left,
                                     "guidance": args.guidance if use_jev else "heuristic-only",
                                     "layer": guide.last_layer, "maneuver": guide.last_mv,
                                     "heuristic_maneuver": guide.last_heuristic_mv,
                                     "reflex_override_left": guide.override_left}
                }
                with open(args.log, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                    
            fresh = False
            
        d.ctrl[:] = pilot(d, v_des, yaw_cmd, dt)
        mujoco.mj_step(m, d)
        t += dt
        
        if viewer and i % 16 == 0:
            viewer.sync()
            
        if writer and i % 17 == 0 and scene:
            cam.lookat[:] = pos
            cam.azimuth = np.rad2deg(yaw)
            m.vis.global_.fovy = 50.0
            big.update_scene(d, cam)
            
            # Use drive.Eye's camera for depth, it's already updated in eye(d)
            depth = np.clip(eye.depth.render(), 0, 25.0)
            standoff = np.linalg.norm(pos[:2] - d.xpos[m.body("rover").id][:2])
            
            tac = tactician if use_jev else None
            tel = {"t": t, "standoff": standoff, "speed": float(np.linalg.norm(d.qvel[:3])),
                   "hits": hits, "model": tac.model if tac else "disabled",
                   "mode": "JEV ENGAGED" if use_jev else "ABLATION: NO JEV",
                   "benchmark_type": "ROVER BENCHMARK",
                   "calls": tac.calls if tac else 0, "skipped": tac.skipped if tac else 0,
                   "tokens": tac.tokens if tac else 0, "hz": (1.0 / tac.min_dt) if tac else 0,
                   "lat": f"{np.median(tac.latency):.2f}s" if (tac and tac.latency) else "--",
                   "tan_h": eye.tan_h, "climbing": False,
                   "guidance": args.guidance if use_jev else "heuristic-only",
                   "layer": guide.last_layer, "maneuver": guide.last_mv}
            writer.append_data(hud.draw(big.render(), depth, scene, judg, tel))
            tape.append({"qpos": d.qpos.copy().tolist(), "mocap": d.mocap_pos.copy().tolist(),
                         "depth": depth.astype(np.float16), "scene": scene,
                         "judg": judg, "tel": {k: v for k, v in tel.items() if k != "tan_h"}})
            
    if writer:
        writer.close()
        np.save(args.video + ".tape.npy", np.array(tape, dtype=object), allow_pickle=True)
            
    if use_jev:
        tactician.close()
    if viewer:
        viewer.close()

    summary = {
        "model": args.model if hasattr(args, 'model') else MODEL if use_jev else "baseline",
        "jev_enabled": use_jev,
        "guidance": args.guidance if use_jev else "heuristic-only",
        "collisions": hits,
        "api_calls": tactician.calls if use_jev else 0,
        "distance_travelled": float(np.linalg.norm(pos - start_pos)),
        "forward_progress_m": round(max_x - float(start_pos[0]), 2),
        "max_x": round(max_x, 2),
        "final_t": t,
        "total_tokens": tactician.tokens if use_jev else 0,
        "layer_ticks": layer_ticks,
    }
    
    if use_jev:
        stats = tactician.stats()
        summary["median_latency_s"] = stats.get("median_latency_s")
        summary["p90_latency_s"] = stats.get("p90_latency_s")
        summary["api_errors"] = stats.get("errors")
        
    if use_jev: 
        summary["jev"] = tactician.stats()
    print(json.dumps(summary))
    return summary

if __name__ == "__main__":
    main()
