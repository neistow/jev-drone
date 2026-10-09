import os, sys, time, argparse
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass
import numpy as np
import mujoco
from drive import Pilot, Eye
from rover_tactics import decision_needed

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

class Guidance:
    def __init__(self, eye):
        self.eye = eye
        self.yaw_sp = None
        self.sweep = 0.0
        self.tgt_w = None
        self.tgt_v = np.zeros(2)
        self.tgt_t = 0.0
        self.commit = None
        self.commit_left = 0

    def _search_heading(self, yaw, t, pos):
        if self.tgt_w is None or pos is None:
            return yaw
        lead = float(np.clip(t - self.tgt_t, 0.0, 5.0))
        aim = self.tgt_w + self.tgt_v * lead
        d = aim - pos[:2]
        if np.linalg.norm(d) < 0.5:
            return yaw
        return float(np.arctan2(d[1], d[0]))

    def __call__(self, scene, judg, yaw, use_jev, fresh=False, t=0.0, pos=None):
        if self.yaw_sp is None:
            self.yaw_sp = yaw
        sec = scene["sector_range_m"]
        tgt = scene["target"]
        acted = 0
        reflex = 0

        # Target tracking (world frame)
        if tgt["visible"] and fresh and pos is not None:
            c, s = np.cos(yaw + tgt["bearing_rad"]), np.sin(yaw + tgt["bearing_rad"])
            w = pos[:2] + tgt["range_m"] * np.array([c, s])
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

        mv = None
        if use_jev and judg and judg.get("maneuver"):
            mv = judg["maneuver"]
            if self.commit is None or self.commit_left <= 0:
                self.commit = mv
                self.commit_left = 32
            else:
                mv = self.commit
                self.commit_left -= 1
                if judg.get("maneuver") != self.commit and judg.get("risk", 0) > 1.7:
                    self.commit = judg["maneuver"]
                    mv = self.commit
            acted = 1

        if mv == "hold_course":
            rng = STANDOFF + 1.5
            if tgt["visible"]:
                absolute_yaw = yaw + tgt["bearing_rad"]
                if tgt["range_m"] < rng:
                    fwd = ROVER_SPEED * (tgt["range_m"] / rng)
            else:
                absolute_yaw = self._search_heading(yaw, t, pos)
        elif mv == "steer_left":
            turn_bias = 0.4
        elif mv == "steer_right":
            turn_bias = -0.4
        elif mv == "brake":
            fwd = 0.0
        elif mv == "reverse":
            fwd = -0.8
        elif mv == "reacquire":
            if fresh:
                self.sweep += 0.35
                self.yaw_sp = self._search_heading(yaw, t, pos) + 0.45 * np.sin(self.sweep)
            absolute_yaw = self.yaw_sp
            fwd = SEARCH_SPEED
        else:
            if fresh:
                self.sweep += 0.3
                self.yaw_sp = self._search_heading(yaw, t, pos) + 0.35 * np.sin(self.sweep)
            absolute_yaw = self.yaw_sp

        # Reflex
        fatal_ranges = [sec[i] if scene["sector_clearance_m"][i] < 0.4 else 25.0 for i in range(5)]
        near_fatal = min(fatal_ranges)
        if near_fatal < 1.3:
            reflex = 1
            absolute_yaw = None  # Reflex overrides guidance
            fwd = min(fwd, 0.25)
            if fatal_ranges[2] < 1.3:
                fwd = -0.8
                turn_bias = 0.5 if fatal_ranges[1] > fatal_ranges[3] else -0.5

        if fresh:
            self.yaw_sp = absolute_yaw if absolute_yaw is not None else yaw + turn_bias
        
        return fwd, self.yaw_sp, acted, reflex

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=65.0)
    parser.add_argument("--no-jev", action="store_true")
    parser.add_argument("--video", default=None)
    parser.add_argument("--log", type=str, default=None, help="Export JSONL logs to this file")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path("rover_world.xml")
    d = mujoco.MjData(m)
    
    pilot = Pilot(m)
    eye = Eye(m)
    guide = Guidance(eye)

    use_jev = not args.no_jev
    if use_jev:
        from rover_tactics import Tactician
        tactician = Tactician()
    
    mujoco.mj_step(m, d)
    
    viewer = None
    if os.environ.get("MUJOCO_GL") != "egl" and not args.video:
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
    
    t = 0.0
    dt = 0.002
    steps = int(args.seconds / dt)
    
    v_des, yaw_cmd = 0.0, 0.0
    scene, fresh = None, False
    judg = {"maneuver": "hold_course", "risk": 0.0, "confidence": 0.0,
            "target_truly_lost": 0.0, "source": "off", "age_s": 0.0, "probabilities": {}}
    hits = 0
    
    for i in range(steps):
        drive_course(m, d, t)
        
        rot = d.xmat[m.body("chaser").id].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        pos = d.xpos[m.body("chaser").id]
        
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
            v_des, yaw_cmd, acted, reflex = guide(scene, judg, yaw, use_jev, fresh, t, pos)
            
            if args.log:
                import json
                log_entry = {
                    "time": t,
                    "pos": pos.tolist(),
                    "yaw": float(yaw),
                    "scene": scene,
                    "judg": judg,
                    "guide_action": {"v_des": float(v_des), "yaw_cmd": float(yaw_cmd), "acted": acted, "reflex": reflex, "commit": guide.commit, "commit_left": guide.commit_left}
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
                   "calls": tac.calls if tac else 0, "skipped": tac.skipped if tac else 0,
                   "tokens": tac.tokens if tac else 0, "hz": (1.0 / tac.min_dt) if tac else 0,
                   "lat": f"{np.median(tac.latency):.2f}s" if (tac and tac.latency) else "--",
                   "tan_h": eye.tan_h, "climbing": False}
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

if __name__ == "__main__":
    main()
