"""Differential-drive controller + onboard-camera perception for the rover."""
import numpy as np
import mujoco

IGNORE_BELOW_DEFAULT = 0.3
MJ_GEOM = mujoco.mjtObj.mjOBJ_GEOM

class Pilot:
    """Velocity and yaw controller for a differential drive rover."""

    def __init__(self, model):
        self.m = model
        self.track_width = 0.5  # distance between left and right wheels
        self.wheel_radius = 0.12
        # Proportional gain for yaw tracking
        self.k_yaw = 2.0
        self.max_v = 3.6
        self.max_w = 4.0
        self.max_accel = 5.0  # m/s^2
        self.v_forward_actual = 0.0
        self.actuator_idx = {
            'fl': model.actuator('drive_fl').id,
            'bl': model.actuator('drive_bl').id,
            'fr': model.actuator('drive_fr').id,
            'br': model.actuator('drive_br').id,
        }

    def __call__(self, data, v_des, yaw_cmd, dt=0.002):
        """
        v_des: forward velocity command (m/s)
        yaw_cmd: desired absolute yaw angle (rad)
        """
        # get current yaw from root joint quaternion
        q = data.qpos[3:7]  # freejoint quaternion (w, x, y, z)
        # yaw from quaternion
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # calculate yaw error (shortest path)
        yaw_err = (yaw_cmd - yaw + np.pi) % (2 * np.pi) - np.pi
        
        # calculate desired yaw rate (omega)
        omega_des = np.clip(self.k_yaw * yaw_err, -self.max_w, self.max_w)
        
        # calculate target forward velocity
        # if the yaw error is very large, slow down to turn in place
        turn_factor = max(0.0, 1.0 - abs(yaw_err) / (np.pi/4))
        if isinstance(v_des, np.ndarray):
            v_target = v_des[0]
        else:
            v_target = v_des
        v_target = np.clip(v_target * turn_factor, -self.max_v, self.max_v)
        
        # Slew rate limit to prevent wheelies
        dv = np.clip(v_target - self.v_forward_actual, -self.max_accel * dt, self.max_accel * dt)
        self.v_forward_actual += dv
        v_forward = self.v_forward_actual

        # differential drive kinematics: 
        # vL = v - (omega * w / 2)
        # vR = v + (omega * w / 2)
        vL = v_forward - (omega_des * self.track_width / 2.0)
        vR = v_forward + (omega_des * self.track_width / 2.0)
        
        # wheel angular velocities (rad/s) = v / r
        wL = vL / self.wheel_radius
        wR = vR / self.wheel_radius

        ctrl = np.zeros(self.m.nu)
        ctrl[self.actuator_idx['fl']] = wL
        ctrl[self.actuator_idx['bl']] = wL
        ctrl[self.actuator_idx['fr']] = wR
        ctrl[self.actuator_idx['br']] = wR
        return ctrl


class Eye:
    """Onboard forward camera -> symbolic scene summary for the rover."""

    W, H = 96, 72
    N_SECTORS = 5
    SECTOR_NAMES = ["far_left", "left", "center", "right", "far_right"]
    MAX_RANGE = 45.0
    TILT_DEG = -5.0
    IGNORE_BELOW_M = 0.3
    EYE_FOVY = 60.0

    def __init__(self, model, target_body="rover",
                 floor_geom="floor", shell=(), walls=()):
        self.shell_ids = np.array([model.geom(n).id for n in shell], dtype=int)
        self.wall_ids = [model.geom(n).id for n in walls]
        self.m = model
        self.depth = mujoco.Renderer(model, self.H, self.W)
        self.depth.enable_depth_rendering()
        self.seg = mujoco.Renderer(model, self.H, self.W)
        self.seg.enable_segmentation_rendering()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 1.0
        self.cam.elevation = self.TILT_DEG
        self.target_ids = np.nonzero(model.geom_bodyid == model.body(target_body).id)[0]
        self.floor_id = model.geom(floor_geom).id
        
        chaser_ids = [model.body(b).id for b in ["chaser", "fl_wheel", "bl_wheel", "fr_wheel", "br_wheel"]]
        self.self_geoms = np.nonzero(np.isin(model.geom_bodyid, chaser_ids))[0]
        
        fovy = np.deg2rad(self.EYE_FOVY)
        self.tan_h = np.tan(fovy / 2) * self.W / self.H
        
        self.last_seen_t = None

    def _pixel_to_bearing(self, x):
        """Relative yaw angle of a pixel column, 0 at center, right is negative."""
        nx = 2.0 * x / self.W - 1.0
        return -np.arctan(nx * self.tan_h)

    def sector_bearing(self, i):
        return self._pixel_to_bearing((i + 0.5) * (self.W / self.N_SECTORS))

    def __call__(self, data, t):
        chaser_id = self.m.body("chaser").id
        pos = data.xpos[chaser_id]
        rot = data.xmat[chaser_id].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        
        pos_cam = pos + rot @ np.array([0.35, 0, 0.15])
        az = yaw
        el = np.deg2rad(self.TILT_DEG)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        
        self.m.vis.global_.fovy = self.EYE_FOVY
        self.cam.azimuth = np.rad2deg(yaw)
        self.cam.lookat[:] = pos_cam + self.cam.distance * fwd

        self.depth.update_scene(data, self.cam)
        d = self.depth.render()
        self.seg.update_scene(data, self.cam)
        seg = self.seg.render()[:, :, 0]

        mask = (seg != -1) & (seg != self.floor_id)
        mask &= ~np.isin(seg, self.shell_ids)
        mask &= ~np.isin(seg, self.self_geoms)
        mask &= ~np.isin(seg, self.target_ids)

        # Calculate Z world of each pixel
        tan_v = np.tan(np.deg2rad(self.EYE_FOVY) / 2)
        ny = 1.0 - 2.0 * np.arange(self.H) / self.H
        NY = np.tile(ny[:, None], (1, self.W))
        pz_cam = d * NY * tan_v
        tilt = np.deg2rad(self.TILT_DEG)
        Z_cam_world = pos_cam[2]
        pz_world = Z_cam_world + d * np.sin(tilt) + pz_cam * np.cos(tilt)

        # Apply sec_scale correction so off-axis objects don't appear deceptively close
        bearings = self._pixel_to_bearing(np.arange(self.W))
        sec_scale = 1.0 / np.cos(bearings)
        d_radial = d * sec_scale[None, :]

        d_threats = np.where(mask, d_radial, self.MAX_RANGE)
        d_tgt = np.where(np.isin(seg, self.target_ids), d_radial, self.MAX_RANGE)
        min_tgt = float(d_tgt.min())

        sectors_d = np.array_split(d_threats, self.N_SECTORS, axis=1)
        sectors_z = np.array_split(pz_world, self.N_SECTORS, axis=1)
        sectors_m = np.array_split(mask, self.N_SECTORS, axis=1)
        sectors_seg = np.array_split(seg, self.N_SECTORS, axis=1)
        
        r = []
        c = []
        sec_geoms = []
        for i in range(self.N_SECTORS):
            sm = sectors_m[i]
            sd = sectors_d[i]
            sz = sectors_z[i]
            sseg = sectors_seg[i]
            if not np.any(sm):
                r.append(self.MAX_RANGE)
                c.append(25.0)
                sec_geoms.append(None)
            else:
                low_sm = sm & (sz < 0.4)
                # If a low-clearance (<0.4m) obstacle is within tactical range (4.0m),
                # measure range, clearance, and object type from the ground-level obstacle
                # so overhead structures (e.g. overpass roof) do not mask obstacles underneath.
                if np.any(low_sm) and sd[low_sm].min() < 4.0:
                    active_m = low_sm
                else:
                    active_m = sm
                d_min = float(sd[active_m].min())
                r.append(d_min)
                close_mask = active_m & (sd < d_min + 0.5)
                c.append(float(sz[close_mask].min()))

                very_close_mask = active_m & (sd < d_min + 0.1)
                sec_geoms.append(int(sseg[very_close_mask][0]) if np.any(very_close_mask) else None)

        closest_geom = None
        if r[2] < 4.0 and c[2] < 0.4 and sec_geoms[2] is not None:
            closest_geom = sec_geoms[2]
        else:
            candidates = [i for i in (1, 3) if r[i] < 4.0 and c[i] < 0.4 and sec_geoms[i] is not None]
            if not candidates:
                candidates = [i for i in range(self.N_SECTORS) if c[i] < 0.4 and sec_geoms[i] is not None]
            if not candidates:
                candidates = [i for i in range(self.N_SECTORS) if sec_geoms[i] is not None]
            if candidates:
                best_i = min(candidates, key=lambda i: r[i])
                closest_geom = sec_geoms[best_i]

        cx = self.W / 2
        bearing = None
        if min_tgt < self.MAX_RANGE - 0.1:
            ys, xs = np.where(d_tgt < self.MAX_RANGE)
            cx = xs.mean()
            bearing = self._pixel_to_bearing(cx)
            self.last_seen_t = t

        unseen = 0.0
        if bearing is None and self.last_seen_t is not None:
            unseen = t - self.last_seen_t

        return {
            "sector_range_m": r,
            "sector_clearance_m": c,
            "sectors_blocked": sum(1 for d in r if d < 2.5),
            "nearest_obstacle_m": min(r),
            "nearest_obstacle_type": "cardboard" if closest_geom is not None and "cardboard" in self.m.geom(closest_geom).name else "solid",
            "target": {
                "visible": bearing is not None,
                "bearing_rad": bearing if bearing is not None else 0.0,
                "range_m": min_tgt,
                "unseen_for_s": unseen,
            }
        }
