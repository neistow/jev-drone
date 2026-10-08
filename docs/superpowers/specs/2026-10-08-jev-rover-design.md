# Jev Rover System Design

## 1. Overview
`jev-rover` is a ground-based autonomous follow-and-avoid simulation built on MuJoCo, serving as a companion experiment to `jev-drone`. A differential-drive chaser rover must track a moving target through a 3D obstacle course. The primary navigator is an LLM (Jev) operating as a System 1 tactical layer, outputting discrete maneuvering decisions from semantic scene summaries.

## 2. World & Physics (MuJoCo)
- **Environment**: Adapted from the existing pillar field. All horizontal beams are removed to accommodate ground-only navigation. Vertical pillars and walls remain.
- **Target**: A dummy mocap body following a predefined path, identical to the drone's target.
- **Chaser (Rover)**: A new differential drive (skid-steer) physics model. 
  - **Kinematics**: 4 wheels. The left two wheels are linked to a single velocity actuator, and the right two wheels are linked to another. This allows the car to steer by spinning in place (yaw rate) and driving forward, mapping smoothly to the drone's `yaw_sp` and horizontal velocity commands.
  - **Aesthetics**: The chassis will be rendered using a simple low-poly car mesh (OBJ) with applied materials to ensure visual quality in the test recordings.
  - **Sensors**: A forward-facing camera mounted on the chassis.

## 3. Architecture & Data Flow
The system operates across three asynchronous layers:

### 3.1 Low-level Controller (500 Hz)
- Lives in `drive.py` (analogous to `flight.py`).
- Inputs: Desired forward velocity ($v$) and desired yaw rate ($\omega$).
- Outputs: Left and right wheel velocities ($v_L$, $v_R$).
- $v_L = v - \omega \times \text{track\_width} / 2$
- $v_R = v + \omega \times \text{track\_width} / 2$

### 3.2 Guidance & Safety Reflex (50 Hz)
- Lives in `run_rover.py`.
- **Perception**: Processes the camera depth buffer to identify free space. Divides the view into horizontal sectors (left, ahead, right) and measures the nearest obstacle in each.
- **Guidance**: Maps the active tactical maneuver (e.g., `steer_left`) into specific continuous ($v$, $\omega$) setpoints.
- **Safety Reflex**: If an obstacle breaches the hard safety radius, this code overrides Jev's command and forces a hard brake or evasive turn.

### 3.3 Tactical Layer (Jev, ~3 Hz)
- Lives in `rover_tactics.py`.
- Runs asynchronously. The vehicle does not block waiting for decisions.
- **State Summary**: The scene is summarized into text (target bearing/distance, obstacle distances in left/center/right sectors).
- **Decisions**: Outputs one discrete maneuver.

## 4. Jev Tactics (System 1)
The tactical prompt removes 3D options (like `climb`) and focuses purely on 2D navigation.

**Available Maneuvers**:
- `hold_course`: Drive straight at the target. Path is clear.
- `steer_left`: Obstacle ahead, left gap is wider.
- `steer_right`: Obstacle ahead, right gap is wider.
- `brake`: Too tight, slow down and assess.
- `reacquire`: Target lost, spin to find it again.

## 5. Implementation Phases
1. **Assets & World**: Create `rover.xml` with mesh and diff-drive setup. Update `rover_world.xml` to remove beams.
2. **Controller**: Implement `drive.py` kinematics and camera depth extraction.
3. **Guidance**: Adapt `run.py` to `run_rover.py` with 2D steering logic and reflex.
4. **Tactics**: Adapt `tactics.py` to `rover_tactics.py` with the 2D maneuver set.
