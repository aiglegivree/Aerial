import cv2
import numpy as np


class MyAssignment:
    def __init__(self):
        self.cruise_altitude = 1.35
        self.gate_vertical_bias = -1.0 / 12.0
        self.home_setpoint = [1.0, 4.0, self.cruise_altitude, 0.0]
        self.arena_center = np.array([4.0, 4.0], dtype=float)
        self.search_radius = 4.0
        self.scan_phase = 0.0
        self.scan_center_yaw = None
        self.scan_last_sine = 0.0
        self.scan_half_swings = 0
        self.max_scan_half_swings = 2
        self.rotate_left_accum = 0.0
        self.rotate_left_last_yaw = None
        self.orbit_angle = None

        self.state = "TAKEOFF"
        self.height_aligned = False

        self.last_seen_yaw = None
        self.last_seen_time = 0.0
        self.time_since_start = 0.0

        self.pass_through_timer = 0.0
        self.pass_through_heading = 0.0
        self.pass_through_altitude = self.cruise_altitude
        self.advance_timer = 0.0
        self.camera_half_fov = np.deg2rad(35.0)
        self.gate_real_height = 0.4
        self.gate_focal_px = 320.0
        self.pending_gate_samples = []
        self.min_gate_samples_to_store = 4
        self.post_gate_backoff = 0.35
        self.peak_gate_area = 0.0
        self.peak_gate_sample = None
        self.peak_gate_sensor = None
        self.peak_area_drop_to_store = 0.003

        self.learned_gates = []
        self.expected_gate_count = 5
        self.replay_gate_index = 0
        self.replay_lap = 0
        self.total_replay_laps = 2
        self.replay_trajectory = []
        self.replay_traj_index = 0
        self.replay_traj_tol = 0.55
        self.replay_speed_scale = 2
        self.recorded_positions = []
        self.recorded_index = 0
        self.recorded_cycle = 0
        self.total_recorded_cycles = 1
        self.recorded_position_tol = 0.35
        self.recorded_speed_scale = 1.4
        self.recorded_wait_seconds = 3.0
        self.recorded_wait_timer = 0.0

        self.locked_center = None
        self.locked_area = None
        self.locked_frames = 0
        self.switch_candidate_center = None
        self.switch_candidate_area = None
        self.switch_candidate_frames = 0

    def reset_search_pattern(self, current_yaw=None):
        self.scan_phase = 0.0
        self.scan_last_sine = 0.0
        self.scan_half_swings = 0
        self.rotate_left_accum = 0.0
        self.rotate_left_last_yaw = None
        if current_yaw is not None:
            self.scan_center_yaw = current_yaw

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_mask(self, camera_data):
        bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([120, 40, 150], dtype=np.uint8)
        upper = np.array([160, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def rectangle_candidates(self, mask, img_shape, min_area=50):
        height, width = img_shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid_contours:
            return []

        candidates = []
        for contour in valid_contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)

            center_x = float(box[:, 0].mean())
            center_y = float(box[:, 1].mean())
            dx = (center_x - width / 2.0) / (width / 2.0)
            dy = (center_y - height / 2.0) / (height / 2.0)
            area_rel = cv2.contourArea(contour) / float(height * width)
            pixel_height = float(np.max(box[:, 1]) - np.min(box[:, 1]))

            candidates.append(
                {
                    "dx": float(dx),
                    "dy": float(dy),
                    "area_rel": float(area_rel),
                    "pixel_height": pixel_height,
                    "center_px": (center_x, center_y),
                }
            )

        return sorted(candidates, key=lambda item: item["area_rel"], reverse=True)
    def choose_gate_candidate(self, candidates, sensor_data):
        if not candidates:
            self.locked_frames = 0
            self.switch_candidate_center = None
            self.switch_candidate_area = None
            self.switch_candidate_frames = 0
            return None
        del sensor_data
        rightmost = max(candidates, key=lambda item: item["center_px"][0])
        largest = max(candidates, key=lambda item: item["area_rel"])
        if largest["area_rel"] >= 1.2 * rightmost["area_rel"]:
            chosen = largest
        else:
            chosen = rightmost

        self.locked_center = chosen["center_px"]
        self.locked_area = chosen["area_rel"]
        self.locked_frames = 1
        self.switch_candidate_center = None
        self.switch_candidate_area = None
        self.switch_candidate_frames = 0
        return chosen

    def gate_world_estimate(self, sensor_data, error):
        pixel_height = max(error.get("pixel_height", 0.0), 1.0)
        gate_distance = self.gate_focal_px * self.gate_real_height / pixel_height
        gate_distance = float(np.clip(gate_distance, 0.5, 4.0))
        bearing = sensor_data["yaw"] - error["dx"] * self.camera_half_fov
        gate_x = sensor_data["x_global"] + gate_distance * np.cos(bearing)
        gate_y = sensor_data["y_global"] + gate_distance * np.sin(bearing)
        gate_z = sensor_data["z_global"]
        return [float(gate_x), float(gate_y), float(gate_z)]

    def pre_gate_world_estimate(self, sensor_data, gate_position):
        gate_xy = np.array(gate_position[:2], dtype=float)
        drone_xy = np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)
        approach_vector = gate_xy - drone_xy
        norm = np.linalg.norm(approach_vector)
        if norm < 1e-6:
            direction = np.array([np.cos(sensor_data["yaw"]), np.sin(sensor_data["yaw"])], dtype=float)
        else:
            direction = approach_vector / norm

        pre_gate_xy = gate_xy - 0.8 * direction
        return [float(pre_gate_xy[0]), float(pre_gate_xy[1]), float(gate_position[2])]

    def earlier_gate_position(self, sensor_data, gate_position, backoff):
        gate_xy = np.array(gate_position[:2], dtype=float)
        drone_xy = np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)
        approach_vector = gate_xy - drone_xy
        norm = np.linalg.norm(approach_vector)
        if norm < 1e-6:
            direction = np.array([np.cos(sensor_data["yaw"]), np.sin(sensor_data["yaw"])], dtype=float)
        else:
            direction = approach_vector / norm

        earlier_xy = gate_xy - backoff * direction
        return [float(earlier_xy[0]), float(earlier_xy[1]), float(gate_position[2])]

    def collect_gate_sample(self, sensor_data, error):
        if len(self.learned_gates) >= self.expected_gate_count:
            return

        candidate = self.gate_world_estimate(sensor_data, error)
        self.pending_gate_samples.append(candidate)
        if len(self.pending_gate_samples) > 8:
            self.pending_gate_samples = self.pending_gate_samples[-8:]

        if error["area_rel"] > self.peak_gate_area:
            self.peak_gate_area = error["area_rel"]
            self.peak_gate_sample = candidate
            self.peak_gate_sensor = {
                "x_global": sensor_data["x_global"],
                "y_global": sensor_data["y_global"],
                "yaw": sensor_data["yaw"],
            }

    def reset_gate_peak(self):
        self.peak_gate_area = 0.0
        self.peak_gate_sample = None
        self.peak_gate_sensor = None

    def store_gate_candidate(self, sensor_data, raw_post_gate_candidate):
        post_gate_candidate = self.earlier_gate_position(
            sensor_data,
            raw_post_gate_candidate,
            self.post_gate_backoff,
        )
        if self.learned_gates:
            previous = np.array(self.learned_gates[-1].get("post_gate", self.learned_gates[-1]["gate"]))
            min_gate_spacing = 1.0
            if np.linalg.norm(np.array(post_gate_candidate) - previous) < min_gate_spacing:
                return False

        pre_gate_candidate = self.pre_gate_world_estimate(sensor_data, post_gate_candidate)
        gate_candidate = (
            0.5 * (np.array(pre_gate_candidate, dtype=float) + np.array(post_gate_candidate, dtype=float))
        ).tolist()
        self.learned_gates.append(
            {
                "pre_gate": pre_gate_candidate,
                "gate": gate_candidate,
                "post_gate": post_gate_candidate,
            }
        )
        self.reset_gate_peak()
        return True

    def maybe_store_peak_gate(self, sensor_data, error):
        if len(self.pending_gate_samples) < self.min_gate_samples_to_store:
            return False
        if self.peak_gate_sample is None:
            return False
        if self.peak_gate_area < 0.02:
            return False
        if error["area_rel"] > self.peak_gate_area - self.peak_area_drop_to_store:
            return False

        peak_sensor = self.peak_gate_sensor if self.peak_gate_sensor is not None else sensor_data
        stored = self.store_gate_candidate(peak_sensor, self.peak_gate_sample)
        if stored:
            self.pending_gate_samples = []
        return stored

    def maybe_store_gate(self, sensor_data, error, collect_current=True, require_min_samples=False):
        if len(self.learned_gates) >= self.expected_gate_count:
            self.pending_gate_samples = []
            self.reset_gate_peak()
            return

        if collect_current:
            self.collect_gate_sample(sensor_data, error)
        if require_min_samples and len(self.pending_gate_samples) < self.min_gate_samples_to_store:
            return
        if not self.pending_gate_samples:
            return

        raw_post_gate_candidate = np.mean(np.array(self.pending_gate_samples), axis=0).tolist()
        self.pending_gate_samples = []
        self.store_gate_candidate(sensor_data, raw_post_gate_candidate)

    def catmull_rom_point(self, p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    def build_replay_trajectory(self):
        if len(self.learned_gates) < self.expected_gate_count:
            self.replay_trajectory = []
            self.replay_traj_index = 0
            return

        control_points = []
        for gate_bundle in self.learned_gates:
            control_points.append(np.array(gate_bundle["pre_gate"], dtype=float))
            control_points.append(np.array(gate_bundle["gate"], dtype=float))
            if "post_gate" in gate_bundle:
                control_points.append(np.array(gate_bundle["post_gate"], dtype=float))

        point_count = len(control_points)
        trajectory = []
        samples_per_segment = 20

        for index in range(point_count):
            p0 = control_points[(index - 1) % point_count]
            p1 = control_points[index % point_count]
            p2 = control_points[(index + 1) % point_count]
            p3 = control_points[(index + 2) % point_count]

            for sample_index in range(samples_per_segment):
                t = sample_index / float(samples_per_segment)
                point = self.catmull_rom_point(p0, p1, p2, p3, t)
                trajectory.append(point.tolist())

        self.replay_trajectory = trajectory
        self.replay_traj_index = 0

    def build_recorded_positions(self):
        if len(self.learned_gates) < self.expected_gate_count:
            self.recorded_positions = []
            self.recorded_index = 0
            return

        self.recorded_positions = []
        for gate_bundle in self.learned_gates:
            self.recorded_positions.append(list(gate_bundle["pre_gate"]))
            self.recorded_positions.append(list(gate_bundle["gate"]))
            self.recorded_positions.append(list(gate_bundle["post_gate"]))
        self.recorded_index = 0
        self.recorded_cycle = 0
        self.recorded_wait_timer = 0.0

    def search_command(self, sensor_data, dt):
        if self.scan_center_yaw is None:
            self.scan_center_yaw = sensor_data["yaw"]

        self.scan_phase += dt * 0.8
        current_sine = np.sin(self.scan_phase)
        if self.scan_last_sine != 0.0 and current_sine * self.scan_last_sine < 0.0:
            self.scan_half_swings += 1
        self.scan_last_sine = current_sine

        if self.scan_half_swings >= self.max_scan_half_swings:
            self.state = "ROTATE_LEFT"
            return self.rotate_left_command(sensor_data, dt)

        yaw_offset = np.deg2rad(70.0) * current_sine
        yaw_target = self.scan_center_yaw + yaw_offset

        z_error = self.cruise_altitude - sensor_data["z_global"]
        z_target = sensor_data["z_global"] + np.clip(1.4 * z_error, -0.12, 0.12)

        return [
            sensor_data["x_global"] + 0.06 * np.cos(sensor_data["yaw"]),
            sensor_data["y_global"] + 0.06 * np.sin(sensor_data["yaw"]),
            z_target,
            yaw_target,
        ]

    def rotate_left_command(self, sensor_data, dt):
        current_yaw = sensor_data["yaw"]
        if self.rotate_left_last_yaw is None:
            self.rotate_left_last_yaw = current_yaw
        else:
            yaw_step = self.wrap_angle(current_yaw - self.rotate_left_last_yaw)
            self.rotate_left_accum += abs(yaw_step)
            self.rotate_left_last_yaw = current_yaw

        if self.rotate_left_accum >= 2.0 * np.pi:
            self.state = "GO_TO_ORBIT"
            return self.go_to_orbit_command(sensor_data)

        yaw_target = sensor_data["yaw"] + np.deg2rad(45.0) * max(dt, 0.05)
        z_error = self.cruise_altitude - sensor_data["z_global"]
        z_target = sensor_data["z_global"] + np.clip(1.2 * z_error, -0.08, 0.08)
        return [
            sensor_data["x_global"],
            sensor_data["y_global"],
            z_target,
            yaw_target,
        ]

    def go_to_orbit_command(self, sensor_data):
        relative = np.array(
            [
                sensor_data["x_global"] - self.arena_center[0],
                sensor_data["y_global"] - self.arena_center[1],
            ],
            dtype=float,
        )
        norm = np.linalg.norm(relative)
        if norm < 1e-6:
            relative = np.array([0.0, -1.0], dtype=float)
            norm = 1.0

        direction = relative / norm
        target_xy = self.arena_center + self.search_radius * direction
        yaw_target = np.arctan2(self.arena_center[1] - target_xy[1], self.arena_center[0] - target_xy[0])

        if np.linalg.norm(target_xy - np.array([sensor_data["x_global"], sensor_data["y_global"]])) < 0.3:
            self.state = "SEARCH_ORBIT"
            self.orbit_angle = np.arctan2(direction[1], direction[0])
            self.reset_search_pattern(yaw_target)
            return self.search_orbit_command(sensor_data, 0.0)

        return [float(target_xy[0]), float(target_xy[1]), self.cruise_altitude, yaw_target]

    def search_orbit_command(self, sensor_data, dt):
        if self.orbit_angle is None:
            relative = np.array(
                [
                    sensor_data["x_global"] - self.arena_center[0],
                    sensor_data["y_global"] - self.arena_center[1],
                ],
                dtype=float,
            )
            self.orbit_angle = np.arctan2(relative[1], relative[0])

        self.orbit_angle += 0.22 * dt
        target_xy = self.arena_center + self.search_radius * np.array(
            [np.cos(self.orbit_angle), np.sin(self.orbit_angle)],
            dtype=float,
        )
        yaw_target = np.arctan2(self.arena_center[1] - target_xy[1], self.arena_center[0] - target_xy[0])

        return [float(target_xy[0]), float(target_xy[1]), self.cruise_altitude, yaw_target]

    def align_and_approach_command(self, sensor_data, error):
        yaw_correction = -0.6 * error["dx"]
        yaw_target = sensor_data["yaw"] + np.clip(yaw_correction, -np.deg2rad(12), np.deg2rad(12))

        z_error = error["dy"] - self.gate_vertical_bias
        if z_error > 0.0:
            z_step = np.clip(-0.75 * z_error, -0.16, 0.0)
        else:
            z_step = np.clip(-0.3 * z_error, 0.0, 0.07)
        z_target = sensor_data["z_global"] + z_step

        horizontally_centered = abs(error["dx"]) < 0.12
        vertically_ready = abs(z_error) < 0.12

        if not self.height_aligned and vertically_ready:
            self.height_aligned = True
        if self.height_aligned and abs(z_error) > 0.28:
            self.height_aligned = False

        centered = horizontally_centered and vertically_ready
        very_centered = abs(error["dx"]) < 0.06 and abs(z_error) < 0.08

        sample_ready = error["area_rel"] > 0.015 and abs(error["dx"]) < 0.2 and abs(z_error) < 0.22
        if sample_ready:
            self.collect_gate_sample(sensor_data, error)

        if sample_ready:
            self.maybe_store_peak_gate(sensor_data, error)

        if error["area_rel"] > 0.08 and very_centered:
            self.maybe_store_gate(sensor_data, error, collect_current=not sample_ready)
            self.state = "PASS_THROUGH"
            self.pass_through_timer = 1.0
            self.pass_through_heading = sensor_data["yaw"]
            self.pass_through_altitude = z_target
            return self.pass_through_command(sensor_data, 0.0)

        forward_step = 0.0
        if not self.height_aligned:
            if horizontally_centered and error["area_rel"] > 0.015 and abs(z_error) < 0.22:
                forward_step = 0.08
        elif centered:
            if error["area_rel"] < 0.025:
                forward_step = 0.45
            elif error["area_rel"] < 0.07:
                forward_step = 0.25
            else:
                forward_step = 0.12
        elif horizontally_centered and abs(z_error) < 0.16:
            forward_step = 0.06
        elif abs(error["dx"]) < 0.2 and abs(z_error) < 0.12:
            forward_step = 0.08

        x_target = sensor_data["x_global"] + forward_step * np.cos(sensor_data["yaw"])
        y_target = sensor_data["y_global"] + forward_step * np.sin(sensor_data["yaw"])
        return [x_target, y_target, z_target, yaw_target]

    def pass_through_command(self, sensor_data, dt):
        self.pass_through_timer -= dt

        distance = 0.65
        x_target = sensor_data["x_global"] + distance * np.cos(self.pass_through_heading)
        y_target = sensor_data["y_global"] + distance * np.sin(self.pass_through_heading)

        if self.pass_through_timer <= 0.0:
            self.state = "ADVANCE"
            self.advance_timer = 0.8
            self.scan_center_yaw = self.pass_through_heading
            self.locked_center = None
            self.locked_area = None
            self.locked_frames = 0
            self.pending_gate_samples = []
            self.reset_gate_peak()
            self.reset_search_pattern(self.pass_through_heading)

        return [
            x_target,
            y_target,
            self.pass_through_altitude,
            self.pass_through_heading,
        ]

    def advance_command(self, sensor_data, dt):
        self.advance_timer -= dt

        distance = 0.3
        x_target = sensor_data["x_global"] + distance * np.cos(self.pass_through_heading)
        y_target = sensor_data["y_global"] + distance * np.sin(self.pass_through_heading)

        if self.advance_timer <= 0.0:
            self.state = "SEARCH"
            self.height_aligned = False
            self.reset_search_pattern(sensor_data["yaw"])
            if len(self.learned_gates) >= self.expected_gate_count:
                self.build_recorded_positions()
                self.state = "RECORDED_VISIT"

        return [
            x_target,
            y_target,
            self.pass_through_altitude,
            self.pass_through_heading,
        ]

    def replay_command(self, sensor_data):
        if not self.replay_trajectory:
            self.state = "SEARCH"
            return self.search_command(sensor_data, 0.0)
        target = np.array(self.replay_trajectory[self.replay_traj_index], dtype=float)
        next_index = (self.replay_traj_index + 1) % len(self.replay_trajectory)
        next_target = np.array(self.replay_trajectory[next_index], dtype=float)

        delta = target - np.array(
            [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
            dtype=float,
        )
        distance = np.linalg.norm(delta)

        if distance < self.replay_traj_tol:
            self.replay_traj_index += 1
            if self.replay_traj_index >= len(self.replay_trajectory):
                self.replay_traj_index = 0
                self.replay_lap += 1
                if self.replay_lap >= self.total_replay_laps:
                    self.state = "RETURN_HOME"
                    return self.return_home_command(sensor_data)

            target = np.array(self.replay_trajectory[self.replay_traj_index], dtype=float)
            next_index = (self.replay_traj_index + 1) % len(self.replay_trajectory)
            next_target = np.array(self.replay_trajectory[next_index], dtype=float)
            delta = target - np.array(
                [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
                dtype=float,
            )

        yaw_target = np.arctan2(next_target[1] - target[1], next_target[0] - target[0])
        return [
            sensor_data["x_global"] + self.replay_speed_scale * delta[0],
            sensor_data["y_global"] + self.replay_speed_scale * delta[1],
            sensor_data["z_global"] + self.replay_speed_scale * delta[2],
            yaw_target,
        ]

    def recorded_visit_command(self, sensor_data, dt):
        if not self.recorded_positions:
            self.state = "SEARCH"
            return self.search_command(sensor_data, 0.0)

        target = np.array(self.recorded_positions[self.recorded_index], dtype=float)
        current = np.array(
            [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
            dtype=float,
        )
        delta = target - current
        distance = np.linalg.norm(delta)

        if distance < self.recorded_position_tol:
            if self.recorded_wait_timer <= 0.0:
                self.recorded_wait_timer = self.recorded_wait_seconds
                print("Reached recorded position", self.recorded_index)

            self.recorded_wait_timer -= dt
            if self.recorded_wait_timer <= 0.0:
                self.recorded_index += 1
                self.recorded_wait_timer = 0.0
                if self.recorded_index >= len(self.recorded_positions):
                    self.recorded_index = 0
                    self.recorded_cycle += 1
                    if self.recorded_cycle >= self.total_recorded_cycles:
                        self.state = "RETURN_HOME"
                        return self.return_home_command(sensor_data)

                target = np.array(self.recorded_positions[self.recorded_index], dtype=float)
                delta = target - current
            else:
                return [
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    sensor_data["yaw"],
                ]

        if abs(delta[0]) + abs(delta[1]) > 1e-6:
            yaw_target = np.arctan2(delta[1], delta[0])
        else:
            yaw_target = sensor_data["yaw"]

        return [
            sensor_data["x_global"] + self.recorded_speed_scale * delta[0],
            sensor_data["y_global"] + self.recorded_speed_scale * delta[1],
            sensor_data["z_global"] + self.recorded_speed_scale * delta[2],
            yaw_target,
        ]

    def return_home_command(self, sensor_data):
        dx = self.home_setpoint[0] - sensor_data["x_global"]
        dy = self.home_setpoint[1] - sensor_data["y_global"]
        dz = self.home_setpoint[2] - sensor_data["z_global"]

        if np.linalg.norm([dx, dy, dz]) < 0.25:
            self.state = "WAIT"
            return self.wait_command(sensor_data)

        yaw_target = np.arctan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else self.home_setpoint[3]
        return [
            self.home_setpoint[0],
            self.home_setpoint[1],
            self.home_setpoint[2],
            yaw_target,
        ]

    def wait_command(self, sensor_data):
        return [
            self.home_setpoint[0],
            self.home_setpoint[1],
            self.home_setpoint[2],
            self.home_setpoint[3],
        ]

    def compute_command(self, sensor_data, camera_data, dt):
        self.time_since_start += dt

        if self.scan_center_yaw is None:
            self.scan_center_yaw = sensor_data["yaw"]

        if self.state == "TAKEOFF":
            if sensor_data["z_global"] < self.cruise_altitude - 0.05:
                return [
                    sensor_data["x_global"],
                    sensor_data["y_global"],
                    self.cruise_altitude,
                    sensor_data["yaw"],
                ]
            self.state = "SEARCH"

        if self.state == "PASS_THROUGH":
            return self.pass_through_command(sensor_data, dt)

        if self.state == "ADVANCE":
            return self.advance_command(sensor_data, dt)

        if self.state == "REPLAY":
            return self.replay_command(sensor_data)

        if self.state == "RECORDED_VISIT":
            return self.recorded_visit_command(sensor_data, dt)

        if self.state == "RETURN_HOME":
            return self.return_home_command(sensor_data)

        if self.state == "WAIT":
            return self.wait_command(sensor_data)

        if self.state == "GO_TO_ORBIT":
            return self.go_to_orbit_command(sensor_data)

        if self.state == "SEARCH_ORBIT":
            mask = self.compute_mask(camera_data)
            candidates = self.rectangle_candidates(mask, camera_data.shape)
            error = self.choose_gate_candidate(candidates, sensor_data)
            if error is not None:
                self.last_seen_yaw = sensor_data["yaw"]
                self.last_seen_time = self.time_since_start
                self.scan_center_yaw = sensor_data["yaw"]
                self.state = "TRACK"
                self.reset_search_pattern(sensor_data["yaw"])
                return self.align_and_approach_command(sensor_data, error)
            return self.search_orbit_command(sensor_data, dt)

        if self.state == "ROTATE_LEFT":
            mask = self.compute_mask(camera_data)
            candidates = self.rectangle_candidates(mask, camera_data.shape)
            error = self.choose_gate_candidate(candidates, sensor_data)
            if error is not None:
                self.last_seen_yaw = sensor_data["yaw"]
                self.last_seen_time = self.time_since_start
                self.scan_center_yaw = sensor_data["yaw"]
                self.state = "TRACK"
                self.reset_search_pattern(sensor_data["yaw"])
                return self.align_and_approach_command(sensor_data, error)
            return self.rotate_left_command(sensor_data, dt)

        mask = self.compute_mask(camera_data)
        candidates = self.rectangle_candidates(mask, camera_data.shape)
        error = self.choose_gate_candidate(candidates, sensor_data)

        if error is not None:
            self.last_seen_yaw = sensor_data["yaw"]
            self.last_seen_time = self.time_since_start
            self.scan_center_yaw = sensor_data["yaw"]
            self.state = "TRACK"
            self.reset_search_pattern(sensor_data["yaw"])
            return self.align_and_approach_command(sensor_data, error)

        recently_seen = (self.time_since_start - self.last_seen_time) < 1.0 and self.last_seen_yaw is not None
        if recently_seen:
            forward_step = 0.12 if self.height_aligned else 0.0
            return [
                sensor_data["x_global"] + forward_step * np.cos(sensor_data["yaw"]),
                sensor_data["y_global"] + forward_step * np.sin(sensor_data["yaw"]),
                sensor_data["z_global"],
                self.last_seen_yaw,
            ]

        self.state = "SEARCH"
        self.height_aligned = False
        self.locked_center = None
        self.locked_area = None
        self.locked_frames = 0
        return self.search_command(sensor_data, dt)


_controller = MyAssignment()


def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
