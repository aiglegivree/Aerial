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
        self.target_switch_count = 0
        self.target_switch_limit = 15
        self.force_rightmost_target = False
        self.last_target_center_px = None

        self.pass_through_timer = 0.0
        self.pass_through_heading = 0.0
        self.pass_through_altitude = self.cruise_altitude
        self.advance_timer = 0.0
        self.camera_half_fov = np.deg2rad(35.0)
        self.camera_fov = 1.5
        self.camera_offset_body = np.array([0.03, 0.0, 0.01], dtype=float)
        self.camera_to_body = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        self.gate_real_height = 0.4
        self.gate_focal_px = 320.0
        self.pending_gate_samples = []
        self.gate_bearing_observations = []
        self.min_triangulation_observations = 12
        self.max_triangulation_observations = 40
        self.min_triangulation_baseline = 0.6
        self.gate_center_backoff = 0.35
        self.side_sample_distance = 0.75
        self.side_sample_speed = 1
        self.side_sample_setpoint_distance = 0.25
        self.side_sample_duration = 30.0
        self.side_sample_target = None
        self.side_sample_right_target = None
        self.side_sample_left_target = None
        self.side_sample_altitude = self.cruise_altitude
        self.side_sample_timer = 0.0
        self.side_sample_used = False
        self.side_sample_phase = None
        self.side_sample_best_area = 0.0
        self.side_sample_best_position = None
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0
        self.side_sample_required_zero_frames = 5
        self.side_sample_zero_area_px = 50.0
        self.replay_approach_distance = 0.7
        self.first_gate_approach_distance = 1.1
        self.first_gate_exit_distance = 0.45

        self.learned_gates = []
        self.expected_gate_count = 5
        self.replay_lap = 0
        self.total_replay_laps = 2
        self.replay_trajectory = []
        self.replay_trajectory_types = []
        self.replay_trajectory_headings = []
        self.replay_traj_index = 0
        self.last_printed_replay_type = None
        self.replay_traj_tol = 0.55
        self.replay_gate_tol = 0.4
        self.replay_lookahead_distance = 0.65

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

    def rectangle_candidates(self, mask, img_shape):
        height, width = img_shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        candidates = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)

            center_x = float(box[:, 0].mean())
            center_y = float(box[:, 1].mean())
            dx = (center_x - width / 2.0) / (width / 2.0)
            dy = (center_y - height / 2.0) / (height / 2.0)
            area_px = cv2.contourArea(contour)
            area_rel = area_px / float(height * width)
            pixel_height = float(np.max(box[:, 1]) - np.min(box[:, 1]))

            candidates.append(
                {
                    "dx": float(dx),
                    "dy": float(dy),
                    "area_px": float(area_px),
                    "area_rel": float(area_rel),
                    "pixel_height": pixel_height,
                    "center_px": (center_x, center_y),
                    "image_shape": (height, width),
                }
            )

        return sorted(candidates, key=lambda item: item["area_rel"], reverse=True)

    def reset_target_switch_tracking(self):
        self.target_switch_count = 0
        self.force_rightmost_target = False
        self.last_target_center_px = None

    def update_target_switch_tracking(self, chosen):
        if len(self.learned_gates) >= self.expected_gate_count or chosen is None:
            return
        if self.force_rightmost_target:
            self.last_target_center_px = chosen["center_px"]
            return

        current_center = np.array(chosen["center_px"], dtype=float)
        if self.last_target_center_px is not None:
            previous_center = np.array(self.last_target_center_px, dtype=float)
            height, width = chosen["image_shape"]
            switch_threshold = 0.22 * np.linalg.norm([width, height])
            if np.linalg.norm(current_center - previous_center) > switch_threshold:
                self.target_switch_count += 1
                if self.target_switch_count > self.target_switch_limit:
                    self.force_rightmost_target = True
                    print("Too many target switches, focusing rightmost gate")

        self.last_target_center_px = chosen["center_px"]

    def choose_gate_candidate(self, candidates, track_switches=True):
        if not candidates:
            return None
        rightmost = max(candidates, key=lambda item: item["center_px"][0])
        if self.force_rightmost_target:
            if track_switches:
                self.update_target_switch_tracking(rightmost)
            return rightmost
        largest = max(candidates, key=lambda item: item["area_rel"])
        if largest["area_rel"] >= 1.3 * rightmost["area_rel"]:
            chosen = largest
        else:
            chosen = rightmost

        if track_switches:
            self.update_target_switch_tracking(chosen)
            if self.force_rightmost_target:
                self.last_target_center_px = rightmost["center_px"]
                return rightmost
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

    def body_to_world_rotation(self, sensor_data):
        roll = sensor_data["roll"]
        pitch = sensor_data["pitch"]
        yaw = sensor_data["yaw"]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        r_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=float,
        )
        r_y = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=float,
        )
        r_z = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return r_z @ r_y @ r_x

    def camera_ray_world(self, sensor_data, error):
        height, width = error["image_shape"]
        center_x, center_y = error["center_px"]
        focal_px = width / (2.0 * np.tan(self.camera_fov / 2.0))
        pixel_x = center_x - width / 2.0
        pixel_y = center_y - height / 2.0

        ray_camera = np.array([pixel_x, pixel_y, focal_px], dtype=float)
        ray_body = self.camera_to_body @ ray_camera
        rotation_body_to_world = self.body_to_world_rotation(sensor_data)
        ray_world = rotation_body_to_world @ ray_body
        ray_norm = np.linalg.norm(ray_world)
        if ray_norm < 1e-6:
            return None

        drone_position = np.array(
            [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
            dtype=float,
        )
        camera_origin = drone_position + rotation_body_to_world @ self.camera_offset_body
        return camera_origin, ray_world / ray_norm

    def collect_gate_sample(self, sensor_data, error):
        if len(self.learned_gates) >= self.expected_gate_count:
            return

        candidate = self.gate_world_estimate(sensor_data, error)
        self.pending_gate_samples.append(candidate)
        if len(self.pending_gate_samples) > 12:
            self.pending_gate_samples = self.pending_gate_samples[-12:]

        ray = self.camera_ray_world(sensor_data, error)
        if ray is None:
            return
        camera_origin, ray_direction = ray
        centered_score = max(0.05, 1.0 - abs(error["dx"]))
        area_score = max(error["area_rel"], 1e-4)
        self.gate_bearing_observations.append(
            {
                "origin": camera_origin,
                "direction": ray_direction,
                "weight": float(area_score * centered_score * centered_score),
            }
        )
        if len(self.gate_bearing_observations) > self.max_triangulation_observations:
            self.gate_bearing_observations = self.gate_bearing_observations[-self.max_triangulation_observations:]

    def reset_gate_localization(self):
        self.pending_gate_samples = []
        self.gate_bearing_observations = []
        self.side_sample_target = None
        self.side_sample_right_target = None
        self.side_sample_left_target = None
        self.side_sample_timer = 0.0
        self.side_sample_used = False
        self.side_sample_phase = None
        self.side_sample_best_area = 0.0
        self.side_sample_best_position = None
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0

    def triangulation_baseline(self):
        if len(self.gate_bearing_observations) < 2:
            return 0.0
        origins = np.array([obs["origin"] for obs in self.gate_bearing_observations], dtype=float)
        separations = origins[:, None, :] - origins[None, :, :]
        return float(np.max(np.linalg.norm(separations, axis=2)))

    def triangulate_gate_position(self):
        if len(self.gate_bearing_observations) < self.min_triangulation_observations:
            return None

        if self.triangulation_baseline() < self.min_triangulation_baseline:
            return None

        a_matrix = np.zeros((3, 3), dtype=float)
        b_vector = np.zeros(3, dtype=float)
        weight_sum = 0.0
        identity = np.eye(3, dtype=float)

        for obs in self.gate_bearing_observations:
            direction = obs["direction"]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue

            direction = direction / norm
            normal_matrix = identity - np.outer(direction, direction)
            weight = obs["weight"]
            a_matrix += weight * normal_matrix
            b_vector += weight * normal_matrix @ obs["origin"]
            weight_sum += weight

        if weight_sum <= 1e-6 or np.linalg.cond(a_matrix) > 40.0:
            return None

        gate_position = np.linalg.solve(a_matrix, b_vector)
        return [float(gate_position[0]), float(gate_position[1]), float(gate_position[2])]

    def shift_gate_toward_drone(self, sensor_data, gate_position):
        gate = np.array(gate_position, dtype=float)
        drone = np.array(
            [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
            dtype=float,
        )
        direction = gate[:2] - drone[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return gate_position

        gate[:2] -= self.gate_center_backoff * direction / norm
        return [float(gate[0]), float(gate[1]), float(gate[2])]

    def maybe_store_gate(self, sensor_data, error, allow_fallback=False):
        if len(self.learned_gates) >= self.expected_gate_count:
            self.reset_gate_localization()
            return False

        self.collect_gate_sample(sensor_data, error)
        gate_candidate = self.triangulate_gate_position()
        if gate_candidate is None:
            if not allow_fallback or len(self.pending_gate_samples) < 6:
                return False
            gate_candidate = np.median(np.array(self.pending_gate_samples), axis=0).tolist()
            print("Triangulation fallback for gate", len(self.learned_gates))

        gate_candidate = self.shift_gate_toward_drone(sensor_data, gate_candidate)

        if self.learned_gates:
            previous = np.array(self.learned_gates[-1]["gate"])
            if np.linalg.norm(np.array(gate_candidate) - previous) < 1.0:
                self.reset_gate_localization()
                return False

        self.learned_gates.append(
            {
                "gate": gate_candidate,
                "heading": float(sensor_data["yaw"]),
            }
        )
        print("Triangulated gate", len(self.learned_gates), "at", np.round(gate_candidate, 2).tolist())
        self.reset_gate_localization()
        return True

    def begin_side_sample(self, sensor_data, altitude, initial_error=None):
        left_direction = np.array([-np.sin(sensor_data["yaw"]), np.cos(sensor_data["yaw"])], dtype=float)
        right_direction = -left_direction
        current_xy = np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)
        right_target = current_xy + self.side_sample_distance * right_direction
        left_target = current_xy + self.side_sample_distance * left_direction

        self.side_sample_target = right_target
        self.side_sample_right_target = right_target
        self.side_sample_left_target = left_target
        self.side_sample_altitude = altitude
        self.side_sample_timer = self.side_sample_duration
        self.side_sample_used = True
        self.side_sample_phase = "RIGHT_SCAN"
        self.side_sample_best_area = initial_error["area_rel"] if initial_error is not None else 0.0
        self.side_sample_best_position = current_xy.copy()
        self.side_sample_last_center_px = initial_error["center_px"] if initial_error is not None else None
        self.side_sample_near_zero_frames = 0
        self.state = "SIDE_SAMPLE"
        print("Side sample sweeping right")
        return [
            sensor_data["x_global"],
            sensor_data["y_global"],
            altitude,
            sensor_data["yaw"],
        ]

    def choose_side_sample_candidate(self, candidates):
        if not candidates:
            return None
        if self.side_sample_last_center_px is None:
            return self.choose_gate_candidate(candidates, track_switches=False)

        previous = np.array(self.side_sample_last_center_px, dtype=float)
        closest = min(
            candidates,
            key=lambda item: np.linalg.norm(np.array(item["center_px"], dtype=float) - previous),
        )
        height, width = closest["image_shape"]
        max_jump = 0.30 * np.linalg.norm([width, height])
        jump = np.linalg.norm(np.array(closest["center_px"], dtype=float) - previous)
        if jump > max_jump and closest["area_rel"] < self.side_sample_best_area * 0.25:
            return None
        return closest

    def switch_side_sample_to_left(self):
        if self.side_sample_left_target is None or self.side_sample_phase != "RIGHT_SCAN":
            return

        self.side_sample_target = self.side_sample_left_target.copy()
        self.side_sample_phase = "LEFT_SCAN"
        self.side_sample_near_zero_frames = 0
        print("Side sample sweeping left")

    def switch_side_sample_to_best(self):
        if self.side_sample_best_position is None:
            return

        self.side_sample_target = self.side_sample_best_position.copy()
        self.side_sample_phase = "RETURN_BEST"
        self.side_sample_near_zero_frames = 0
        print("Side sample returning to max area")

    def side_sample_command(self, sensor_data, camera_data, dt):
        self.side_sample_timer -= dt
        if self.side_sample_target is None:
            self.state = "TRACK"
            return [
                sensor_data["x_global"],
                sensor_data["y_global"],
                sensor_data["z_global"],
                sensor_data["yaw"],
            ]

        mask = self.compute_mask(camera_data)
        candidates = self.rectangle_candidates(mask, camera_data.shape)
        error = self.choose_side_sample_candidate(candidates)
        usable_error = None
        yaw_target = sensor_data["yaw"]
        current_xy = np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)

        if error is not None:
            self.last_seen_yaw = sensor_data["yaw"]
            self.last_seen_time = self.time_since_start
            self.scan_center_yaw = sensor_data["yaw"]
            yaw_correction = -0.7 * error["dx"]
            yaw_target = sensor_data["yaw"] + np.clip(yaw_correction, -np.deg2rad(14), np.deg2rad(14))

            current_area = error["area_rel"]
            area_near_zero = error["area_px"] < self.side_sample_zero_area_px
            if area_near_zero:
                self.side_sample_near_zero_frames += 1
                if self.side_sample_near_zero_frames >= self.side_sample_required_zero_frames:
                    if self.side_sample_phase == "RIGHT_SCAN":
                        self.switch_side_sample_to_left()
                    elif self.side_sample_phase == "LEFT_SCAN":
                        self.switch_side_sample_to_best()
            else:
                self.side_sample_near_zero_frames = 0
                self.side_sample_last_center_px = error["center_px"]
                self.collect_gate_sample(sensor_data, error)
                usable_error = error

            if not area_near_zero and current_area > self.side_sample_best_area:
                self.side_sample_best_area = current_area
                self.side_sample_best_position = current_xy.copy()
        else:
            self.side_sample_near_zero_frames += 1
            if self.side_sample_near_zero_frames >= self.side_sample_required_zero_frames:
                if self.side_sample_phase == "RIGHT_SCAN":
                    self.switch_side_sample_to_left()
                elif self.side_sample_phase == "LEFT_SCAN":
                    self.switch_side_sample_to_best()

        reached_target = np.linalg.norm(self.side_sample_target - current_xy) < 0.05
        if reached_target and self.side_sample_phase == "RIGHT_SCAN":
            self.switch_side_sample_to_left()
            reached_target = False
        elif reached_target and self.side_sample_phase == "LEFT_SCAN":
            self.switch_side_sample_to_best()
            reached_target = False

        if self.side_sample_timer <= 0.0 and self.side_sample_phase != "RETURN_BEST":
            self.switch_side_sample_to_best()
            reached_target = False

        if (reached_target and self.side_sample_phase == "RETURN_BEST") or self.side_sample_timer <= 0.0:
            self.side_sample_target = None
            self.side_sample_timer = 0.0
            self.side_sample_phase = None
            self.state = "TRACK"
            if usable_error is not None:
                return self.align_and_approach_command(sensor_data, usable_error)
            return [
                sensor_data["x_global"],
                sensor_data["y_global"],
                self.side_sample_altitude,
                yaw_target,
            ]

        delta = self.side_sample_target - current_xy
        distance = np.linalg.norm(delta)
        if distance > 1e-6:
            max_step = self.side_sample_setpoint_distance
            command_xy = current_xy + min(distance, max_step) * delta / distance
        else:
            command_xy = current_xy

        return [
            float(command_xy[0]),
            float(command_xy[1]),
            self.side_sample_altitude,
            yaw_target,
        ]

    def build_replay_trajectory(self):
        if len(self.learned_gates) < self.expected_gate_count:
            self.replay_trajectory = []
            self.replay_trajectory_types = []
            self.replay_trajectory_headings = []
            self.replay_traj_index = 0
            return

        trajectory = []
        trajectory_types = []
        trajectory_headings = []
        for index, gate_bundle in enumerate(self.learned_gates):
            gate_point = np.array(gate_bundle["gate"], dtype=float)
            heading = gate_bundle.get("heading")
            if heading is None:
                previous_gate = np.array(self.learned_gates[(index - 1) % len(self.learned_gates)]["gate"], dtype=float)
                approach_direction = gate_point[:2] - previous_gate[:2]
                heading = np.arctan2(approach_direction[1], approach_direction[0])

            approach_direction = np.array([np.cos(heading), np.sin(heading)], dtype=float)
            approach_distance = self.first_gate_approach_distance if index == 0 else self.replay_approach_distance
            approach_point = gate_point.copy()
            approach_point[:2] -= approach_distance * approach_direction
            exit_point = gate_point.copy()
            exit_point[:2] += self.first_gate_exit_distance * approach_direction

            trajectory.append(approach_point.tolist())
            trajectory_types.append("approach")
            trajectory_headings.append(heading)
            trajectory.append(gate_point.tolist())
            trajectory_types.append("gate")
            trajectory_headings.append(heading)
            trajectory.append(exit_point.tolist())
            trajectory_types.append("exit")
            trajectory_headings.append(heading)

        self.replay_trajectory = trajectory
        self.replay_trajectory_types = trajectory_types
        self.replay_trajectory_headings = trajectory_headings
        self.replay_traj_index = 0
        self.last_printed_replay_type = None

    def replay_current_tolerance(self):
        if not self.replay_trajectory_types:
            return self.replay_traj_tol
        target_type = self.replay_trajectory_types[self.replay_traj_index]
        if target_type == "approach":
            return self.replay_traj_tol
        if target_type == "gate":
            return self.replay_gate_tol
        return self.replay_traj_tol

    def print_replay_target_reached(self):
        if not self.replay_trajectory_types:
            return
        target_type = self.replay_trajectory_types[self.replay_traj_index]
        if target_type == self.last_printed_replay_type:
            return
        print("Reached", target_type)
        self.last_printed_replay_type = target_type

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

        if error["area_rel"] > 0.012 and abs(error["dx"]) < 0.2 and abs(z_error) < 0.2:
            self.collect_gate_sample(sensor_data, error)

        enough_bearings = len(self.gate_bearing_observations) >= self.min_triangulation_observations
        if error["area_rel"] > 0.08 and very_centered and enough_bearings:
            baseline_ready = self.triangulation_baseline() >= self.min_triangulation_baseline
            if not baseline_ready and not self.side_sample_used:
                return self.begin_side_sample(sensor_data, z_target, error)
            if not self.maybe_store_gate(sensor_data, error, allow_fallback=True):
                return [
                    sensor_data["x_global"] + 0.04 * np.cos(sensor_data["yaw"]),
                    sensor_data["y_global"] + 0.04 * np.sin(sensor_data["yaw"]),
                    z_target,
                    yaw_target,
                ]
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
            self.reset_gate_localization()
            self.reset_target_switch_tracking()
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
                self.build_replay_trajectory()
                self.state = "REPLAY"

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

        delta = target - np.array(
            [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
            dtype=float,
        )
        distance = np.linalg.norm(delta)

        if distance < self.replay_current_tolerance():
            self.print_replay_target_reached()
            self.replay_traj_index += 1
            if self.replay_traj_index >= len(self.replay_trajectory):
                self.replay_traj_index = 0
                self.replay_lap += 1
                if self.replay_lap >= self.total_replay_laps:
                    self.state = "RETURN_HOME"
                    return self.return_home_command(sensor_data)

            target = np.array(self.replay_trajectory[self.replay_traj_index], dtype=float)
            delta = target - np.array(
                [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
                dtype=float,
            )
            distance = np.linalg.norm(delta)

        if len(self.replay_trajectory_headings) == len(self.replay_trajectory):
            yaw_target = self.replay_trajectory_headings[self.replay_traj_index]
        else:
            yaw_target = sensor_data["yaw"]

        command_target = target
        is_final_replay_target = (
            self.replay_lap == self.total_replay_laps - 1
            and self.replay_traj_index == len(self.replay_trajectory) - 1
        )
        if not is_final_replay_target and len(self.replay_trajectory) > 1 and distance < self.replay_lookahead_distance:
            next_index = (self.replay_traj_index + 1) % len(self.replay_trajectory)
            next_target = np.array(self.replay_trajectory[next_index], dtype=float)
            segment = next_target - target
            segment_norm = np.linalg.norm(segment)
            if segment_norm > 1e-6:
                lookahead_ratio = 1.0 - distance / self.replay_lookahead_distance
                command_target = target + np.clip(lookahead_ratio, 0.0, 1.0) * segment

        return [
            float(command_target[0]),
            float(command_target[1]),
            float(command_target[2]),
            yaw_target,
        ]

    def return_home_command(self, sensor_data):
        dx = self.home_setpoint[0] - sensor_data["x_global"]
        dy = self.home_setpoint[1] - sensor_data["y_global"]
        dz = self.home_setpoint[2] - sensor_data["z_global"]

        if np.linalg.norm([dx, dy, dz]) < 0.25:
            self.state = "WAIT"
            return self.wait_command()

        yaw_target = np.arctan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else self.home_setpoint[3]
        return [
            self.home_setpoint[0],
            self.home_setpoint[1],
            self.home_setpoint[2],
            yaw_target,
        ]

    def wait_command(self):
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

        if self.state == "SIDE_SAMPLE":
            return self.side_sample_command(sensor_data, camera_data, dt)

        if self.state == "REPLAY":
            return self.replay_command(sensor_data)

        if self.state == "RETURN_HOME":
            return self.return_home_command(sensor_data)

        if self.state == "WAIT":
            return self.wait_command()

        if self.state == "GO_TO_ORBIT":
            return self.go_to_orbit_command(sensor_data)

        if self.state == "SEARCH_ORBIT":
            mask = self.compute_mask(camera_data)
            candidates = self.rectangle_candidates(mask, camera_data.shape)
            error = self.choose_gate_candidate(candidates)
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
            error = self.choose_gate_candidate(candidates)
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
        error = self.choose_gate_candidate(candidates)

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
        return self.search_command(sensor_data, dt)


_controller = MyAssignment()


def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
