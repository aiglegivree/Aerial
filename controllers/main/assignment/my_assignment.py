import cv2
import numpy as np


class MyAssignment:
    def __init__(self):
        self.cruise_altitude = 1.35
        self.gate_vertical_bias = -1.0 / 12.0
        self.home_setpoint = [1.0, 4.0, self.cruise_altitude, 0.0]
        self.arena_center = np.array([4.0, 4.0], dtype=float)
        self.search_radius = 4.0
        self.default_search_radius = 4.0
        self.first_lap_cone_radius = 3.0
        self.first_lap_cone_angle_offset = np.deg2rad(2.0)
        self.first_lap_cone_tol = 0.35
        self.first_lap_cone_step = 0.35
        self.first_lap_cone_target = None
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
        self.pass_through_duration = 1.35
        self.pass_through_distance = 0.8
        self.advance_duration = 1.1
        self.advance_distance = 0.45
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
        self.min_triangulation_observations = 6
        self.max_triangulation_observations = 40
        self.min_triangulation_baseline = 0.8
        self.last_triangulation_failure_reason = None
        self.triangulation_trigger_area = 0.05
        self.max_sample_area_after_baseline = 0.1
        self.min_sample_origin_spacing = 0.03
        self.gate_center_backoff = 0.0
        self.side_sample_distance = 0.7
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
        self.side_sample_phase_best_area = 0.0
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0
        self.side_sample_required_zero_frames = 5
        self.side_sample_zero_area_px = 50.0
        self.side_switch_area_rel = 0.01
        self.default_side_switch_area_rel = self.side_switch_area_rel
        self.require_side_sample_before_store = True
        self.reswipe_backoff_distance = 0.45
        self.reswipe_setpoint_distance = 0.25
        self.reswipe_attempts = 0
        self.max_reswipe_attempts = 2
        self.reswipe_target = None
        self.reswipe_altitude = self.cruise_altitude
        self.replay_approach_distance = 1.1
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
        self.replay_traj_tol = 0.45
        self.replay_gate_tol = 0.2
        self.replay_lookahead_distance = 0.65
        self.replay_switch_distance = 1.2
        self.replay_segment_lookahead = 1.2
        self.replay_xy_step = 0.45
        self.replay_z_step = 0.12
        self.search_sweep_translation = 0.00001

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

    def assignment_segment_center_angle(self, segment_index):
        segment_count = self.expected_gate_count + 1
        return 2.0 * segment_index * np.pi / segment_count

    def assignment_segment_radius_target(self, segment_index, altitude=None):
        angle = self.assignment_segment_center_angle(segment_index) + self.first_lap_cone_angle_offset
        target_xy = self.arena_center - self.first_lap_cone_radius * np.array(
            [np.cos(angle), np.sin(angle)],
            dtype=float,
        )
        yaw_target = np.arctan2(
            self.arena_center[1] - target_xy[1],
            self.arena_center[0] - target_xy[0],
        )
        if altitude is None:
            altitude = self.cruise_altitude
        return np.array([target_xy[0], target_xy[1], altitude, yaw_target], dtype=float)

    def next_first_lap_cone_segment(self):
        return min(len(self.learned_gates), self.expected_gate_count)

    def current_angle_radius_target(self, sensor_data, altitude=None):
        relative = np.array(
            [
                sensor_data["x_global"] - self.arena_center[0],
                sensor_data["y_global"] - self.arena_center[1],
            ],
            dtype=float,
        )
        if np.linalg.norm(relative) < 1e-6:
            relative = np.array([0.0, -1.0], dtype=float)

        angle = np.arctan2(relative[1], relative[0]) + self.first_lap_cone_angle_offset
        target_xy = self.arena_center + self.first_lap_cone_radius * np.array(
            [np.cos(angle), np.sin(angle)],
            dtype=float,
        )
        yaw_target = np.arctan2(
            self.arena_center[1] - target_xy[1],
            self.arena_center[0] - target_xy[0],
        )
        if altitude is None:
            altitude = self.cruise_altitude
        return np.array([target_xy[0], target_xy[1], altitude, yaw_target], dtype=float)

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
        baseline_ready = self.triangulation_baseline() >= self.min_triangulation_baseline
        if baseline_ready and error["area_rel"] > self.max_sample_area_after_baseline:
            return

        if self.gate_bearing_observations:
            nearest_origin_distance = min(
                np.linalg.norm(camera_origin - obs["origin"])
                for obs in self.gate_bearing_observations
            )
            if nearest_origin_distance < self.min_sample_origin_spacing:
                return

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
            self.gate_bearing_observations = self.select_best_bearing_observations()

    def select_best_bearing_observations(self):
        observations = self.gate_bearing_observations
        max_count = self.max_triangulation_observations
        if len(observations) <= max_count:
            return observations

        remaining = list(observations)
        first_index = max(range(len(remaining)), key=lambda idx: remaining[idx]["weight"])
        selected = [remaining.pop(first_index)]

        while remaining and len(selected) < max_count:
            best_index = 0
            best_score = -np.inf
            for index, obs in enumerate(remaining):
                min_distance = min(
                    np.linalg.norm(obs["origin"] - kept["origin"])
                    for kept in selected
                )
                score = obs["weight"] * (0.35 + min_distance)
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(remaining.pop(best_index))

        return selected

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
        self.side_sample_phase_best_area = 0.0
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0
        self.reswipe_attempts = 0
        self.reswipe_target = None
        self.reswipe_altitude = self.cruise_altitude
        self.side_switch_area_rel = self.default_side_switch_area_rel

    def begin_reswipe_backoff(self, sensor_data, altitude):
        backward_direction = -np.array(
            [np.cos(sensor_data["yaw"]), np.sin(sensor_data["yaw"])],
            dtype=float,
        )
        current_xy = np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)
        self.reswipe_target = current_xy + self.reswipe_backoff_distance * backward_direction
        self.reswipe_altitude = altitude
        self.reswipe_attempts += 1
        self.side_switch_area_rel = 0.5 * self.default_side_switch_area_rel
        self.state = "RESWIPE_BACKOFF"
        print("Post-swipe triangulation failed, backing off for reswipe", self.reswipe_attempts)
        return [
            sensor_data["x_global"],
            sensor_data["y_global"],
            altitude,
            sensor_data["yaw"],
        ]

    def post_side_sample_resolution(self, sensor_data, error, altitude):
        print(
            "Post-swipe resolution:",
            "samples",
            len(self.gate_bearing_observations),
            "baseline",
            round(self.triangulation_baseline(), 3),
            "area",
            round(error["area_rel"], 4),
        )
        z_error = error["dy"] - self.gate_vertical_bias
        if z_error > 0.0:
            z_step = np.clip(-0.75 * z_error, -0.16, 0.0)
        else:
            z_step = np.clip(-0.3 * z_error, 0.0, 0.07)
        z_target = altitude + z_step

        if self.maybe_store_gate(sensor_data, error):
            print("Post-swipe triangulation succeeded, passing through")
            self.state = "PASS_THROUGH"
            self.pass_through_timer = self.pass_through_duration
            self.pass_through_heading = sensor_data["yaw"]
            self.pass_through_altitude = z_target
            return self.pass_through_command(sensor_data, 0.0)

        print(
            "Post-swipe triangulation failure reason:",
            self.last_triangulation_failure_reason or "unknown",
        )
        if self.reswipe_attempts < self.max_reswipe_attempts:
            return self.begin_reswipe_backoff(sensor_data, altitude)

        print("Post-swipe triangulation failed, reswipe limit reached; accepting best estimate")
        if self.accept_best_gate_estimate(sensor_data, error):
            self.state = "PASS_THROUGH"
            self.pass_through_timer = self.pass_through_duration
            self.pass_through_heading = sensor_data["yaw"]
            self.pass_through_altitude = z_target
            return self.pass_through_command(sensor_data, 0.0)

        print("No fallback gate estimate available after reswipe limit")
        return [
            sensor_data["x_global"],
            sensor_data["y_global"],
            z_target,
            sensor_data["yaw"],
        ]

    def triangulation_baseline(self):
        if len(self.gate_bearing_observations) < 2:
            return 0.0
        origins = np.array([obs["origin"] for obs in self.gate_bearing_observations], dtype=float)
        separations = origins[:, None, :] - origins[None, :, :]
        return float(np.max(np.linalg.norm(separations, axis=2)))

    def triangulate_gate_position(self):
        if len(self.gate_bearing_observations) < self.min_triangulation_observations:
            self.last_triangulation_failure_reason = "not enough samples"
            return None

        baseline = self.triangulation_baseline()
        if baseline < self.min_triangulation_baseline:
            self.last_triangulation_failure_reason = (
                f"baseline too small ({baseline:.3f} < {self.min_triangulation_baseline:.3f})"
            )
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

        condition = np.linalg.cond(a_matrix)
        if weight_sum <= 1e-6:
            self.last_triangulation_failure_reason = "zero weight sum"
            return None
        if condition > 40.0:
            self.last_triangulation_failure_reason = f"ill-conditioned solve ({float(condition):.2f})"
            return None

        gate_position = np.linalg.solve(a_matrix, b_vector)
        self.last_triangulation_failure_reason = None
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

    def store_gate_candidate(self, sensor_data, gate_candidate, source_label="Triangulated", allow_close=False):
        gate_candidate = self.shift_gate_toward_drone(sensor_data, gate_candidate)

        if self.learned_gates and not allow_close:
            previous = np.array(self.learned_gates[-1]["gate"])
            if np.linalg.norm(np.array(gate_candidate) - previous) < 1.0:
                self.last_triangulation_failure_reason = "candidate too close to previous gate"
                self.reset_gate_localization()
                return False

        self.learned_gates.append(
            {
                "gate": gate_candidate,
                "heading": float(sensor_data["yaw"]),
            }
        )
        print(source_label, "gate", len(self.learned_gates), "at", np.round(gate_candidate, 2).tolist())
        self.reset_gate_localization()
        return True

    def accept_best_gate_estimate(self, sensor_data, error):
        if len(self.learned_gates) >= self.expected_gate_count:
            self.reset_gate_localization()
            return False

        if not self.pending_gate_samples:
            self.collect_gate_sample(sensor_data, error)
        if not self.pending_gate_samples:
            return False

        gate_candidate = np.mean(np.array(self.pending_gate_samples, dtype=float), axis=0).tolist()
        return self.store_gate_candidate(
            sensor_data,
            gate_candidate,
            source_label="Fallback accepted",
            allow_close=True,
        )

    def maybe_store_gate(self, sensor_data, error):
        if len(self.learned_gates) >= self.expected_gate_count:
            self.reset_gate_localization()
            return False

        self.collect_gate_sample(sensor_data, error)
        gate_candidate = self.triangulate_gate_position()
        if gate_candidate is None:
            return False

        return self.store_gate_candidate(sensor_data, gate_candidate)

    def begin_side_sample(self, sensor_data, altitude, initial_error=None):
        left_direction = np.array([-np.sin(sensor_data["yaw"]), np.cos(sensor_data["yaw"])], dtype=float)
        right_direction = -left_direction
        current_xy = np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)
        right_target = current_xy + self.side_sample_distance * right_direction
        left_target = current_xy + self.side_sample_distance * left_direction

        self.side_sample_target = left_target
        self.side_sample_right_target = right_target
        self.side_sample_left_target = left_target
        self.side_sample_altitude = altitude
        self.side_sample_timer = self.side_sample_duration
        self.side_sample_used = True
        self.side_sample_phase = "LEFT_SCAN"
        self.side_sample_best_area = initial_error["area_rel"] if initial_error is not None else 0.0
        self.side_sample_best_position = current_xy.copy()
        self.side_sample_phase_best_area = self.side_sample_best_area
        self.side_sample_last_center_px = initial_error["center_px"] if initial_error is not None else None
        self.side_sample_near_zero_frames = 0
        self.state = "SIDE_SAMPLE"
        print("Side sample sweeping left")
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
        self.side_sample_phase_best_area = 0.0
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0
        print("Side sample sweeping left")

    def switch_side_sample_to_right(self):
        if self.side_sample_right_target is None or self.side_sample_phase != "LEFT_SCAN":
            return

        self.side_sample_target = self.side_sample_right_target.copy()
        self.side_sample_phase = "RIGHT_SCAN"
        self.side_sample_phase_best_area = 0.0
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0
        print("Side sample sweeping right")

    def switch_side_sample_to_best(self):
        if self.side_sample_best_position is None:
            return

        self.side_sample_target = self.side_sample_best_position.copy()
        self.side_sample_phase = "RETURN_BEST"
        self.side_sample_phase_best_area = 0.0
        self.side_sample_last_center_px = None
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
            phase_best_area = self.side_sample_phase_best_area
            switch_area_threshold = 0.2 * phase_best_area
            should_switch_side = (
                self.side_sample_phase in ("LEFT_SCAN", "RIGHT_SCAN")
                and phase_best_area > 0.0
                and current_area < switch_area_threshold
            )
            if should_switch_side:
                print(
                    "Swipe switch check:",
                    "phase",
                    self.side_sample_phase,
                    "area",
                    round(current_area, 4),
                    "threshold",
                    round(switch_area_threshold, 4),
                )
                if self.side_sample_phase == "LEFT_SCAN":
                    self.switch_side_sample_to_right()
                elif self.side_sample_phase == "RIGHT_SCAN":
                    self.switch_side_sample_to_left()
            else:
                self.side_sample_near_zero_frames = 0
                self.side_sample_last_center_px = error["center_px"]
                self.collect_gate_sample(sensor_data, error)
                usable_error = error

            if not should_switch_side and current_area > self.side_sample_best_area:
                self.side_sample_best_area = current_area
                self.side_sample_best_position = current_xy.copy()
            if not should_switch_side and current_area > self.side_sample_phase_best_area:
                self.side_sample_phase_best_area = current_area
        else:
            self.side_sample_near_zero_frames += 1
            if (
                self.side_sample_phase == "LEFT_SCAN"
                and self.side_sample_near_zero_frames >= self.side_sample_required_zero_frames
            ):
                print("Swipe lost gate on left, switching to right")
                self.switch_side_sample_to_right()

        reached_target = np.linalg.norm(self.side_sample_target - current_xy) < 0.05
        if reached_target and self.side_sample_phase == "LEFT_SCAN":
            print("Swipe reached left target, switching to right")
            self.switch_side_sample_to_right()
            reached_target = False
        if reached_target and self.side_sample_phase == "RIGHT_SCAN":
            print("Swipe reached right target, returning to best area")
            self.switch_side_sample_to_best()
            reached_target = False

        if self.side_sample_timer <= 0.0 and self.side_sample_phase != "RETURN_BEST":
            print("Swipe timer expired, returning to best area")
            self.switch_side_sample_to_best()
            self.side_sample_timer = 1.5
            reached_target = False

        if (reached_target and self.side_sample_phase == "RETURN_BEST") or self.side_sample_timer <= 0.0:
            self.side_sample_target = None
            self.side_sample_timer = 0.0
            self.side_sample_phase = None
            self.state = "TRACK"
            if usable_error is None:
                mask = self.compute_mask(camera_data)
                candidates = self.rectangle_candidates(mask, camera_data.shape)
                usable_error = self.choose_gate_candidate(candidates, track_switches=False)
            if usable_error is not None:
                return self.post_side_sample_resolution(sensor_data, usable_error, self.side_sample_altitude)
            if self.reswipe_attempts < self.max_reswipe_attempts:
                return self.begin_reswipe_backoff(sensor_data, self.side_sample_altitude)
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

    def reswipe_backoff_command(self, sensor_data, camera_data):
        if self.reswipe_target is None:
            self.state = "TRACK"
            return [
                sensor_data["x_global"],
                sensor_data["y_global"],
                sensor_data["z_global"],
                sensor_data["yaw"],
            ]

        current_xy = np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)
        delta = self.reswipe_target - current_xy
        distance = np.linalg.norm(delta)
        if distance > 0.05:
            step = min(distance, self.reswipe_setpoint_distance)
            command_xy = current_xy + step * delta / distance
            return [
                float(command_xy[0]),
                float(command_xy[1]),
                self.reswipe_altitude,
                sensor_data["yaw"],
            ]

        self.reswipe_target = None
        self.side_sample_used = False
        mask = self.compute_mask(camera_data)
        candidates = self.rectangle_candidates(mask, camera_data.shape)
        error = self.choose_gate_candidate(candidates, track_switches=False)
        if error is not None:
            return self.begin_side_sample(sensor_data, self.reswipe_altitude, error)

        self.state = "TRACK"
        return [
            sensor_data["x_global"],
            sensor_data["y_global"],
            self.reswipe_altitude,
            sensor_data["yaw"],
        ]

    def build_replay_trajectory(self, entry_position=None):
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

        if entry_position is not None and trajectory:
            entry_point = np.array(entry_position, dtype=float)
            first_target = np.array(trajectory[0], dtype=float)
            if np.linalg.norm(first_target - entry_point) > self.replay_switch_distance:
                merge_heading = np.arctan2(
                    first_target[1] - entry_point[1],
                    first_target[0] - entry_point[0],
                )
                trajectory.insert(0, entry_point.tolist())
                trajectory_types.insert(0, "entry")
                trajectory_headings.insert(0, merge_heading)

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

    def replay_target_distance(self, target, current_position):
        target_type = (
            self.replay_trajectory_types[self.replay_traj_index]
            if self.replay_trajectory_types
            else None
        )
        if target_type in ("approach", "gate", "exit", "entry"):
            return float(np.linalg.norm(target[:2] - current_position[:2]))
        return float(np.linalg.norm(target - current_position))

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
            sensor_data["x_global"] + self.search_sweep_translation * np.cos(sensor_data["yaw"]),
            sensor_data["y_global"] + self.search_sweep_translation * np.sin(sensor_data["yaw"]),
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

        if self.rotate_left_accum >= np.pi:
            self.search_radius = self.default_search_radius
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

        if (
            self.require_side_sample_before_store
            and not self.side_sample_used
            and error["area_rel"] > 0.04
            and centered
        ):
            return self.begin_side_sample(sensor_data, z_target, error)

        enough_bearings = len(self.gate_bearing_observations) >= self.min_triangulation_observations
        baseline_ready = self.triangulation_baseline() >= self.min_triangulation_baseline
        if error["area_rel"] > self.triangulation_trigger_area and very_centered and enough_bearings:
            if not baseline_ready and not self.side_sample_used:
                return self.begin_side_sample(sensor_data, z_target, error)
            if not baseline_ready:
                return [
                    sensor_data["x_global"],
                    sensor_data["y_global"],
                    z_target,
                    yaw_target,
                ]
            if not self.maybe_store_gate(sensor_data, error):
                if self.side_sample_used and self.reswipe_attempts < self.max_reswipe_attempts:
                    return self.begin_reswipe_backoff(sensor_data, z_target)
                return [
                    sensor_data["x_global"] + 0.04 * np.cos(sensor_data["yaw"]),
                    sensor_data["y_global"] + 0.04 * np.sin(sensor_data["yaw"]),
                    z_target,
                    yaw_target,
                ]
            self.state = "PASS_THROUGH"
            self.pass_through_timer = self.pass_through_duration
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

        distance = self.pass_through_distance
        x_target = sensor_data["x_global"] + distance * np.cos(self.pass_through_heading)
        y_target = sensor_data["y_global"] + distance * np.sin(self.pass_through_heading)

        if self.pass_through_timer <= 0.0:
            self.state = "ADVANCE"
            self.advance_timer = self.advance_duration
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

        distance = self.advance_distance
        x_target = sensor_data["x_global"] + distance * np.cos(self.pass_through_heading)
        y_target = sensor_data["y_global"] + distance * np.sin(self.pass_through_heading)

        if self.advance_timer <= 0.0:
            self.state = "SEARCH"
            self.height_aligned = False
            self.reset_search_pattern(sensor_data["yaw"])
            if len(self.learned_gates) >= self.expected_gate_count:
                self.build_replay_trajectory(
                    entry_position=[
                        sensor_data["x_global"],
                        sensor_data["y_global"],
                        self.pass_through_altitude,
                    ]
                )
                self.state = "REPLAY"
            elif self.learned_gates:
                self.state = "GO_TO_FIRST_LAP_CONE"
                self.first_lap_cone_target = self.current_angle_radius_target(
                    sensor_data,
                    self.pass_through_altitude,
                )
                return self.go_to_first_lap_cone_command(sensor_data)

        return [
            x_target,
            y_target,
            self.pass_through_altitude,
            self.pass_through_heading,
        ]

    def go_to_first_lap_cone_command(self, sensor_data):
        if self.first_lap_cone_target is None:
            self.first_lap_cone_target = self.current_angle_radius_target(
                sensor_data,
                self.pass_through_altitude,
            )
        target = self.first_lap_cone_target
        current_xy = np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)
        delta_xy = target[:2] - current_xy
        distance = np.linalg.norm(delta_xy)

        if distance < self.first_lap_cone_tol:
            self.first_lap_cone_target = None
            self.state = "SEARCH"
            self.height_aligned = False
            self.reset_search_pattern(float(target[3]))
            return self.search_command(sensor_data, 0.0)

        if distance > 1e-6:
            command_xy = current_xy + min(distance, self.first_lap_cone_step) * delta_xy / distance
        else:
            command_xy = current_xy

        return [
            float(command_xy[0]),
            float(command_xy[1]),
            float(target[2]),
            sensor_data["yaw"],
        ]

    def replay_command(self, sensor_data):
        if not self.replay_trajectory:
            self.state = "SEARCH"
            return self.search_command(sensor_data, 0.0)
        current_position = np.array(
            [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
            dtype=float,
        )

        while True:
            target = np.array(self.replay_trajectory[self.replay_traj_index], dtype=float)
            delta = target - current_position
            distance = self.replay_target_distance(target, current_position)
            target_type = (
                self.replay_trajectory_types[self.replay_traj_index]
                if self.replay_trajectory_types
                else None
            )
            switch_distance = self.replay_current_tolerance()
            if target_type not in ("approach", "gate"):
                switch_distance = max(switch_distance, self.replay_switch_distance)

            if distance >= switch_distance:
                break

            self.print_replay_target_reached()
            self.replay_traj_index += 1
            if self.replay_traj_index >= len(self.replay_trajectory):
                self.replay_traj_index = 0
                self.replay_lap += 1
                if self.replay_lap >= self.total_replay_laps:
                    self.state = "RETURN_HOME"
                    return self.return_home_command(sensor_data)

        target = np.array(self.replay_trajectory[self.replay_traj_index], dtype=float)
        delta = target - current_position
        distance = self.replay_target_distance(target, current_position)

        yaw_target = sensor_data["yaw"]

        command_target = target
        is_final_replay_target = (
            self.replay_lap == self.total_replay_laps - 1
            and self.replay_traj_index == len(self.replay_trajectory) - 1
        )
        if not is_final_replay_target and len(self.replay_trajectory) > 1:
            next_index = (self.replay_traj_index + 1) % len(self.replay_trajectory)
            next_target = np.array(self.replay_trajectory[next_index], dtype=float)
            segment = next_target - target
            segment_norm = np.linalg.norm(segment)
            if segment_norm > 1e-6:
                lookahead_ratio = np.clip(
                    1.0 - distance / self.replay_lookahead_distance,
                    0.0,
                    1.0,
                )
                along_segment = min(self.replay_segment_lookahead, segment_norm)
                command_target = target + lookahead_ratio * along_segment * segment / segment_norm

        command_xy = command_target[:2]
        command_delta_xy = command_xy - current_position[:2]
        command_distance_xy = np.linalg.norm(command_delta_xy)
        if command_distance_xy > self.replay_xy_step:
            command_xy = current_position[:2] + self.replay_xy_step * command_delta_xy / command_distance_xy

        z_error = float(command_target[2] - current_position[2])
        command_z = current_position[2] + np.clip(
            z_error,
            -self.replay_z_step,
            self.replay_z_step,
        )

        return [
            float(command_xy[0]),
            float(command_xy[1]),
            float(command_z),
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

        if self.state == "GO_TO_FIRST_LAP_CONE":
            return self.go_to_first_lap_cone_command(sensor_data)

        if self.state == "SIDE_SAMPLE":
            return self.side_sample_command(sensor_data, camera_data, dt)

        if self.state == "RESWIPE_BACKOFF":
            return self.reswipe_backoff_command(sensor_data, camera_data)

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
