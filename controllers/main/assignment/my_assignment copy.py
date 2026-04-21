import numpy as np
import time
import cv2


class MyAssignment:
    def __init__(self):
        self.portals = []
        self.i = 0
        self.lap = 0

        """ self.check_points_list = [
            [1.5, 3.5, 1.35, np.deg2rad(-30)],
            [3.5, 1.5, 1.35, np.deg2rad(30)],
            [6.5, 3.5, 1.35, np.deg2rad(90)],
            [6.5, 5.5, 1.35, np.deg2rad(160)],
            [4.5, 6.5, 1.35, np.deg2rad(-140)],
            [2.5, 4.5, 1.35, np.deg2rad(-80)]
        ]"""
        self.check_points_list = [
            [4, 4, 1.35, np.deg2rad(-120)],
            [4, 4, 1.35, np.deg2rad(-90)],
            [4, 4, 1.35, np.deg2rad(-30)],
            [4, 4, 1.35, np.deg2rad(30)],
            [4, 4, 1.35, np.deg2rad(90)],
            [4, 4, 1.35, np.deg2rad(120)]
        ]

        # Finite state machine
        self.state = "TAKEOFF"     # TAKEOFF -> CHECKPOINTS -> DETECT -> RACE -> DONE

        # Portal detection / traversal internals
        self.wait = True
        self.x_wait = 0
        self.y_wait = 0
        self.portal_detected = False
        self.in_portal = False
        self.swipeR = True

    def compute_mask(self, camera_data):
        bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([120, 40, 150])
        upper = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    def rectangle_error(self, mask, img_shape, min_area=100):
        H, W = img_shape[:2]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box)

        # centre du rectangle
        cx = box[:, 0].mean()
        cy = box[:, 1].mean()

        # centre image
        ix = W / 2
        iy = H / 2

        # erreur normalisée
        dx = (cx - ix) / (W / 2)
        dy = (cy - iy) / (H / 2)

        # aire relative
        area_rect = cv2.contourArea(c)
        area_rel = area_rect / (H * W)

        return {
            "dx": dx,
            "dy": dy,
            "area_rel": area_rel
        }

    def go_forward(self, error, sensor_data):
        if error is None and not self.in_portal:
            if self.swipeR and sensor_data['yaw'] <= (self.check_points_list[self.i][3] + np.deg2rad(19)) % (2 * np.pi):
                print("right swipe")
                return (
                    sensor_data['x_global'],
                    sensor_data['y_global'],
                    1.35,
                    (self.check_points_list[self.i][3] + np.deg2rad(20)) % (2 * np.pi)
                )
            else:
                print("left swipe")
                self.swipeR = False
                return (
                    sensor_data['x_global'],
                    sensor_data['y_global'],
                    1.35,
                    (self.check_points_list[self.i][3] - np.deg2rad(40)) % (2 * np.pi)
                )

        if self.portal_detected or error['area_rel'] > 0.2:
            # slow down when close to the portal
            x = sensor_data['x_global'] + 0.1 * np.cos(sensor_data['yaw'])
            y = sensor_data['y_global'] + 0.1 * np.sin(sensor_data['yaw'])
            self.portal_detected = True

        if self.in_portal or (self.portal_detected and error['area_rel'] < 0.01):
            self.in_portal = True
            x = sensor_data['x_global']
            y = sensor_data['y_global']
            n = 1
            x = x + n*(self.check_points_list[self.i+1][0]-x)
            y = y + n*(self.check_points_list[self.i+1][1]-y)
            print("strait, ",x - self.x_wait)
            if self.wait:
                print("wait")
                self.x_wait = sensor_data['x_global']
                self.y_wait = sensor_data['y_global']
                self.wait = False

            elif abs(x - self.x_wait) > 0.1 and abs(y - self.y_wait) > 0.1:
                print("portal added", sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'])
                self.portals.append((sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']))
                print("Portals:", self.portals)

                self.wait = True
                self.in_portal = False
                self.portal_detected = False
                self.swipeR = True

                # return to checkpoint navigation
                self.state = "CHECKPOINTS"
                time.sleep(2)

            return x, y, sensor_data['z_global'], sensor_data['yaw']

        elif abs(error['dx']) < 0.2 and abs(error['dy']) < 0.2:
            x = sensor_data['x_global'] + 0.5 * np.cos(sensor_data['yaw'])
            y = sensor_data['y_global'] + 0.5 * np.sin(sensor_data['yaw'])
        else:
            x = sensor_data['x_global']
            y = sensor_data['y_global']

        if error['dx'] > 0.15:
            yaw = sensor_data['yaw'] - np.radians(1)
        elif error['dx'] < -0.15:
            yaw = sensor_data['yaw'] + np.radians(1)
        else:
            yaw = sensor_data['yaw']

        if error['dy'] > 0.1:
            z = sensor_data['z_global'] - 0.05
        elif error['dy'] < -0.1:
            z = sensor_data['z_global'] + 0.05
        else:
            z = sensor_data['z_global']

        self.swipeR = True
        return x, y, z, yaw

    def checkpoint_reached(self, sensor_data, checkpoint):
        return (
            checkpoint[0] - 0.5 < sensor_data['x_global'] < checkpoint[0] + 0.5 and
            checkpoint[1] - 0.5 < sensor_data['y_global'] < checkpoint[1] + 0.5 and
            checkpoint[2] - 0.5 < sensor_data['z_global'] < checkpoint[2] + 0.5 and
            abs(sensor_data['yaw'] - checkpoint[3]) < np.deg2rad(2)
        )

    def portal_reached(self, sensor_data, portal):
        return (
            portal[0] - 0.2 < sensor_data['x_global'] < portal[0] + 0.2 and
            portal[1] - 0.2 < sensor_data['y_global'] < portal[1] + 0.2 and
            portal[2] - 0.2 < sensor_data['z_global'] < portal[2] + 0.2
        )

    def compute_command(self, sensor_data, camera_data, dt):
        # -------------------------
        # STATE: TAKEOFF
        # -------------------------
        if self.state == "TAKEOFF":
            if sensor_data['z_global'] < 0.5:
                return [sensor_data['x_global'], sensor_data['y_global'], 1.35, sensor_data['yaw']]

            self.state = "CHECKPOINTS"

        # -------------------------
        # STATE: CHECKPOINTS
        # -------------------------
        if self.state == "CHECKPOINTS":
            current_checkpoint = self.check_points_list[self.i]

            if self.checkpoint_reached(sensor_data, current_checkpoint):
                print("Checkpoint reached:", self.i)

                if self.i < len(self.check_points_list) - 1:
                    self.state = "DETECT"
                else:
                    print("All checkpoints reached, Race mode.")
                    self.i = 0
                    self.state = "RACE"
                    return [sensor_data['x_global'], sensor_data['y_global'], 1.35, sensor_data['yaw']]

                time.sleep(1)

            return current_checkpoint

        # -------------------------
        # STATE: DETECT
        # -------------------------
        if self.state == "DETECT":
            mask = self.compute_mask(camera_data)
            cv2.imwrite("mask.png", mask)

            error = self.rectangle_error(mask, camera_data.shape)
            x, y, z, yaw = self.go_forward(error, sensor_data)

            # if portal was traversed, go_forward may have switched state back to CHECKPOINTS
            if self.state == "CHECKPOINTS":
                if self.i < len(self.check_points_list) - 1:
                    self.i += 1

            return [x, y, z, yaw]

        # -------------------------
        # STATE: RACE
        # -------------------------
        if self.state == "RACE":
            if len(self.portals) == 0:
                return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

            current_portal = self.portals[self.i]

            if self.portal_reached(sensor_data, current_portal):
                if self.i < len(self.portals) - 1:
                    print("Portal passed:", self.portals[self.i])
                    time.sleep(3)
                    self.i += 1
                elif self.lap < 1:
                    print("All portals passed - Lap", self.lap + 1)
                    self.i = 0
                    self.lap += 1
                else:
                    print("Race completed")
                    self.state = "DONE"

            if self.state == "RACE":
                return [
                    self.portals[self.i][0],
                    self.portals[self.i][1],
                    self.portals[self.i][2],
                    sensor_data['yaw']
                ]

        # -------------------------
        # STATE: DONE
        # -------------------------
        if self.state == "DONE":
            return [
                sensor_data['x_global'],
                sensor_data['y_global'],
                sensor_data['z_global'],
                sensor_data['yaw']
            ]

        return [
            sensor_data['x_global'],
            sensor_data['y_global'],
            sensor_data['z_global'],
            sensor_data['yaw']
        ]


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)