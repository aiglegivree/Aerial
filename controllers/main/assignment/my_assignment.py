import numpy as np
import time
import cv2

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate


class MyAssignment:
    def __init__(self):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.angle = -90
        self.x_next = 3*np.cos(np.radians(self.angle))+4
        self.y_next = 3*np.sin(np.radians(self.angle))+4
        self.go = False

    def compute_mask(self, camera_data):
        bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([120, 40, 150])
        upper = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return mask
            
    def rectangle_error(self,mask, img_shape, min_area=200):

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

        # rectangle center
        cx = box[:,0].mean()
        cy = box[:,1].mean()

        # image center
        ix = W / 2
        iy = H / 2

        dx = cx - ix
        dy = cy - iy

        # normalized (recommended)
        dx_n = dx / (W/2)
        dy_n = dy / (H/2)

        return {
            "dx_px": dx,
            "dy_px": dy,
            "dx_norm": dx_n,
            "dy_norm": dy_n,
            "center_rect": (cx, cy),
            "center_img": (ix, iy),
            "box": box
        }
    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Take off example
        if sensor_data['z_global'] < 0.49:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, np.radians(-30)]
            return control_command

        # ---- YOUR CODE HERE ----
        """
        if sensor_data['x_global'] < self.x_next+0.5 and sensor_data['x_global'] > self.x_next-0.5 and sensor_data['y_global'] < self.y_next+0.5 and sensor_data['y_global'] > self.y_next-0.5:
            self.angle += 10
            self.x_next = 3*np.cos(np.radians(self.angle))+4
            self.y_next = 3*np.sin(np.radians(self.angle))+4
        control_command = [self.x_next, self.y_next, 1.0, np.radians(self.angle+90)] """

        mask = self.compute_mask(camera_data)
        cv2.imwrite("mask.png", mask)

        error=self.rectangle_error(mask, camera_data.shape)

        if error is None:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            return control_command
        elif error['dx_px'] > 0.4:
            yaw = sensor_data['yaw'] - np.radians(1)
        elif error['dx_px'] < -0.4:
            yaw = sensor_data['yaw'] + np.radians(1)
        else:
            yaw = sensor_data['yaw']

        if error['dy_px'] > 0.2:
            z = sensor_data['z_global'] - 0.05
        elif error['dy_px'] < -0.2:
            z = sensor_data['z_global'] + 0.05
        else:
            z = sensor_data['z_global']

        if z == sensor_data['z_global'] and yaw == sensor_data['yaw'] or self.go:
            self.go = True
            control_command = [sensor_data['x_global']+1*np.cos(sensor_data['yaw']), sensor_data['y_global']+1*np.sin(sensor_data['yaw']), z, yaw]
            print('go forward')
        else:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], z, yaw]
            print('adjusting position')
        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
