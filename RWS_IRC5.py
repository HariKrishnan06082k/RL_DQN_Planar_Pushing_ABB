from requests.auth import HTTPDigestAuth
from requests import ConnectTimeout, Session
import ast
import time
import json
import math
  

class RWS_IRC5:
   

    def __init__(self, base_url, username='Default User', password='robotics'):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.session = Session()  # create persistent HTTP communication
        self.session.auth = HTTPDigestAuth(self.username, self.password)

    def set_rapid_variable(self,  mechunit , module, var, value,):
        """Sets the value of any RAPID variable.
        Unless the variable is of type 'num', 'value' has to be a string.
        """

        payload = {'value': value}
        resp = self.session.post(self.base_url + '/rw/rapid/symbol/data/RAPID/'+ mechunit +'/'+ module+'/' + var + '?action=set',
                                 data=payload)
        return resp

    def get_rapid_variable(self,  mechunit , module, var):
        """Gets the raw value of any RAPID variable.
        """

        resp = self.session.get(self.base_url + '/rw/rapid/symbol/data/RAPID/'+ mechunit +'/'+ module+'/' + var + ';value?json=1')
        json_string = resp.text
        _dict = json.loads(json_string)
        value = _dict["_embedded"]["_state"][0]["value"]
        return value

    def get_rapid_IO(self, Signal_name):
        resp = self.session.get(self.base_url + '/rw/iosystem/signals/'+ Signal_name +'?json=1')
        json_string = resp.text
        _dict = json.loads(json_string)
        value = _dict["_embedded"]["_state"][0]["lvalue"]
        return value


    def set_rapid_IO(self,  Signal_name , value ):
        """Gets the raw value of any RAPID variable.
        """
        payload = {'lvalue': value}
        try:
            resp = self.session.post(self.base_url + '/rw/iosystem/signals/'+ Signal_name +  '?action=set',
                                data=payload,timeout=0.1)
        except ConnectTimeout:
            print("Cannot setup signal")
            return
        return resp


    def get_robtarget_variables(self ,mechunit, module, var):
        """Gets both translational and rotational data from robtarget.
        """

        resp = self.session.get(self.base_url + '/rw/rapid/symbol/data/RAPID/'+ mechunit +'/'+ module+'/' + var + ';value?json=1')
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["value"]
        data_list = ast.literal_eval(data)  # Convert the pure string from data to list
        trans = data_list[0]  # Get x,y,z from robtarget relative to work object (table)
        rot = data_list[1]  # Get orientation of robtarget
        return trans, rot


    def reset_pp(self):
        """Resets the program pointer to main procedure in RAPID.
        """

        resp = self.session.post(self.base_url + '/rw/rapid/execution?action=resetpp')
        if resp.status_code == 204:
            print('Program pointer reset to main')
        else:
            print('Could not reset program pointer to main')

    def request_mastership(self):
        resp = self.session.post(self.base_url + '/rw/mastership')

    def release_mastership(self):
        resp = self.session.post(self.base_url + '/rw/mastership?action=release')

    def motors_on(self):
        """Turns the robot's motors on.
        Operation mode has to be AUTO.
        """

        payload = {'ctrl-state': 'motoron'}
        resp = self.session.post(self.base_url + "/rw/panel/ctrlstate?action=setctrlstate", data=payload)

        if resp.status_code == 204:
            print("Robot motors turned on")
        else:
            print("Could not turn on motors. The controller might be in manual mode")

    def motors_off(self):
        """Turns the robot's motors off.
        """

        payload = {'ctrl-state': 'motoroff'}
        resp = self.session.post(self.base_url + "/rw/panel/ctrlstate?action=setctrlstate", data=payload)

        if resp.status_code == 204:
            print("Robot motors turned off")
        else:
            print("Could not turn off motors")

    def start_RAPID(self):
        """Resets program pointer to main procedure in RAPID and starts RAPID execution.
        """

        self.reset_pp()
        payload = {'regain': 'continue', 'execmode': 'continue', 'cycle': 'once', 'condition': 'none',
                   'stopatbp': 'disabled', 'alltaskbytsp': 'false'}
        resp = self.session.post(self.base_url + "/rw/rapid/execution?action=start", data=payload)
        if resp.status_code == 204:
            print("RAPID execution started from main")
        else:
            opmode = self.get_operation_mode()
            ctrlstate = self.get_controller_state()

            print(f"""
            Could not start RAPID. Possible causes:
            * Operating mode might not be AUTO. Current opmode: {opmode}.
            * Motors might be turned off. Current ctrlstate: {ctrlstate}.
            * RAPID might have write access. 
            """)

    def stop_RAPID(self):
        """Stops RAPID execution.
        """

        payload = {'stopmode': 'stop', 'usetsp': 'normal'}
        resp = self.session.post(self.base_url + "/rw/rapid/execution?action=stop", data=payload)
        if resp.status_code == 204:
            print('RAPID execution stopped')
        else:
            print('Could not stop RAPID execution')

    def get_execution_state(self):
        """Gets the execution state of the controller.
        """

        resp = self.session.get(self.base_url + "/rw/rapid/execution?json=1")
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["ctrlexecstate"]
        return data

    def is_running(self):
        """Checks the execution state of the controller and
        """

        execution_state = self.get_execution_state()
        if execution_state == "running":
            return True
        else:
            return False

    def get_operation_mode(self):
        """Gets the operation mode of the controller.
        """

        resp = self.session.get(self.base_url + "/rw/panel/opmode?json=1")
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["opmode"]
        return data

    def get_controller_state(self):
        """Gets the controller state.
        """

        resp = self.session.get(self.base_url + "/rw/panel/ctrlstate?json=1")
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["ctrlstate"]
        return data

    def set_speed_ratio(self, speed_ratio):
        """Sets the speed ratio of the controller.
        """

        if not 0 < speed_ratio <= 100:
            print("You have entered a false speed ratio value! Try again.")
            return

        payload = {'speed-ratio': speed_ratio}
        resp = self.session.post(self.base_url + "/rw/panel/speedratio?action=setspeedratio", data=payload)
        if resp.status_code == 204:
            print(f'Set speed ratio to {speed_ratio}%')
        else:
            print('Could not set speed ratio!')


def quaternion_to_radians(quaternion):
    """Convert a Quaternion to a rotation about the z-axis in degrees.
    """
    w, x, y, z = quaternion
    t1 = +2.0 * (w * z + x * y)
    t2 = +1.0 - 2.0 * (y * y + z * z)
    rotation_z = math.atan2(t1, t2)

    return rotation_z


def z_degrees_to_quaternion(rotation_z_degrees):
    """Convert a rotation about the z-axis in degrees to Quaternion.
    """
    roll = math.pi
    pitch = 0
    yaw = math.radians(rotation_z_degrees)

    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(
        pitch / 2) * math.sin(yaw / 2)
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(
        pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(
        pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(
        pitch / 2) * math.cos(yaw / 2)

    return [qw, qx, qy, qz]


