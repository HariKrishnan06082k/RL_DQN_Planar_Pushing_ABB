from RWS_IRC5 import RWS_IRC5 
import time

class Robot:
    def __init__(self):

        self.robot = RWS_IRC5(base_url='http://192.168.132.56', username='Default User', password='robotics')
        self.robot.request_mastership
        self.stay_same = 500

    def is_cam_ready(self):
        ''' If it becomes 1 then take picture from realsense and get RGB frame'''
        return self.robot.get_rapid_IO('do_check')
    
    def reset(self):
        ''' Digital signal to make the robot go to home pos in (0,0) analogous to env.reset() in gym'''
        self.robot.set_rapid_IO('do_RL_reset',1) #pulse triggered no need to hard set to 0
        self.robot.set_rapid_IO('do_startmove', 1)
        
    def move_rob(self,action,action_check,offset_X=60,offset_Y=60):
        '''Move the robot in either LEFT, RIGHT, FORWARD or BACKWARD'''
        
        '''
        IF the action received is left then +y (550,500)
        If the action received is right then -y (450,500)
        If the action received is forward then +x (500,550)
        If the action received is backward then -x (500,450)
        '''
        
        if action == 1:
            if action_check == False:
                self.robot.set_rapid_IO('GO_y', self.stay_same)
                self.robot.set_rapid_IO('GO_x', self.stay_same)
            else:
                self.robot.set_rapid_IO('GO_y', offset_Y + 500)
                self.robot.set_rapid_IO('GO_x', 0 + 500)
        if action == 0:
            if action_check == False:
                self.robot.set_rapid_IO('GO_y', self.stay_same)
                self.robot.set_rapid_IO('GO_x', self.stay_same)
            else:
                self.robot.set_rapid_IO('GO_y', -offset_Y + 500)
                self.robot.set_rapid_IO('GO_x', 0 + 500)
        if action == 3:
            if action_check == False:
                self.robot.set_rapid_IO('GO_y', self.stay_same)
                self.robot.set_rapid_IO('GO_x', self.stay_same)
            else:
                self.robot.set_rapid_IO('GO_x', offset_X + 500)
                self.robot.set_rapid_IO('GO_y', 500)
        if action == 2:
            if action_check == False:
                self.robot.set_rapid_IO('GO_y', self.stay_same)
                self.robot.set_rapid_IO('GO_x', self.stay_same)
            else:
                self.robot.set_rapid_IO('GO_x', -offset_X + 500)
                self.robot.set_rapid_IO('GO_y', 500)
                
        self.robot.set_rapid_IO('do_startmove', 1)
    