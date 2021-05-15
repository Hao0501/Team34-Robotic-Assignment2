#! /usr/bin/python

# Import the core Python modules for ROS and to implement ROS Actions:
import rospy
import actionlib

# Import all the necessary ROS message types:
from com2009_actions.msg import SearchFeedback, SearchResult, SearchAction
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image

# Import some image processing modules:
import cv2
from cv_bridge import CvBridge

# Import some other modules from within this package
from move_tb3 import MoveTB3
from tb3_odometry import TB3Odometry

# Import some other useful Python Modules
from math import sqrt, pow
import numpy as np

class BeaconFindServer(object):
    feedback = SearchFeedback() 
    result = SearchResult()
    def __init__(self):
        self.actionserver = actionlib.SimpleActionServer("/beacon_find_server", 
            SearchAction, self.action_server_launcher, auto_start=False)
        self.actionserver.start()

        self.base_image_path = '/home/student/myrosdata/week6_images'
        #subscribe to camera node
        self.camera_subscriber = rospy.Subscriber("/camera/rgb/image_raw",
            Image, self.camera_callback)
        self.cvbridge_interface = CvBridge()

        #subscribe to lidar scan data node
        self.scan_subscriber = rospy.Subscriber("/scan",
            LaserScan, self.scan_callback)
        
        self.robot_odom = TB3Odometry()
        self.robot_controller = MoveTB3()

        self.turn_vel_fast = 0.5
        self.turn_vel_fast_rev = -0.5
        self.turn_vel_med = 0.15
        self.turn_vel_slow = -0.1


        self.robot_controller.set_move_cmd(0.0, self.turn_vel_fast)

        self.move_rate = '' # fast, slow or stop
        
        self.task_stage = 1
        self.found_colour = False
        self.need_reverse = False
        self.found_beacon = False

        self.start_angle= self.robot_odom.yaw
        #convert angle to 0-360
        if self.start_angle<0:
            self.start_angle += 360
        #sometimes the odom doesn't return the actual angle, so set it to 90 by default
        if self.start_angle == 0.0:
            self.start_angle = 90
        
        self.min_distance = 0.5
        self.object_angle = 0
        self.arc_angles = np.arange(-40, 41)

        self.ctrl_c = False
        rospy.on_shutdown(self.shutdown_ops)

        self.lowers ={
            "blue":(115, 224, 100),
            "red":(-5, 225, 100),
            "green":(40, 150, 100),
            "cyan":(75,100,100),
            "purple":(145,485,100),
            "yellow":(22, 93, 0)
        }
        self.uppers ={
            "blue":(130, 255, 255),
            "red":(8, 255, 255),
            "green":(65, 255, 255),
            "cyan":(95,255,255),
            "purple":(150,255,255),
            "yellow":(45, 255, 255)
        }

        #blue, red, green, cyan, purple, yellow
        self.colour_boundaries = [
            [(115, 224, 100),(130, 255, 255),"blue"],
            [(-5, 225, 100),(8, 255, 255),"red"],
            [(40, 150, 100),(65, 255, 255),"green"],
            [(75,100,100),(95,255,255),"cyan"],
            [(145,485,100),(150,255,255),"purple"],
            [(22, 93, 0),(45, 255, 255),"yellow"]
        ]

        self.centre_pixel_colours = [0,0,0]
        self.rate = rospy.Rate(10)
        
        self.m00 = 0
        self.m00_min = 10000

        self.search_colour = "blue"

    def shutdown_ops(self):
        self.robot_controller.stop()
        cv2.destroyAllWindows()
        self.ctrl_c = True
    
    def camera_callback(self, img_data):
        try:
            cv_img = self.cvbridge_interface.imgmsg_to_cv2(img_data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        
        height, width, channels = cv_img.shape
        crop_width = width - 800
        crop_height = 400
        crop_x = int((width/2) - (crop_width/2))
        crop_y = int((height/2) - (crop_height/2))

        crop_img = cv_img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        centre_pixel = cv_img[crop_height/2:(crop_height/2)+1, crop_width/2:(crop_width/2)+1]
        centre_pixel_hsv = cv2.cvtColor(centre_pixel, cv2.COLOR_BGR2HSV)
        self.centre_pixel_colours = (centre_pixel_hsv[0][0][0],centre_pixel_hsv[0][0][1],centre_pixel_hsv[0][0][2])


        lower = (115, 224, 100)
        upper = (130, 255, 255)
        lower = self.lowers[self.search_colour]
        upper = self.uppers[self.search_colour]
        
        mask = cv2.inRange(hsv_img, lower, upper)
        res = cv2.bitwise_and(crop_img, crop_img, mask = mask)

        m = cv2.moments(mask)
        self.m00 = m['m00']
        self.cy = m['m10'] / (m['m00'] + 1e-5)

        if self.m00 > self.m00_min:
            cv2.circle(crop_img, (int(self.cy), 200), 10, (0, 0, 255), 2)
        
        cv2.imshow('cropped image', crop_img)
        cv2.waitKey(1)

    #lidar scan callback function - update new scan points and vision in front of robot
    def scan_callback(self, scan_data):
        left_arc = scan_data.ranges[0:37]
        right_arc = scan_data.ranges[-36:]
        #front arc is a combination of left and right arcs, normalised so angles run left to right continuously
        front_arc = np.array(left_arc[::-1] + right_arc[::-1])
        #closest distance measurement
        self.min_distance = front_arc.min()
        #direction of closest measurement
        self.object_angle = self.arc_angles[np.argmin(front_arc)]
    
    def action_server_launcher(self, goal):
        r = rospy.Rate(10)
        #goal is when the robot is about to hit an obstacle (movement controllers prevent this from happening)
        success = True
        #check for forward speed goal (can only move within robot's limits)
        if goal.fwd_velocity <= 0 or goal.fwd_velocity > 0.26:
            print("Invalid velocity.  Select a value between 0 and 0.26 m/s.")
            success = False
        #lower limit for stop distance - if lower, the robot will collide before it stops
        if goal.approach_distance <= 0.18:
            print("Invalid stop distance: Robot will crash")
            success = False

        #aborts action if any checks are violated
        if not success:
            self.actionserver.set_aborted()
            return

        print("Request to move at {:.3f}m/s and stop if less than{:.2f}m infront of an obstacle".format(goal.fwd_velocity, goal.approach_distance))

        # Get the current robot odometry:   
        self.posx0 = self.robot_odom.posx
        self.posy0 = self.robot_odom.posy


        while not self.ctrl_c:
            #convert angle to 0-360
            current_angle = self.robot_odom.yaw 
            if current_angle<0:
                current_angle += 360
            angle_difference = current_angle - self.start_angle
            if angle_difference<0:
                angle_difference+=360            
            #print(angle_difference)
            #print(current_angle)
            #get starting area colour
            #The reason why I've set several different current_angle statement is because they all have a different starting angle for all three starting area. Or there are better ways to finish
            #the searching state
            if self.task_stage == 1:
                #turn to the right and check centre pixel against colour ranges
                #if not fully turned 
                #start at area A(Green) When it starts from colour green, it comes up with "target colour is red"
                if current_angle < 92:
                    if self.found_colour == False and angle_difference<90:
                        self.robot_controller.set_move_cmd(0.0, -(self.turn_vel_med))
                    #if turned and hasn't got start colour yet
                    elif angle_difference>=90 and self.found_colour == False:
                        
                        self.robot_controller.set_move_cmd(0.0,0.0)
                        self.robot_controller.publish()
                        for colour in self.colour_boundaries:
                            if (self.centre_pixel_colours > colour[0]) and (self.centre_pixel_colours < colour[1]):
                                print("SEARCH INITIATED: The target colour is: " + colour[2])
                                self.search_colour = colour[2]
                                self.found_colour=True

                    
                    elif (angle_difference<10 or angle_difference>350) and self.found_colour == True:
                        self.robot_controller.set_move_cmd(0.0,0.0)
                        self.robot_controller.publish()
                        #move onto next stage
                        self.task_stage = 2
                    #hasn't yet returned to start angle
                    elif self.found_colour == True:
                        self.robot_controller.set_move_cmd(0.0, self.turn_vel_med)
                        self.robot_controller.publish()
                    else:
                        print("finished turning")

                #start at area B(Cyan) For colour cyan, it can correctly return the target colour. But when it's facing the target obstacle, it cannot recognise it. 
                elif current_angle > 92 and current_angle <=181:
                    if self.found_colour == False and angle_difference > 3:
                        self.robot_controller.set_move_cmd(0.0, -(self.turn_vel_med))
                    #if turned and hasn't got start colour yet
                    elif angle_difference <= 3 and self.found_colour == False:
                        
                        self.robot_controller.set_move_cmd(0.0,0.0)
                        self.robot_controller.publish()
                        for colour in self.colour_boundaries:
                            if (self.centre_pixel_colours > colour[0]) and (self.centre_pixel_colours < colour[1]):
                                print("SEARCH INITIATED: The target colour is: " + colour[2])
                                self.search_colour = colour[2]
                                self.found_colour=True

                    
                    elif (angle_difference > 88 or angle_difference>350) and self.found_colour == True:
                        self.robot_controller.set_move_cmd(0.0,0.0)
                        self.robot_controller.publish()
                        #move onto next stage
                        self.task_stage = 2
                    #hasn't yet returned to start angle
                    elif self.found_colour == True:
                        self.robot_controller.set_move_cmd(0.0, self.turn_vel_med)
                        self.robot_controller.publish()
                    else:
                        print("finished turning")

                #start at areaC(Blue) The way I think can improve it is make it turns about 75 or 90 degrees when it's going to make its first turn, and that should roughly facing towards to the blue target
                #obstacle, I've already made it go straight to the target obstacle when it detect the exact colour. 
                elif current_angle >= 180:
                    #print(angle_difference)
                    if self.found_colour == False and angle_difference <= 260:
                        self.robot_controller.set_move_cmd(0.0, self.turn_vel_med)
                    #if turned and hasn't got start colour yet
                    elif angle_difference > 261 and self.found_colour == False:
                        
                        self.robot_controller.set_move_cmd(0.0,0.0)
                        self.robot_controller.publish()
                        for colour in self.colour_boundaries:
                            if (self.centre_pixel_colours > colour[0]) and (self.centre_pixel_colours < colour[1]):
                                print("SEARCH INITIATED: The target colour is: " + colour[2])
                                self.search_colour = colour[2]
                                self.found_colour=True

                    
                    elif (angle_difference <= 179) and self.found_colour == True:
                        print("finished turning")
                        self.robot_controller.set_move_cmd(0.0,0.0)
                        self.robot_controller.publish()
                        #move onto next stage
                        self.task_stage = 2
                    #hasn't yet returned to start angle
                    elif self.found_colour == True:
                        self.robot_controller.set_move_cmd(0.0, -self.turn_vel_med)
                        self.robot_controller.publish()
                    else:
                        print("finished turning")

            #explore the arena
            if self.task_stage == 2:
                #exploration/obstacle avoidance code here
                #if not turning fast enough and about to hit a wall

                if self.m00 > self.m00_min:
                    # blob detected
                    if self.cy >= 560-100 and self.cy <= 560+100:
                        if self.move_rate == 'slow':
                            self.robot_controller.set_move_cmd(0.0, self.turn_vel_slow)
                            self.found_beacon = True
                            print("BEACON DETECTED: Beaconing initiated.")
                
                if self.min_distance <= 0.4 and self.found_beacon == False:
                    #stop moving and keep turning until the coast is clear
                    self.robot_controller.set_move_cmd(0, self.turning_direction * 1.2)
                  
                #if approaching an obstacle
                elif self.min_distance <= 0.55 and self.found_beacon == False:

                    #if the object is on the left or is directly in front
                    if self.object_angle<=0:
                        #update current turning direction
                        self.turning_direction = -1
                        #turn left at speed inversely proportional to proximity to obstacle
                        self.robot_controller.set_move_cmd(goal.fwd_velocity, -0.8*(1/self.min_distance))
                    
                    #if the object is on the right
                    else:
                        #update turning direction
                        self.turning_direction = 1
                        #turn right at speed inversely proportional to proximity to obstacle
                        self.robot_controller.set_move_cmd(goal.fwd_velocity, 0.8*(1/self.min_distance))

                elif self.found_beacon == True:
                    if self.min_distance > 0.25:
                        self.robot_controller.set_move_cmd(0.22, 0)
                    else:
                        self.robot_controller.set_move_cmd(0.0, 0)
                        print("BEACONING COMPLETE: The robot has now stopped.")
                        self.robot_controller.stop()


                
                else:
                    #otherwise, no objects in the way, so set speed to normal forward velocity
                    self.robot_controller.set_move_cmd(goal.fwd_velocity, 0.0)
                    self.robot_controller.set_move_cmd(0.22, 0.0)
                    self.robot_controller.publish()
                    
            #try to find colour, and turn to face it if dound     
            self.robot_controller.publish()

            # check if there has been a request to cancel the action mid-way through:
            if self.actionserver.is_preempt_requested():
                rospy.loginfo("Cancelling the beacon search.")
                
                self.actionserver.set_preempted()
                # stop the robot:
                self.robot_controller.stop()
                success = False
                # exit the loop:
                break
            else:
                #calculate distance from origin/start position
                self.distance = sqrt(pow(self.posx0 - self.robot_odom.posx, 2) + pow(self.posy0 - self.robot_odom.posy, 2))
                # populate the feedback message and publish it:
                self.feedback.current_distance_travelled = self.distance
                self.actionserver.publish_feedback(self.feedback)

        #if robot has reached the goal pillar, end the task
        if success:
            rospy.loginfo("approach completed sucessfully, beacon found.")
            self.result.total_distance_travelled = self.distance
            self.result.closest_object_distance = self.min_distance
            self.result.closest_object_angle = self.object_angle

            self.actionserver.set_succeeded(self.result)
            self.robot_controller.stop()
            
if __name__ == '__main__':
    rospy.init_node("beacon_find_server")
    BeaconFindServer()
    rospy.spin()
