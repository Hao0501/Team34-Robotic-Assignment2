#! /usr/bin/python

# Import the core Python modules for ROS and to implement ROS Actions:
import rospy
import actionlib

# Import all the necessary ROS message types:
from com2009_actions.msg import SearchFeedback, SearchResult, SearchAction
from sensor_msgs.msg import LaserScan

# Import some other modules from within this package
from move_tb3 import MoveTB3
from tb3_odometry import TB3Odometry

# Import some other useful Python Modules
from math import sqrt, pow, radians
import numpy as np

class SearchActionServer(object):
    feedback = SearchFeedback() 
    result = SearchResult()

    def __init__(self):
        self.actionserver = actionlib.SimpleActionServer("/search_action_server", 
            SearchAction, self.action_server_launcher, auto_start=False)
        self.actionserver.start()

        #subscribe to lidar scan data node
        self.scan_subscriber = rospy.Subscriber("/scan",
            LaserScan, self.scan_callback)

        self.min_distance = 0.5
        self.object_angle = 0
        self.robot_controller = MoveTB3()
        self.robot_odom = TB3Odometry()
        self.arc_angles = np.arange(-50, 51)

        self.start_angle = self.robot_odom.yaw
    
    #lidar scan callback function - update new scan points and vision in front of robot
    def scan_callback(self, scan_data):
        
        left_arc = scan_data.ranges[0:33]
        right_arc = scan_data.ranges[-33:]
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

        #has_returned is used to make the robot move in new directions when it returns to near the starting location
        self.has_returned = False
        #turning_direction is used to let the robot continue turning in the same direction when not turning fast enough close to an obstacle
        self.turning_direction = -1

        #sets an initial value for distance from origin/start point
        self.distance=0

        print("The robot will start to move now...")
        # set the robot velocity:
        self.robot_controller.set_move_cmd(goal.fwd_velocity, 0.0)
        
        #while the robot hasn't collided
        while self.min_distance > goal.approach_distance:
            current_angle = self.robot_odom.yaw 
            #---object avoidance---
            #if not turning fast enough and about to hit a wall
            if self.min_distance <= 0.39:
                #stop moving and keep turning until the coast is clear
                self.robot_controller.set_move_cmd(0, self.turning_direction * 2.2)

            #if approaching an obstacle
            elif self.min_distance <= 0.62:

                #if the object is on the left or is directly in front
                if self.object_angle<=0:
                    #update current turning direction
                    self.turning_direction = -1
                    #turn left at speed inversely proportional to proximity to obstacle
                    self.robot_controller.set_move_cmd(goal.fwd_velocity, -0.29*(1/self.min_distance))
                    
                    #print(current_angle - self.start_angle)
                
                #if the object is on the right
                else:
                    #update turning direction
                    self.turning_direction = 1
                    #turn right at speed inversely proportional to proximity to obstacle
                    self.robot_controller.set_move_cmd(goal.fwd_velocity, 0.28*(1/self.min_distance))
                    #print(current_angle - self.start_angle)

            else:
                #otherwise, no objects in the way, so set speed to normal forward velocity
                self.robot_controller.set_move_cmd(goal.fwd_velocity, 0.0)

            self.robot_controller.publish()

            # check if there has been a request to cancel the action mid-way through:
            if self.actionserver.is_preempt_requested():
                rospy.loginfo("Cancelling the search.")
                
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
            

        #if robot has gone too close to a wall, action is complete
        if success:
            rospy.loginfo("approach completed sucessfully, too close to a wall.")
            self.result.total_distance_travelled = self.distance
            self.result.closest_object_distance = self.min_distance
            self.result.closest_object_angle = self.object_angle

            self.actionserver.set_succeeded(self.result)
            self.robot_controller.stop()
            
if __name__ == '__main__':
    rospy.init_node("search_action_server")
    SearchActionServer()
    rospy.spin()
