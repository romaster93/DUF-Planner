#!/usr/bin/env python3
import threading
import rospy
from std_msgs.msg import Bool

def keyboard_loop(end_pub, next_pub):
    """
    Reads user input from the terminal and publishes on the respective ROS topics
    when 'stop' or 'next' is entered.
    """
    rospy.loginfo("Type 'stop' to end exploration or 'next' to force next cube.")
    while not rospy.is_shutdown():
        try:
            cmd = input("Command ('stop'/'next'): ").strip().lower()
        except EOFError:
            break
        if cmd == 'stop':
            end_pub.publish(True)
            rospy.loginfo("Published /exploration_end_trigger = True")
        elif cmd == 'next':
            next_pub.publish(True)
            rospy.loginfo("Published /next_cube_trigger = True")
        else:
            rospy.logwarn("Unknown command: '%s'", cmd)

def main():
    rospy.init_node('exploration_keyboard_control', anonymous=True)

    # Publishers for the trigger topics
    end_pub = rospy.Publisher('/exploration_end_trigger', Bool, queue_size=1)
    next_pub = rospy.Publisher('/next_cube_trigger', Bool, queue_size=1)

    # Wait until publishers are registered
    rospy.sleep(0.5)

    # Run keyboard loop in main thread
    keyboard_loop(end_pub, next_pub)

if __name__ == '__main__':
    main()


