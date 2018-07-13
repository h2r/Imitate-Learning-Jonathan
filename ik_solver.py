import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import itertools


class ArmSolver:

    def __init__(self, limb):
        self.limb = limb
        self.ns = "ExternalTools/" + self.limb + "/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK)
        self.ikreq = SolvePositionIKRequest()
        print "iksvc: ", self.iksvc
        print "ikreq: ", self.ikreq


    def ik_solve(self, pos, orient):
        #~ rospy.init_node("rsdk_ik_service_client")
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        poses = {
            str(self.limb): PoseStamped(header=hdr,
                pose=Pose(position=pos, orientation=orient))}

        self.ikreq.pose_stamp.append(poses[self.limb])
        try:
            rospy.wait_for_service(self.ns, 5.0)
            resp = self.iksvc(self.ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return 1
        finally:
            self.ikreq.pose_stamp[:] = []
        if (resp.isValid[0]):
            print("SUCCESS - Valid Joint Solution Found:")
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            print limb_joints
            return limb_joints
        else:
            print("INVALID POSE - No Valid Joint Solution Found.")

        return -1
    def batch_ik_solve(self, pos, orient):
        #~ rospy.init_node("rsdk_ik_service_client")
        for pos, orient in zip(pos, orient):
            hdr = Header(stamp=rospy.Time.now(), frame_id='base')
            poses = {
                str(self.limb): PoseStamped(header=hdr,
                    pose=Pose(position=pos, orientation=orient))}

            self.ikreq.pose_stamp.append(poses[self.limb])
        try:
            rospy.wait_for_service(self.ns, 5.0)
            resp = self.iksvc(self.ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return 1
        finally:
            self.ikreq.pose_stamp[:] = []
        if (resp.isValid[0]):
            print("SUCCESS - Valid Joint Solution Found:")
            # Format solution into Limb API-compatible dictionary
            a = []
            limb_joints = [dict(zip(joint.name, joint.position)) for joint in resp.joints]
            print limb_joints
            return limb_joints
        else:
            print("INVALID POSE - No Valid Joint Solution Found.")

        return -1

def ik_solve(limb, pos, orient):
    #~ rospy.init_node("rsdk_ik_service_client")
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    print "iksvc: ", iksvc
    print "ikreq: ", ikreq
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        str(limb): PoseStamped(header=hdr,
            pose=Pose(position=pos, orientation=orient))}

    ikreq.pose_stamp.append(poses[limb])
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1
    if (resp.isValid[0]):
        print("SUCCESS - Valid Joint Solution Found:")
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        print limb_joints
        return limb_joints
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")

    return -1