"""Regression: approaching or retreating a jaw must not move an unheld tube prior."""
import unittest
import numpy as np
from small_part.visual_transfer import VisualObservation,VisualTransfer

class Robot:
    tip=np.array([.466,.228,.300])
    def pose(self,q):return self.tip.copy(),np.eye(3),(None,None)
    def solve(self,goal,q,axis):return q[:5].copy()

class Tracker:
    center=np.array([.466,.228,.26])
    def estimate(self,images,cameras,**kwargs):
        self.prior=kwargs['predicted_center'].copy()
        return self.center.copy(),np.array([0.,0.,1.]),{'views':[0]}

class PredictionRegression(unittest.TestCase):
    def test_unheld_and_held_motion_priors(self):
        for stage in ('hover','descend','grasp','release','retreat','settle','lift','carry','insert'):
            with self.subTest(stage=stage):
                robot=Robot();policy=VisualTransfer(robot);tracker=Tracker();policy.tracker=tracker
                policy.stage=stage;policy.source=tracker.center[:2].copy();policy.target=np.array([.466,-.04])
                policy.last_center=tracker.center.copy();policy.last_axis=np.array([0.,0.,1.])
                delta=np.array([.001,0.,.002]);policy.last_tip=robot.tip-delta
                policy.act(VisualObservation((None,None),np.zeros(6)))
                expected=tracker.center+delta if stage in ('lift','carry','insert') else tracker.center
                np.testing.assert_array_equal(tracker.prior,expected)

if __name__=='__main__':unittest.main()
