"""Regression checks for camera, physical-contact and policy boundaries."""
from pathlib import Path
from dataclasses import fields
import sys,unittest
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'training/lewm_official')]
from small_part.tube_scene import load_tube_model,LEGACY_PHYSICS,hole_xy,TARGET_ROW,TUBE_CENTER_Z
from small_part.tube_evaluation import TubeEvaluator
from visual_mpc import Observation

class ContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.legacy,_=load_tube_model(LEGACY_PHYSICS);cls.model,_=load_tube_model()
 def test_cameras_including_mount_chain(self):
  for name in ('overhead','wrist'):
   before=self.legacy.camera(name).id;after=self.model.camera(name).id
   for attr in ('cam_pos','cam_quat','cam_fovy'):
    np.testing.assert_array_equal(getattr(self.legacy,attr)[before],getattr(self.model,attr)[after])
   a=int(self.legacy.cam_bodyid[before]);b=int(self.model.cam_bodyid[after])
   while a or b:
    self.assertEqual(self.legacy.body(a).name,self.model.body(b).name)
    np.testing.assert_array_equal(self.legacy.body_pos[a],self.model.body_pos[b]);np.testing.assert_array_equal(self.legacy.body_quat[a],self.model.body_quat[b])
    a=int(self.legacy.body_parentid[a]);b=int(self.model.body_parentid[b])
 def test_contacts_and_no_attachments(self):
  m=self.model;self.assertEqual(m.neq,0);self.assertEqual(m.nmocap,0)
  np.testing.assert_array_equal(m.actuator_forcerange,np.tile([-3.35,3.35],(6,1)))
  for f in range(2,m.nflex):
   for name in ('tube_wall','tube_bottom','table','source_upper_wall_1_0','target_upper_wall_1_0'):
    g=m.geom(name).id
    self.assertTrue((m.flex_contype[f]&m.geom_conaffinity[g]) or (m.geom_contype[g]&m.flex_conaffinity[f]))
 def test_policy_has_no_object_state_input(self):
  self.assertEqual([f.name for f in fields(Observation)],['pixels','joints','previous_action'])
  source=(ROOT/'training/lewm_official/visual_mpc.py').read_text()
  self.assertNotIn('import mujoco',source);self.assertNotIn('tube_position(',source)
 def test_already_seated_without_transfer_does_not_pass(self):
  m=self.model;d=mujoco.MjData(m);mujoco.mj_resetData(m,d)
  target=hole_xy(TARGET_ROW,1);q=int(m.jnt_qposadr[m.joint('tube_free').id])
  d.qpos[q:q+7]=[*target,TUBE_CENTER_Z,1,0,0,0];mujoco.mj_forward(m,d)
  evaluator=TubeEvaluator(m,target);evaluator.update(m,d)
  result=evaluator.result(m,d);self.assertFalse(result['success']);self.assertFalse(result['lifted'])
if __name__=='__main__':unittest.main()
