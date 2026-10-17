"""Run the native-jaw, pure-contact tube transfer."""
from small_part.tube_cycle_physical import run_tube_cycle

def main():
    import sys,runpy
    if '--viewer' in sys.argv:
        import time,mujoco.viewer
        from small_part.tube_scene import load_tube_model
        m,d=load_tube_model()
        with mujoco.viewer.launch_passive(m,d) as viewer:
            viewer.cam.lookat[:]=(0.40,0.07,0.18)
            viewer.cam.distance=.95;viewer.cam.azimuth=120.;viewer.cam.elevation=-28.
            def show(stage,model,data):
                if not viewer.is_running():raise SystemExit(0)
                viewer.sync();time.sleep(model.opt.timestep)
            print(run_tube_cycle(m,d,on_step=show))
            while viewer.is_running():viewer.sync();time.sleep(.05)
    elif '--no-gif' in sys.argv:
        sys.argv.remove('--no-gif')
        from small_part.tube_cycle_physical import main as validate
        validate()
    else:
        runpy.run_module('small_part.record_tube',run_name='__main__')

if __name__=='__main__':main()
