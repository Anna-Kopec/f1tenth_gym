"""!
@package f110_gym.unittest
@brief Development scripts: early standalone prototypes, legacy test-data generation,
       rendering experiments and a random track generator.

Nothing in the gym imports these modules. The maintained code lives in f110_gym.envs.
- collision_checks.py, dynamics_test.py, scan_sim.py: older prototypes of
  collision_models.py, dynamic_models.py and laser_models.py.
- legacy_scan_gen.py: generated legacy_scan.npz using the old C++ gym.
- pyglet_test.py, pyglet_test_camera.py: rendering and camera experiments.
- random_trackgen.py: generates random race tracks (map image + YAML + centreline).

@note The import below refers to @c gym.envs.unittest.scan_sim, which does not exist
      (gym's own @c gym.envs.unittest package has no scan_sim), so
      @c import f110_gym.unittest raises ModuleNotFoundError. Run the scripts directly instead.
"""
from gym.envs.unittest.scan_sim import *
