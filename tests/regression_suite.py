import os
import sys
import unittest
import ctypes as ct
import numpy as np

# Insert the marl_python directory into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../marl_python")))

from env_wrapper import SatelliteEnv, StatePacket, ActionPacket
from observation import ObservationBuilder
from config import EnvConfig, ObsConfig

class TestABI(unittest.TestCase):
    def test_statepacket_size(self):
        """Verify that Python ctypes StatePacket matches version 3 size of 222 bytes."""
        self.assertEqual(ct.sizeof(StatePacket), 222)

    def test_actionpacket_size(self):
        """Verify that Python ctypes ActionPacket matches version 1 size of 20 bytes."""
        self.assertEqual(ct.sizeof(ActionPacket), 20)


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.builder = ObservationBuilder()

    def test_obs_dim(self):
        """Verify that the S-MAS Phase A observation vector has exactly 42 dimensions."""
        self.assertEqual(self.builder.obs_dim, 42)

    def test_normalization_ranges(self):
        """Verify that normalized observation values are properly bounded and mapped."""
        cfg = EnvConfig()
        env = SatelliteEnv(cfg)
        state = env.reset()
        
        obs = self.builder.build(state)
        
        # Check that all features are floats and have no NaNs/Infs
        for idx, val in enumerate(obs):
            self.assertTrue(np.isfinite(val), f"Observation index {idx} contains non-finite value: {val}")
            
        # Verify specific bound constraints
        # Altitude is MinMax scaled between 200 and 800km -> must be in [0, 1]
        self.assertTrue(0.0 <= obs[0] <= 1.0, f"Normalized altitude out of bounds: {obs[0]}")
        # Battery SoC is already in [0, 1]
        self.assertTrue(0.0 <= obs[7] <= 1.0, f"Normalized SoC out of bounds: {obs[7]}")
        # Fuel Fraction is in [0, 1]
        self.assertTrue(0.0 <= obs[26] <= 1.0, f"Normalized Fuel Fraction out of bounds: {obs[26]}")
        
        env.close()


class TestPhysics(unittest.TestCase):
    def test_fuel_conservation(self):
        """Verify fuel consumption follows Tsiolkovsky mass dynamics when thrusters are active."""
        cfg = EnvConfig()
        env = SatelliteEnv(cfg)
        state = env.reset()
        
        initial_fuel = float(state.fuel_fraction)
        self.assertAlmostEqual(initial_fuel, 1.0, places=4, msg="Initial fuel fraction should be 1.0")

        # 1. No-thrust check: stepping with zero thrust should NOT consume fuel
        action_no_thrust = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 0}
        state, _, _, _ = env.step(action_no_thrust)
        self.assertAlmostEqual(state.fuel_fraction, initial_fuel, places=5, 
                               msg="Passive stepping consumed fuel unexpectedly!")

        # 2. Thrust check: active thrust must consume fuel (step 10 times to flush actuator delay queue)
        action_thrust = {"nav": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32), "bus": 0, "mission": 0}
        for _ in range(10):
            state, _, _, _ = env.step(action_thrust)
        
        # Fuel fraction must decrease
        self.assertTrue(state.fuel_fraction < initial_fuel, 
                        f"Active thrust did not consume propellant! Fuel remaining: {state.fuel_fraction}")
        
        env.close()

    def test_thermal_equilibrium(self):
        """Verify that temperature nodes remain bounded and behave thermodynamically."""
        cfg = EnvConfig()
        env = SatelliteEnv(cfg)
        state = env.reset()
        
        # Start at nominal temps
        self.assertTrue(-40.0 <= state.temp_bus <= 60.0)
        self.assertTrue(-40.0 <= state.temp_battery <= 60.0)
        self.assertTrue(-40.0 <= state.temp_payload <= 60.0)

        # Run 500 steps (approx 41 mins) in a sunlight-only scenario to verify thermal balance
        env._lib.smas_set_time(env._handle, ct.c_double(86400.0 * 172.0))
        
        step = 0
        done = False
        while step < 500 and not done:
            action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 0}
            state, _, done, _ = env.step(action)
            step += 1
            
        # Temperatures must stay within reasonable thermodynamic operational envelopes
        self.assertTrue(-35.0 <= state.temp_bus <= 55.0, f"Bus temp out of physics envelope: {state.temp_bus}°C")
        self.assertTrue(-30.0 <= state.temp_battery <= 50.0, f"Battery temp out of physics envelope: {state.temp_battery}°C")
        self.assertTrue(-35.0 <= state.temp_payload <= 45.0, f"Payload temp out of physics envelope: {state.temp_payload}°C")
        
        env.close()

    def test_orbital_decay(self):
        """Verify that orbital altitude decay rates are physically realistic and bounded."""
        cfg = EnvConfig()
        cfg.density_multiplier = 0.01
        env = SatelliteEnv(cfg)
        state = env.reset()
        
        initial_alt = state.altitude_km
        
        # Propagate for 5 orbits (approx 8.1 hours) passively
        step = 0
        done = False
        while step < 1176 * 5 and not done:
            action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 0}
            state, _, done, _ = env.step(action)
            step += 1
            
        self.assertFalse(done, f"Satellite died prematurely during decay test. Reason: {state.done_reason}")
        
        # Bounded decay check. Altitude oscillates between 551km and 580km due to e=0.002.
        # We ensure it stays in the stable physical SSO band [-35km, +35km] from initial alt.
        self.assertTrue(abs(state.altitude_km - initial_alt) < 35.0, 
                        f"Orbital propagation is unstable! Alt went from {initial_alt} to {state.altitude_km}")
        
        env.close()


if __name__ == "__main__":
    unittest.main()
