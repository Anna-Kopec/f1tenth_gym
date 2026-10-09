# Copyright 2020 Technical University of Munich, Professorship of Cyber-Physical Systems, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""!
@file dynamic_models.py
@brief Vehicle dynamics for the F1TENTH gym: single-track (bicycle) models,
       input constraints, and the low-level speed / steering controller.

Follows the CommonRoad single-track vehicle models.
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/

@section dynamic_models_models Models in this file
- **Kinematic single-track (KS)**, vehicle_dynamics_ks(): a 5-state bicycle model
  with no tyre slip. The wheels go exactly where they point. Accurate at low
  speed, wrong when cornering hard.
- **Dynamic single-track (ST)**, vehicle_dynamics_st(): a 7-state bicycle model with
  a linear tyre model, slip angle and yaw dynamics, including load transfer from
  longitudinal acceleration. Below 0.5 m/s it switches to the KS equations,
  because the ST equations divide by the speed.

Both functions return the state *derivative* f(x, u). They do not step the
state themselves; an integrator (Euler or RK4 in base_classes.RaceCar.update_pose)
does that.

@section dynamic_models_inputs Control inputs
The models take u = [steering angle velocity, longitudinal acceleration].
The gym's users command a *desired steering angle and speed* instead, so pid()
converts between the two. Every model call first passes u through
steering_constraint() and accl_constraints() to enforce the physical limits.

@section dynamic_models_params Vehicle parameters (shared by every function)
| Name      | Meaning                                                         | Units    |
|-----------|-----------------------------------------------------------------|----------|
| mu        | tyre-road friction coefficient                                  | -        |
| C_Sf      | front cornering stiffness coefficient (per unit of normal load) | 1/rad    |
| C_Sr      | rear cornering stiffness coefficient                            | 1/rad    |
| lf, lr    | distance from centre of gravity to front / rear axle            | m        |
| h         | height of the centre of gravity                                 | m        |
| m         | vehicle mass                                                    | kg       |
| I         | yaw moment of inertia about the z axis                          | kg m^2   |
| s_min/max | steering angle limits                                           | rad      |
| sv_min/max| steering angle velocity limits                                  | rad/s    |
| v_switch  | speed above which acceleration is limited by motor power        | m/s      |
| a_max     | maximum longitudinal acceleration magnitude                     | m/s^2    |
| v_min/max | speed limits (v_min is negative, for reversing)                 | m/s      |

@author Hongrui Zheng
"""

import numpy as np
from numba import njit

import unittest
import time

@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """!
    @brief Clamp a requested longitudinal acceleration to what the vehicle can do.

    Three limits are applied:
    - **Power limit:** above @p v_switch, the maximum forward acceleration falls
      off as a_max * v_switch / vel. This models a motor with constant power,
      where force (and so acceleration) is proportional to 1/speed.
    - **Speed limits:** once the car is at @p v_min it cannot accelerate further
      negative, and once at @p v_max it cannot accelerate further positive.
    - **Braking limit:** deceleration is capped at -a_max.

    @param vel      Current longitudinal velocity (m/s).
    @param accl     Requested (unconstrained) acceleration (m/s^2).
    @param v_switch Switching velocity: above it, acceleration is power-limited (m/s).
    @param a_max    Maximum acceleration magnitude (m/s^2).
    @param v_min    Minimum allowed velocity (m/s, negative for reverse).
    @param v_max    Maximum allowed velocity (m/s).
    @return The constrained acceleration (m/s^2).
    """

    # positive accl limit (power-limited above v_switch)
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        # already at a speed limit and pushing past it: no acceleration
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl

@njit(cache=True)
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """!
    @brief Clamp a requested steering angle velocity to the steering limits.

    - If the wheels are already at a lock (@p s_min or @p s_max) and the request
      would turn them further, the steering velocity is set to 0.
    - Otherwise the rate is clamped to [@p sv_min, @p sv_max], which models the
      top speed of the steering servo.

    @param steering_angle    Current front-wheel steering angle (rad).
    @param steering_velocity Requested (unconstrained) steering angle velocity (rad/s).
    @param s_min             Minimum steering angle (rad).
    @param s_max             Maximum steering angle (rad).
    @param sv_min            Minimum steering velocity (rad/s).
    @param sv_max            Maximum steering velocity (rad/s).
    @return The constrained steering velocity (rad/s).
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        # at full lock and still turning further: hold the angle
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity


@njit(cache=True)
def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """!
    @brief Kinematic single-track (bicycle) model: state derivative f(x, u).

    It assumes the tyres roll without slipping, so the rear axle always moves
    along the car's heading and the yaw rate is set purely by geometry:

        x_dot     = v * cos(psi)
        y_dot     = v * sin(psi)
        delta_dot = u[0]  (steering velocity)
        v_dot     = u[1]  (acceleration)
        psi_dot   = v / (lf + lr) * tan(delta)

    Here (x, y) is the rear-axle position.

    @param x      np.ndarray of shape (5,): state [x, y, delta, v, psi]
                  - x[0]: x position in global coordinates (m)
                  - x[1]: y position in global coordinates (m)
                  - x[2]: front-wheel steering angle delta (rad)
                  - x[3]: longitudinal velocity v (m/s)
                  - x[4]: yaw angle psi (rad)
    @param u_init np.ndarray of shape (2,): requested input [steering velocity (rad/s),
                  longitudinal acceleration (m/s^2)]. It is clamped by
                  steering_constraint() and accl_constraints() before use.
    @param mu,C_Sf,C_Sr,lf,lr,h,m,I,s_min,s_max,sv_min,sv_max,v_switch,a_max,v_min,v_max
                  Vehicle parameters (see the parameter table in the file documentation).
                  Only lf, lr and the constraint limits are used by this model.
    @return np.ndarray of shape (5,): the state derivative dx/dt.
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])
    return f

@njit(cache=True)
def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """!
    @brief Dynamic single-track (bicycle) model with tyre slip: state derivative f(x, u).

    **Above 0.5 m/s** (the dynamic model):
    - Position moves along the *velocity* direction, psi + beta, not the heading
      psi. The car can drift sideways.
    - Each axle's lateral tyre force is linear in its slip angle. The force is
      scaled by mu, by that axle's cornering stiffness (C_Sf or C_Sr) and by the
      axle's normal load.
    - Normal loads include longitudinal load transfer. The front axle carries
      m*(g*lr - a*h)/(lf + lr) and the rear axle m*(g*lf + a*h)/(lf + lr), so
      accelerating (a > 0) shifts load to the rear and braking shifts it to the front.
    - The yaw acceleration (x[5] derivative) and slip angle rate (x[6] derivative)
      come from the lateral force and moment balance on these tyre forces.

    **Below 0.5 m/s** (kinematic fallback): the dynamic equations divide by v and
    v^2, so they blow up near standstill. The KS model is used instead for the
    first five states, and:
    - the yaw rate derivative is the time derivative of the KS yaw rate,
      d/dt [ v / lwb * tan(delta) ] = a/lwb * tan(delta) + v/(lwb * cos^2(delta)) * delta_dot.
    - the slip angle derivative is 0, so the slip angle is held constant.

    @param x      np.ndarray of shape (7,): state [x, y, delta, v, psi, psi_dot, beta]
                  - x[0]: x position in global coordinates (m)
                  - x[1]: y position in global coordinates (m)
                  - x[2]: front-wheel steering angle delta (rad)
                  - x[3]: longitudinal velocity v (m/s)
                  - x[4]: yaw angle psi (rad)
                  - x[5]: yaw rate psi_dot (rad/s)
                  - x[6]: slip angle beta at the centre of gravity (rad)
    @param u_init np.ndarray of shape (2,): requested input [steering velocity (rad/s),
                  longitudinal acceleration (m/s^2)]. It is clamped before use.
    @param mu,C_Sf,C_Sr,lf,lr,h,m,I,s_min,s_max,sv_min,sv_max,v_switch,a_max,v_min,v_max
                  Vehicle parameters (see the parameter table in the file documentation).
    @return np.ndarray of shape (7,): the state derivative dx/dt.

    @note (x, y) is the centre of gravity in the dynamic branch, but the KS
          fallback treats it as the rear axle. The difference is ignored when
          switching between them.
    """

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:
        # wheelbase
        lwb = lf + lr

        # system dynamics: KS for the first 5 states, then
        # yaw-rate derivative = d/dt of the KS yaw rate, and slip angle held constant
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))

    else:
        # system dynamics
        # Terms (g*lr - u[1]*h) and (g*lf + u[1]*h) are the front / rear normal
        # loads (per unit mass, times wheelbase) including longitudinal load transfer.
        f = np.array([
            # x_dot, y_dot: move along the velocity direction (heading + slip)
            x[3]*np.cos(x[6] + x[4]),
            x[3]*np.sin(x[6] + x[4]),
            # delta_dot, v_dot: the constrained inputs
            u[0],
            u[1],
            # psi_dot: yaw rate is itself a state
            x[5],
            # psi_ddot: yaw moment from front and rear lateral tyre forces / I.
            # The three terms are the contributions of yaw rate, slip angle and steering angle.
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            # beta_dot: lateral force balance, with the same three contributions
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f

@njit(cache=True)
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """!
    @brief Low-level controller: desired (speed, steering angle) -> (acceleration, steering velocity).

    This is what lets the gym take "go this fast, steer this much" as its action.

    **Steering (bang-bang):** if the steering angle is off target by more than
    1e-4 rad, steer toward the target at full rate (+/- @p max_sv). Otherwise
    hold still. The angle can overshoot by up to max_sv * dt in one step.

    **Speed (proportional only, no integral or derivative term):**
    acceleration = kp * (target speed - current speed). The gain kp depends on
    the situation:

    | Current speed     | Speed error      | Situation       | kp                      |
    |-------------------|------------------|-----------------|-------------------------|
    | forward (v > 0)   | positive         | speeding up     | 10 * max_a / max_v      |
    | forward (v > 0)   | zero or negative | braking         | 10 * max_a / (-min_v)   |
    | v <= 0            | positive         | braking         | 2 * max_a / max_v       |
    | v <= 0            | zero or negative | speeding up     | 2 * max_a / (-min_v)    |

    A car at a standstill (v == 0) uses the v <= 0 rows, so pulling away from
    rest uses the gentler gain of 2 rather than 10.

    Outputs are not clamped here. accl_constraints() and steering_constraint()
    clamp them inside the dynamics model.

    @param speed         Desired longitudinal speed (m/s).
    @param steer         Desired steering angle (rad).
    @param current_speed Current longitudinal speed (m/s).
    @param current_steer Current steering angle (rad).
    @param max_sv        Maximum steering velocity (rad/s).
    @param max_a         Maximum acceleration (m/s^2).
    @param max_v         Maximum speed (m/s).
    @param min_v         Minimum speed (m/s, negative).
    @return Tuple (accl, sv): acceleration command (m/s^2) and steering velocity command (rad/s).
    """
    # steering: full-rate toward the target, or hold if close enough
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    # accl: proportional on speed error, gain chosen by direction of travel
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.:
        if (vel_diff > 0):
            # accelerate
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if (vel_diff > 0):
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv

def func_KS(x, t, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """!
    @brief Adapter for scipy.integrate.odeint: wraps vehicle_dynamics_ks() with the (x, t, ...) signature odeint expects.

    @param x State, as in vehicle_dynamics_ks().
    @param t Time (unused; the model does not depend on time).
    @param u Input, as in vehicle_dynamics_ks().
    @param mu,C_Sf,C_Sr,lf,lr,h,m,I,s_min,s_max,sv_min,sv_max,v_switch,a_max,v_min,v_max
             Vehicle parameters, passed through unchanged.
    @return The state derivative from vehicle_dynamics_ks().
    """
    f = vehicle_dynamics_ks(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
    return f

def func_ST(x, t, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """!
    @brief Adapter for scipy.integrate.odeint: wraps vehicle_dynamics_st() with the (x, t, ...) signature odeint expects.

    @param x State, as in vehicle_dynamics_st().
    @param t Time (unused; the model does not depend on time).
    @param u Input, as in vehicle_dynamics_st().
    @param mu,C_Sf,C_Sr,lf,lr,h,m,I,s_min,s_max,sv_min,sv_max,v_switch,a_max,v_min,v_max
             Vehicle parameters, passed through unchanged.
    @return The state derivative from vehicle_dynamics_st().
    """
    f = vehicle_dynamics_st(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
    return f

class DynamicsTest(unittest.TestCase):
    """!
    @brief Unit tests comparing the KS and ST models against CommonRoad reference values.

    The test vehicle is the full-size CommonRoad reference car (converted from
    imperial units), not the F1TENTH car, so the parameters differ greatly from
    the gym defaults.

    Run with: @c python dynamic_models.py
    """

    def setUp(self):
        """!
        @brief Set up the full-size test vehicle's parameters.
        """
        # test params
        self.mu = 1.0489
        self.C_Sf = 21.92/1.0489
        self.C_Sr = 21.92/1.0489
        self.lf = 0.3048*3.793293
        self.lr = 0.3048*4.667707
        self.h = 0.3048*2.01355
        self.m = 4.4482216152605/0.3048*74.91452
        self.I = 4.4482216152605*0.3048*1321.416

        #steering constraints
        self.s_min = -1.066  #minimum steering angle [rad]
        self.s_max = 1.066  #maximum steering angle [rad]
        self.sv_min = -0.4  #minimum steering velocity [rad/s]
        self.sv_max = 0.4  #maximum steering velocity [rad/s]

        #longitudinal constraints
        self.v_min = -13.6  #minimum velocity [m/s]
        self.v_max = 50.8  #minimum velocity [m/s]
        self.v_switch = 7.319  #switching velocity [m/s]
        self.a_max = 11.5  #maximum absolute acceleration [m/s^2]

    def test_derivatives(self):
        """!
        @brief Single derivative evaluations match the reference values, and the ST
               model runs faster than 5000 evaluations per second.
        """
        # ground truth derivatives
        f_ks_gt = [16.3475935934250209, 0.4819314886013121, 0.1500000000000000, 5.1464424102339752, 0.2401426578627629]
        f_st_gt = [15.7213512030862397, 0.0925527979719355, 0.1500000000000000, 5.3536773276413925, 0.0529001056654038, 0.6435589397748606, 0.0313297971641291]

        # system dynamics
        g = 9.81
        x_ks = np.array([3.9579422297936526, 0.0391650102771405, 0.0378491427211811, 16.3546957860883566, 0.0294717351052816])
        x_st = np.array([2.0233348142065677, 0.0041907137716636, 0.0197545248559617, 15.7216236334290116, 0.0025857914776859, 0.0529001056654038, 0.0033012170610298])
        v_delta = 0.15
        acc = 0.63*g
        u = np.array([v_delta,  acc])

        f_ks = vehicle_dynamics_ks(x_ks, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        f_st = vehicle_dynamics_st(x_st, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)

        start = time.time()
        for i in range(10000):
            f_st = vehicle_dynamics_st(x_st, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        duration = time.time() - start
        avg_fps = 10000/duration

        self.assertAlmostEqual(np.max(np.abs(f_ks_gt-f_ks)), 0.)
        self.assertAlmostEqual(np.max(np.abs(f_st_gt-f_st)), 0.)
        self.assertGreater(avg_fps, 5000)

    def test_zeroinit_roll(self):
        """!
        @brief From rest with zero input, both models stay exactly at rest (no singularity at v = 0).
        """
        from scipy.integrate import odeint

        # testing for zero initial state, zero input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set input: rolling car (velocity should stay constant)
        u = np.array([0., 0.])

        # simulate single-track model
        x_roll_st = odeint(func_ST, x0_ST, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_roll_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        self.assertTrue(all(x_roll_st[-1]==x0_ST))
        self.assertTrue(all(x_roll_ks[-1]==x0_KS))

    def test_zeroinit_dec(self):
        """!
        @brief From rest with a braking input, the car reverses in a straight line, matching the reference after 1 s.
        """
        from scipy.integrate import odeint

        # testing for zero initial state, decelerating input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0., -0.7*g])

        # simulate single-track model
        x_dec_st = odeint(func_ST, x0_ST, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_dec_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for single-track model
        x_dec_st_gt = [-3.4335000000000013, 0.0000000000000000, 0.0000000000000000, -6.8670000000000018, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        # ground truth for kinematic single-track model
        x_dec_ks_gt = [-3.4335000000000013, 0.0000000000000000, 0.0000000000000000, -6.8670000000000018, 0.0000000000000000]

        self.assertTrue(all(abs(x_dec_st[-1] - x_dec_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_dec_ks[-1] - x_dec_ks_gt) < 1e-2))

    def test_zeroinit_acc(self):
        """!
        @brief From rest, accelerating while steering left matches the reference trajectory after 1 s.

        The ST model passes through the low-speed KS fallback into the dynamic
        regime, so this also checks that switching between them.
        """
        from scipy.integrate import odeint

        # testing for zero initial state, accelerating with left steer input singularities
        # wheel spin and velocity should increase more wheel spin at rear
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set accelerate + steer-left input
        u = np.array([0.15, 0.63*g])

        # simulate single-track model
        x_acc_st = odeint(func_ST, x0_ST, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_acc_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for single-track model
        x_acc_st_gt = [3.0731976046859715, 0.2869835398304389, 0.1500000000000000, 6.1802999999999999, 0.1097747074946325, 0.3248268063223301, 0.0697547542798040]
        # ground truth for kinematic single-track model
        x_acc_ks_gt = [3.0845676868494927, 0.1484249221523042, 0.1500000000000000, 6.1803000000000017, 0.1203664469224163]

        self.assertTrue(all(abs(x_acc_st[-1] - x_acc_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_acc_ks[-1] - x_acc_ks_gt) < 1e-2))

    def test_zeroinit_rollleft(self):
        """!
        @brief From rest, steering left with no throttle only changes the steering angle; the car does not move.
        """
        from scipy.integrate import odeint

        # testing for zero initial state, rolling and steering left input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set steer-left input, no acceleration
        u = np.array([0.15, 0.])

        # simulate single-track model
        x_left_st = odeint(func_ST, x0_ST, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_left_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for single-track model
        x_left_st_gt = [0.0000000000000000, 0.0000000000000000, 0.1500000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        # ground truth for kinematic single-track model
        x_left_ks_gt = [0.0000000000000000, 0.0000000000000000, 0.1500000000000000, 0.0000000000000000, 0.0000000000000000]

        self.assertTrue(all(abs(x_left_st[-1] - x_left_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_left_ks[-1] - x_left_ks_gt) < 1e-2))

if __name__ == '__main__':
    unittest.main()
