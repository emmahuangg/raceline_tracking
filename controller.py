import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:

    # 1. Extract Current State and Desired References
    current_steering_angle = state[2]  # delta (steering angle)
    current_velocity = state[3]  # v (longitudinal velocity)

    desired_steering_angle = desired[0]  # delta_r
    desired_velocity = desired[1]  # v_r

    # --- Lateral Control (C2: Steering Rate) ---
    # Goal: Drive current_steering_angle -> desired_steering_angle
    K_P_steer = 1.0  # **TUNING REQUIRED**
    steering_error = desired_steering_angle - current_steering_angle

    # Simple P-Control for Steering Rate (v_delta)
    steering_rate = K_P_steer * steering_error

    # --- Longitudinal Control (C1: Acceleration) ---
    # Goal: Drive current_velocity -> desired_velocity
    K_P_accel = 0.5  # **TUNING REQUIRED**
    velocity_error = desired_velocity - current_velocity

    # Simple P-Control for Acceleration (a)
    acceleration = K_P_accel * velocity_error

    # The output order is [v_delta, a]
    return np.array([steering_rate, acceleration])


# Helper function to
def find_target_point(car_position, centerline, lookahead_indices=5):
    # Calculate distance to all centerline points
    distances = np.linalg.norm(centerline - car_position, axis=1)

    # Find the index of the closest point (i)
    closest_idx = np.argmin(distances)

    # The target point is the next point (i + lookahead), wrapping around
    num_points = centerline.shape[0]

    # FIX: Add the lookahead offset here
    target_idx = (closest_idx + lookahead_indices) % num_points

    return centerline[target_idx]


def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:

    # 1. Extract Car State
    car_position = state[0:2]
    current_heading = state[4]
    wheelbase = parameters[0]

    # TUNING PARAMETER 1: Lookahead Index
    # If your track points are dense, increase this (try 5, 10, or 20)
    LOOKAHEAD_IDX = 8

    # TUNING PARAMETER 2: Reference Velocity
    V_REFERENCE = 15.0

    # TUNING PARAMETER 3: Convergence Time
    # Do NOT use the simulation time_step (0.1s).
    # Use a larger value (e.g., 0.5s to 1.0s) to smooth steering.
    LOOKAHEAD_TIME = 0.5

    # --- Lateral Reference ---

    # FIX: Pass the lookahead index
    target_point = find_target_point(car_position, racetrack.centerline, LOOKAHEAD_IDX)

    delta_x = target_point[0] - car_position[0]
    delta_y = target_point[1] - car_position[1]

    desired_heading = np.arctan2(delta_y, delta_x)
    delta_phi = desired_heading - current_heading

    # Correct angle wrapping
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))

    # FIX: Divide by a smoothing time constant, not the physics tick
    dot_phi_approx = delta_phi / LOOKAHEAD_TIME

    # Calculate desired steering
    desired_steering_angle = (wheelbase * dot_phi_approx) / V_REFERENCE

    return np.array([desired_steering_angle, V_REFERENCE])
