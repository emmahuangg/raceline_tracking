import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Computes low-level inputs (steering rate and acceleration).
    """
    # Extract state: [x, y, delta, v, phi]
    current_steering_angle = state[2]
    current_velocity = state[3]

    desired_steering_angle = desired[0]
    desired_velocity = desired[1]

    # --- Lateral Control (Steering) ---
    # High gain to ensure the wheels snap to the desired angle quickly.
    # If this is too low, the car lags behind the command and runs wide.
    K_P_steer = 4.0  
    steering_error = desired_steering_angle - current_steering_angle
    steering_rate = K_P_steer * steering_error

    # --- Longitudinal Control (Acceleration) ---
    # We want crisp acceleration and braking.
    K_P_accel = 2.0 
    velocity_error = desired_velocity - current_velocity
    acceleration = K_P_accel * velocity_error

    return np.array([steering_rate, acceleration])


def find_target_point(car_position, centerline, lookahead_indices):
    """
    Finds the point on the track to aim for.
    """
    distances = np.linalg.norm(centerline - car_position, axis=1)
    closest_idx = np.argmin(distances)

    # Wrap around the track indices
    target_idx = (closest_idx + int(lookahead_indices)) % centerline.shape[0]
    return centerline[target_idx]


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    Controller tracking the centerline tightly using Velocity-Based Lookahead.
    """
    
    # 1. Extract Car State
    car_position = state[0:2]
    current_heading = state[4]
    current_velocity = max(state[3], 1.0) # Avoid division by zero
    wheelbase = parameters[0] 

    # --- 1. VELOCITY-BASED LOOKAHEAD ---
    # This is the key to tight tracking.
    # Rule: Lookahead Index = Gain * Velocity
    # At 20 m/s -> Look ~12 points ahead (Stable)
    # At 8 m/s  -> Look ~5 points ahead (Tight cornering)
    LOOKAHEAD_GAIN = 0.3
    
    lookahead_idx = LOOKAHEAD_GAIN * current_velocity
    
    # Clamp the lookahead to sane values
    # Minimum 4 prevents aiming at the car's own bumper (instability)
    # Maximum 20 prevents cutting straight across massive curves
    lookahead_idx = np.clip(lookahead_idx, 4, 20)

    # --- 2. COMPUTE HEADING ERROR ---
    target_point = find_target_point(car_position, racetrack.centerline, lookahead_idx)

    delta_x = target_point[0] - car_position[0]
    delta_y = target_point[1] - car_position[1]
    
    # The angle we WANT to be facing
    desired_heading = np.arctan2(delta_y, delta_x)
    
    # The error between current heading and desired heading
    delta_phi = desired_heading - current_heading

    # Wrap error to [-pi, pi] to handle the 360-0 degree crossover
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))

    # --- 3. LATERAL CONTROL LAW ---
    # We use a "Time Constant" approach. 
    # We want to correct the error delta_phi in LOOKAHEAD_TIME seconds.
    # Smaller time = Tighter tracking (more aggressive)
    # Larger time = Smoother tracking (less oscillation)
    LOOKAHEAD_TIME = 0.4 
    
    dot_phi_desired = delta_phi / LOOKAHEAD_TIME

    # Kinematic Bicycle Model: delta = arctan( L * phi_dot / v )
    desired_steering_angle = np.arctan((wheelbase * dot_phi_desired) / current_velocity)

    # Clip to physical limits
    desired_steering_angle = np.clip(desired_steering_angle, -0.9, 0.9)

    # --- 4. PREDICTIVE SPEED CONTROL ---
    # Slow down based on the SHARPNESS of the turn we are approaching.
    # We use 'delta_phi' (heading error) to decide. 
    # Large heading error = Sharp turn incoming = BRAKE.
    
    MAX_SPEED = 22.0
    MIN_SPEED = 8.0
    
    # If heading error is > 0.3 rad (~17 degrees), we start braking hard.
    error_scaling = np.clip(np.abs(delta_phi) / 0.5, 0.0, 1.0)
    
    desired_velocity = MAX_SPEED - (error_scaling * (MAX_SPEED - MIN_SPEED))

    return np.array([desired_steering_angle, desired_velocity])