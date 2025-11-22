import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# --- 1. LOWER CONTROLLER GAINS ---
K_P_STEER = 4.0
K_P_ACCEL = 5.0

# --- 2. LOOKAHEAD LOGIC (Steering) ---
LOOKAHEAD_GAIN = 0.25
LOOKAHEAD_MIN_IDX = 3
LOOKAHEAD_MAX_IDX = 20

# --- 3. LATERAL CONTROL ---
STEERING_LOOKAHEAD_TIME = 0.4
PHYSICAL_STEERING_LIMIT = 0.9

# --- 4. PREDICTIVE SPEED CONTROL ---
MAX_SPEED = 100.0
MIN_SPEED = 5.0

# BRAKING_LOOKAHEAD
BRAKING_LOOKAHEAD = 150

# LATERAL_ACCEL_LIMIT (Safety Margin)
LATERAL_ACCEL_LIMIT = 9.0

# AVAILABLE_DECEL (Braking Confidence)
AVAILABLE_DECEL = 7.0


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """Computes low-level inputs (steering rate and acceleration)."""
    current_steering_angle = state[2]
    current_velocity = state[3]

    desired_steering_angle = desired[0]
    desired_velocity = desired[1]

    steering_error = desired_steering_angle - current_steering_angle
    steering_rate = K_P_STEER * steering_error

    velocity_error = desired_velocity - current_velocity
    acceleration = K_P_ACCEL * velocity_error

    return np.array([steering_rate, acceleration])


def find_target_point(car_position, centerline, lookahead_indices):
    """
    Finds the target point on the centerline given the car's position and lookahead distance.
    Returns the target point as a 2D numpy array.
    """
    # Calculate distances from car position to all centerline points
    distances = np.linalg.norm(centerline - car_position, axis=1)
    closest_idx = np.argmin(distances)
    target_idx = (closest_idx + int(lookahead_indices)) % centerline.shape[0]

    # Return the closest point on the centerline as the target point
    return centerline[target_idx]


def calculate_curvature_velocity(state, racetrack, parameters):
    """
    Calculates target velocity using distance-based braking logic.
    """
    car_position = state[0:2]
    distances = np.linalg.norm(racetrack.centerline - car_position, axis=1)
    current_idx = np.argmin(distances)
    num_points = racetrack.centerline.shape[0]

    min_safe_velocity = MAX_SPEED

    step = 5

    # for each point in lookahead window, calculate curvature and safe speed
    for i in range(0, BRAKING_LOOKAHEAD, step):
        # Retrieve three points that are 'step' apart
        idx1 = (current_idx + i) % num_points
        idx2 = (current_idx + i + step) % num_points
        idx3 = (current_idx + i + 2 * step) % num_points

        p1 = racetrack.centerline[idx1]
        p2 = racetrack.centerline[idx2]
        p3 = racetrack.centerline[idx3]

        v1 = p2 - p1
        v2 = p3 - p2

        if np.linalg.norm(v1) < 0.1 or np.linalg.norm(v2) < 0.1:
            continue

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # if angle is too small, means we are on a straight line
        # R is radius curvature (small R => sharp turn)
        if angle < 0.035:
            # Approximate straight line with large radius
            R = 10000.0
        else:
            # Calculate radius of curvature
            ds = np.linalg.norm(v1)
            R = ds / angle

        # Corner Speed Limit (max speed at the apex to avoid sliding out)
        # Is the target speed at the corner itself
        v_corner_limit = np.sqrt(LATERAL_ACCEL_LIMIT * R)

        # 2. Braking Distance Logic
        dist_to_corner = np.linalg.norm(p1 - car_position)
        v_allowable = np.sqrt(v_corner_limit**2 + 2 * AVAILABLE_DECEL * dist_to_corner)

        # Update minimum safe velocity across all lookahead points
        if v_allowable < min_safe_velocity:
            min_safe_velocity = v_allowable

    return np.clip(min_safe_velocity, MIN_SPEED, MAX_SPEED)


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """Controller tracking the centerline tightly using Velocity-Based Lookahead."""

    car_position = state[0:2]
    current_heading = state[4]
    current_velocity = max(state[3], 1.0)
    wheelbase = parameters[0] 

    # Lookahead Logic
    lookahead_idx = LOOKAHEAD_GAIN * current_velocity
    lookahead_idx = np.clip(lookahead_idx, LOOKAHEAD_MIN_IDX, LOOKAHEAD_MAX_IDX)

    target_point = find_target_point(car_position, racetrack.centerline, lookahead_idx)

    delta_x = target_point[0] - car_position[0]
    delta_y = target_point[1] - car_position[1]

    desired_heading = np.arctan2(delta_y, delta_x)
    delta_phi = desired_heading - current_heading
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))

    # Lateral Control
    dot_phi_desired = delta_phi / STEERING_LOOKAHEAD_TIME
    desired_steering_angle = np.arctan((wheelbase * dot_phi_desired) / current_velocity)
    desired_steering_angle = np.clip(
        desired_steering_angle, -PHYSICAL_STEERING_LIMIT, PHYSICAL_STEERING_LIMIT
    )

    # Speed Control
    desired_velocity = calculate_curvature_velocity(state, racetrack, parameters)

    return np.array([desired_steering_angle, desired_velocity])
