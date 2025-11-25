import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# =============================================================================
# TUNING PARAMETERS
# =============================================================================

# --- Lower Controller Gains ---
K_P_STEER = 6.0
K_P_ACCEL = 8.0

# --- Pure Pursuit Lookahead ---
LOOKAHEAD_GAIN = 0.4
LOOKAHEAD_MIN = 8.0
LOOKAHEAD_MAX = 50.0

# --- Speed Profile Generation ---
LATERAL_ACCEL_MAX = 18.0
LONGITUDINAL_ACCEL = 15.0
LONGITUDINAL_DECEL = 18.0
GLOBAL_MAX_SPEED = 100.0
MIN_SPEED = 8.0

# --- Curvature Smoothing ---
CURVATURE_SMOOTH_WINDOW = 5 

# =============================================================================
# GLOBAL CACHE
# =============================================================================
SPEED_PROFILE_CACHE = None

# =============================================================================
# SPEED PROFILE GENERATION
# =============================================================================

def generate_speed_profile(racetrack, parameters):
    """
    Pre-computes optimal speed profile based on track geometry.
    """
    centerline = racetrack.centerline
    num_points = len(centerline)
    
    # --- Calculate segment distances ---
    segments = np.roll(centerline, -1, axis=0) - centerline
    distances = np.linalg.norm(segments, axis=1)
    distances = np.maximum(distances, 0.01)
    
    # --- Calculate curvature using 3-point method ---
    curvatures = np.zeros(num_points)
    
    # Pre-calculate indices for efficiency
    indices = np.arange(num_points)
    prev_indices = (indices - CURVATURE_SMOOTH_WINDOW) % num_points
    next_indices = (indices + CURVATURE_SMOOTH_WINDOW) % num_points
    
    # Vectorized Curvature Calculation
    p_prev = centerline[prev_indices]
    p_curr = centerline[indices]
    p_next = centerline[next_indices]
    
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    
    len_v1 = np.linalg.norm(v1, axis=1)
    len_v2 = np.linalg.norm(v2, axis=1)
    
    # Valid mask to avoid division by zero
    valid = (len_v1 > 0.01) & (len_v2 > 0.01)
    
    v1_norm = np.zeros_like(v1)
    v2_norm = np.zeros_like(v2)
    v1_norm[valid] = v1[valid] / len_v1[valid, None]
    v2_norm[valid] = v2[valid] / len_v2[valid, None]
    
    # Calculate angle and curvature
    dot_prod = np.sum(v1_norm * v2_norm, axis=1)
    cos_angle = np.clip(dot_prod, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    avg_dist = (len_v1 + len_v2) / 2.0
    curvatures[valid] = angle[valid] / avg_dist[valid]
    
    # --- FIX: Correct Smoothing Padding ---
    # We want output size to match input size exactly.
    # Pad by half the window size on both sides.
    pad_size = CURVATURE_SMOOTH_WINDOW // 2
    
    curvatures_padded = np.concatenate([
        curvatures[-pad_size:], 
        curvatures, 
        curvatures[:pad_size]
    ])
    
    # Convolve 'valid' results in exactly 'num_points'
    curvatures = np.convolve(
        curvatures_padded, 
        np.ones(CURVATURE_SMOOTH_WINDOW) / CURVATURE_SMOOTH_WINDOW, 
        mode='valid'
    )
    
    curvatures = np.maximum(curvatures, 1e-6)

    # --- Calculate maximum cornering speed ---
    v_max_corners = np.sqrt(LATERAL_ACCEL_MAX / curvatures)
    v_max_corners = np.clip(v_max_corners, MIN_SPEED, GLOBAL_MAX_SPEED)
    
    # --- Forward pass ---
    v_forward = np.zeros(num_points)
    v_forward[0] = MIN_SPEED
    
    for i in range(1, num_points):
        dist = distances[i-1]
        v_accel = np.sqrt(v_forward[i-1]**2 + 2 * LONGITUDINAL_ACCEL * dist)
        v_forward[i] = min(v_max_corners[i], v_accel)
    
    # --- Backward pass ---
    v_profile = np.zeros(num_points)
    v_profile[-1] = v_forward[-1]
    
    for i in range(num_points - 2, -1, -1):
        dist = distances[i]
        v_brake = np.sqrt(v_profile[i+1]**2 + 2 * LONGITUDINAL_DECEL * dist)
        v_profile[i] = min(v_forward[i], v_brake)
    
    # --- Loop closure ---
    for _ in range(3):
        for i in range(num_points):
            next_i = (i + 1) % num_points
            dist = distances[i]
            v_brake = np.sqrt(v_profile[next_i]**2 + 2 * LONGITUDINAL_DECEL * dist)
            v_profile[i] = min(v_profile[i], v_brake)
    
    return v_profile

# =============================================================================
# HELPERS & CONTROLLERS
# =============================================================================

def find_closest_point_idx(position, centerline):
    distances = np.linalg.norm(centerline - position, axis=1)
    return np.argmin(distances)

def find_lookahead_point(current_idx, centerline, current_pos, lookahead_dist):
    cumulative_dist = 0.0
    num_points = len(centerline)
    
    for i in range(1, num_points):
        idx = (current_idx + i) % num_points
        prev_idx = (current_idx + i - 1) % num_points
        segment_length = np.linalg.norm(centerline[idx] - centerline[prev_idx])
        cumulative_dist += segment_length
        if cumulative_dist >= lookahead_dist:
            return idx
    return (current_idx + num_points // 4) % num_points

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    current_steering = state[2]
    current_velocity = state[3]
    desired_steering = desired[0]
    desired_velocity = desired[1]
    
    steering_rate = K_P_STEER * (desired_steering - current_steering)
    acceleration = K_P_ACCEL * (desired_velocity - current_velocity)
    
    return np.array([steering_rate, acceleration])

def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    global SPEED_PROFILE_CACHE
    if SPEED_PROFILE_CACHE is None:
        SPEED_PROFILE_CACHE = generate_speed_profile(racetrack, parameters)
    
    car_pos = state[0:2]
    heading = state[4]
    velocity = max(abs(state[3]), 0.1)
    
    wheelbase = parameters[0]
    max_steering = parameters[4]
    
    # 1. Lookahead logic
    closest_idx = find_closest_point_idx(car_pos, racetrack.centerline)
    lookahead_dist = np.clip(LOOKAHEAD_GAIN * velocity, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
    lookahead_idx = find_lookahead_point(closest_idx, racetrack.centerline, car_pos, lookahead_dist)
    target_point = racetrack.centerline[lookahead_idx]
    
    # 2. Pure Pursuit
    target_vector = target_point - car_pos
    L = np.linalg.norm(target_vector)
    
    if L < 0.1:
        desired_steering = 0.0
    else:
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        alpha = normalize_angle(target_angle - heading)
        desired_steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), L)
        desired_steering = np.clip(desired_steering, -max_steering, max_steering)
    
    # 3. Velocity
    speed_idx = (closest_idx + 5) % len(racetrack.centerline)
    desired_velocity = SPEED_PROFILE_CACHE[speed_idx]
    
    return np.array([desired_steering, desired_velocity])