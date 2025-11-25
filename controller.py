import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# =============================================================================
# TUNING PARAMETERS - Adjust these for performance
# =============================================================================

# --- Lower Controller Gains ---
K_P_STEER = 6.0      # Faster steering response
K_P_ACCEL = 8.0      # Faster acceleration response

# --- Pure Pursuit Lookahead ---
LOOKAHEAD_GAIN = 0.6      # Higher = looks further ahead = smoother but cuts corners more
LOOKAHEAD_MIN = 15.0      # Minimum lookahead distance (meters)
LOOKAHEAD_MAX = 50.0      # Maximum lookahead distance (meters)

# --- Speed Profile Generation ---
LATERAL_ACCEL_MAX = 18.0   # Maximum lateral G-force (higher = faster cornering)
LONGITUDINAL_ACCEL = 15.0  # Acceleration limit
LONGITUDINAL_DECEL = 18.0  # Braking limit (can brake harder than accelerate)
GLOBAL_MAX_SPEED = 100.0    # Top speed limit
MIN_SPEED = 20.0            # Minimum speed

# --- Curvature Smoothing ---
CURVATURE_SMOOTH_WINDOW = 7  # Points to average for curvature calculation

# =============================================================================
# GLOBAL CACHE
# =============================================================================
SPEED_PROFILE_CACHE = None

# =============================================================================
# SPEED PROFILE GENERATION (Pre-computed once)
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
    
    # --- Calculate curvature ---
    curvatures = np.zeros(num_points)
    
    for i in range(num_points):
        prev_i = (i - CURVATURE_SMOOTH_WINDOW) % num_points
        next_i = (i + CURVATURE_SMOOTH_WINDOW) % num_points
        
        p_prev = centerline[prev_i]
        p_curr = centerline[i]
        p_next = centerline[next_i]
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 > 0.01 and len_v2 > 0.01:
            v1_norm = v1 / len_v1
            v2_norm = v2 / len_v2
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            avg_dist = (len_v1 + len_v2) / 2.0
            curvatures[i] = angle / avg_dist if avg_dist > 0.01 else 0.0
    
    # --- FIX: Safe Smoothing ---
    # We want the output to be exactly 'num_points' long.
    # We pad the array circularly before convolving.
    window = 5
    pad_width = window // 2
    
    # Pad with wrap-around data
    curvatures_padded = np.concatenate([
        curvatures[-pad_width:], 
        curvatures, 
        curvatures[:pad_width]
    ])
    
    # Convolve with 'valid' mode to remove padding artifacts
    smoothed = np.convolve(
        curvatures_padded, 
        np.ones(window) / window, 
        mode='valid'
    )
    
    # Ensure strict size matching (trim any float rounding excess)
    curvatures = smoothed[:num_points]
    
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
        
        # ERROR WAS HERE: v_max_corners must be size 'num_points'
        # With the fix above, this is now guaranteed.
        v_forward[i] = min(v_max_corners[i], v_accel)
    
    # --- Backward pass ---
    v_profile = np.zeros(num_points)
    v_profile[-1] = v_forward[-1]
    
    for i in range(num_points - 2, -1, -1):
        dist = distances[i]
        v_brake = np.sqrt(v_profile[i+1]**2 + 2 * LONGITUDINAL_DECEL * dist)
        v_profile[i] = min(v_forward[i], v_brake)
    
    # --- Loop closure ---
    for iteration in range(3):
        for i in range(num_points):
            next_i = (i + 1) % num_points
            dist = distances[i]
            v_brake = np.sqrt(v_profile[next_i]**2 + 2 * LONGITUDINAL_DECEL * dist)
            v_profile[i] = min(v_profile[i], v_brake)
    
    return v_profile

# =============================================================================
# PURE PURSUIT HELPERS
# =============================================================================

def find_closest_point_idx(position, centerline):
    """Find index of closest point on centerline."""
    distances = np.linalg.norm(centerline - position, axis=1)
    return np.argmin(distances)

def find_lookahead_point(current_idx, centerline, current_pos, lookahead_dist):
    """
    Find point on centerline that is approximately lookahead_dist meters ahead.
    Uses arc length along the path.
    """
    cumulative_dist = 0.0
    num_points = len(centerline)
    
    for i in range(1, num_points):
        idx = (current_idx + i) % num_points
        prev_idx = (current_idx + i - 1) % num_points
        
        segment_length = np.linalg.norm(centerline[idx] - centerline[prev_idx])
        cumulative_dist += segment_length
        
        if cumulative_dist >= lookahead_dist:
            return idx
    
    # If we complete the loop, return a point far ahead
    return (current_idx + num_points // 4) % num_points

def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

# =============================================================================
# CONTROLLERS
# =============================================================================

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level P controller for steering rate and acceleration.
    
    State: [sx, sy, steering_angle, velocity, heading]
    Desired: [desired_steering_angle, desired_velocity]
    Returns: [steering_rate, acceleration]
    """
    current_steering = state[2]
    current_velocity = state[3]
    
    desired_steering = desired[0]
    desired_velocity = desired[1]
    
    # P control for steering
    steering_error = desired_steering - current_steering
    steering_rate = K_P_STEER * steering_error
    
    # P control for velocity
    velocity_error = desired_velocity - current_velocity
    acceleration = K_P_ACCEL * velocity_error
    
    return np.array([steering_rate, acceleration])

def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    Pure Pursuit controller with pre-computed speed profile.
    
    State: [sx, sy, steering_angle, velocity, heading]
    Returns: [desired_steering_angle, desired_velocity]
    """
    global SPEED_PROFILE_CACHE
    
    # Generate speed profile once
    if SPEED_PROFILE_CACHE is None:
        SPEED_PROFILE_CACHE = generate_speed_profile(racetrack, parameters)
    
    # Extract state
    car_pos = state[0:2]
    heading = state[4]
    velocity = max(abs(state[3]), 0.1)  # Avoid division by zero
    
    # Parameters
    wheelbase = parameters[0]
    max_steering = parameters[4]
    
    # --- 1. Find closest point on centerline ---
    closest_idx = find_closest_point_idx(car_pos, racetrack.centerline)
    
    # --- 2. Calculate adaptive lookahead distance ---
    # Faster speed = look further ahead = smoother but wider turns
    lookahead_dist = LOOKAHEAD_GAIN * velocity
    lookahead_dist = np.clip(lookahead_dist, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
    
    # --- 3. Find lookahead point ---
    lookahead_idx = find_lookahead_point(
        closest_idx, racetrack.centerline, car_pos, lookahead_dist
    )
    target_point = racetrack.centerline[lookahead_idx]
    
    # --- 4. Pure Pursuit Steering Calculation ---
    # Vector from car to target
    target_vector = target_point - car_pos
    
    # Distance to target
    L = np.linalg.norm(target_vector)
    
    if L < 0.1:  # Too close to target
        desired_steering = 0.0
    else:
        # Angle to target in global frame
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        
        # Angle error (alpha) = angle between heading and target
        alpha = normalize_angle(target_angle - heading)
        
        # Pure Pursuit formula: steering = atan(2 * L_wb * sin(alpha) / L_d)
        desired_steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), L)
        
        # Clip to steering limits
        desired_steering = np.clip(desired_steering, -max_steering, max_steering)
    
    # --- 5. Get desired velocity from pre-computed profile ---
    # Look slightly ahead for smoother braking
    speed_lookahead_idx = (closest_idx + 3) % len(racetrack.centerline)
    desired_velocity = SPEED_PROFILE_CACHE[speed_lookahead_idx]
    
    return np.array([desired_steering, desired_velocity])