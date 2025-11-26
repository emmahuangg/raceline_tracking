import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# Proportional gains (Kp)
Kp_steer = 6.0
Kp_accel = 8.0

# Pure Pursuit Parameters
# L = Gain * v (adaptive lookahead)
L_GAIN = 0.6
L_MIN = 20.0
L_MAX = 50.0

# Speed limits
V_MAX = 100.0
V_MIN = 20.0

# Smoothing window for curvature calculation
WINDOW_SIZE = 7

# Reference velocities
ref_velocities = None

def generate_ref_velocities(racetrack, parameters):
    """
    Generates reference velocities along the racetrack centerline.
    """
    centerline = racetrack.centerline
    max_acceleration = parameters[10]
    max_deceleration = -parameters[8] - 1 # Slightly reduce max deceleration for safety
    N = len(centerline)
    
    # 1. Calculate distances between centerline points
    ds = np.maximum(np.linalg.norm(np.roll(centerline, -1, axis=0) - centerline, axis=1), 0.01) # Avoid zero distances
    
    # 2. Estimate curvature at each point
    curvatures = np.zeros(N)
    for i in range(N):
        p_prev = centerline[(i - WINDOW_SIZE) % N]
        p_next = centerline[(i + WINDOW_SIZE) % N]
        p_curr = centerline[i]
        
        v1 = p_curr - p_prev # Vector from prev to curr
        v2 = p_next - p_curr # Vector from curr to next

        norm_v1 = np.linalg.norm(v1) # Distance between prev and curr
        norm_v2 = np.linalg.norm(v2) # Distance between curr and next
        
        if norm_v1 > 0.01 and norm_v2 > 0.01:
            v1_u = v1 / norm_v1 # Unit vector from prev to curr
            v2_u = v2 / norm_v2 # Unit vector from curr to next
            theta = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) # Angle between vectors
            
            # curvature = change in angle / avg distance
            avg_dist = (norm_v1 + norm_v2) / 2.0
            curvatures[i] = theta / avg_dist if avg_dist > 0.01 else 0.0 # Avoid division by zero
    
    # 3. Smooth curvature
    curvatures = np.pad(curvatures, 2, mode='wrap') # Add 2 points on each side of the array for wrap-around smoothing
    curvatures = np.convolve(curvatures, np.ones(5)/5, mode='valid') # Calculates a moving average to smooth curvature
    curvatures = np.maximum(curvatures, 1e-6) # Avoid zero curvature
    
    # 4. Lateral acceleration limit
    # Ensures that the car can make the turn without losing grip
    # a_lat = v² / R => v = sqrt(a_lat * R)
    v_max_lat = np.clip(np.sqrt(max_deceleration / curvatures), V_MIN, V_MAX)
    
    # 5. Forward pass using acceleration limits
    # Assign velocities considering acceleration limits
    v_fwd = np.zeros(N) # Stores results of forward pass
    v_fwd[0] = V_MIN
    
    for i in range(1, N):
        dist = ds[i-1]
        # v_next^2 = v_curr^2 + 2*a*d (kinematic equation)
        v_possible = np.sqrt(v_fwd[i-1]**2 + 2 * max_acceleration * dist) # Max velocity achievable at this point, given the previous velocity and acceleration limit
        v_fwd[i] = min(v_max_lat[i], v_possible)
    
    # 6. Backward pass using braking limits
    # Assign velocities considering braking limits by going backwards
    v_final = np.zeros(N) # Final reference velocities
    v_final[-1] = v_fwd[-1]
    
    for i in range(N - 2, -1, -1):
        dist = ds[i]
        # v_curr^2 = v_next^2 + 2*a*d (kinematic equation)
        v_brake = np.sqrt(v_final[i+1]**2 + 2 * max_deceleration * dist) # Max velocity allowed at this point, given the next velocity and braking limit
        v_final[i] = min(v_fwd[i], v_brake)
    
    # 7. Wrap-around smoothing
    # Repeat backward pass multiple times to ensure smoothness at the start/end
    for _ in range(3):
        for i in range(N):
            next_i = (i + 1) % N
            dist = ds[i]
            v_brake = np.sqrt(v_final[next_i]**2 + 2 * max_deceleration * dist)
            v_final[i] = min(v_final[i], v_brake)
    
    return v_final

def get_closest_index(pos, centerline):
    dists = np.linalg.norm(centerline - pos, axis=1)
    return np.argmin(dists)

def get_lookahead_index(curr_idx, centerline, lookahead_dist):
    """
    Finds the index on the centerline that is at least lookahead_dist away from curr_idx.
    """
    cum_dist = 0.0 # Cumulative distance that has been traversed
    N = len(centerline)
    
    for i in range(1, N): # Start from next point
        idx = (curr_idx + i) % N # Wrap around
        prev_idx = (curr_idx + i - 1) % N # Previous index
        d = np.linalg.norm(centerline[idx] - centerline[prev_idx]) # Distance between points
        cum_dist += d
        
        if cum_dist >= lookahead_dist:
            return idx
            
    return (curr_idx + N - 1) % N # Fallback to last point if not found

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Calculates control inputs (v_delta, a) to track references (delta_ref, v_ref).
    """
    delta = state[2]
    v = state[3]    
    
    delta_ref = desired[0]
    v_ref = desired[1]
    
    e_delta = delta_ref - delta
    v_delta = Kp_steer * e_delta
    
    e_v = v_ref - v
    a = Kp_accel * e_v
    
    return np.array([v_delta, a])

def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    Calculates references (delta_ref, v_ref) for the lower controller.
    """
    global ref_velocities
    
    if ref_velocities is None:
        ref_velocities = generate_ref_velocities(racetrack, parameters) # Precompute reference velocities once
    
    s_x = state[0]
    s_y = state[1]
    phi = state[4]
    v = max(abs(state[3]), 0.1) # Avoid zero velocity
    l_wb = parameters[0]  # wheelbase
    delta_max = parameters[4]
    
    # 1. Find closest point to current racecar position on centerline
    closest_idx = get_closest_index(np.array([s_x, s_y]), racetrack.centerline)
    
    # 2. Determine lookahead distance
    L = np.clip(L_GAIN * v, L_MIN, L_MAX) # Adaptive lookahead based on speed
    
    # 3. Find target point at lookahead distance
    target_idx = get_lookahead_index(closest_idx, racetrack.centerline, L)
    p_ref = racetrack.centerline[target_idx]
    
    # 4. Pure pursuit
    # Vector to target
    dx = p_ref[0] - s_x
    dy = p_ref[1] - s_y
    
    # Heading to target
    heading_angle = np.arctan2(dy, dx)
    
    # Heading error relative to car heading
    heading_error = np.arctan2(np.sin(heading_angle - phi), np.cos(heading_angle - phi)) # Normalize angle to [-pi, pi]
    
    # Steering angle calculation
    delta_ref = np.arctan2(2.0 * l_wb * np.sin(heading_error), L)
    delta_ref = np.clip(delta_ref, -delta_max, delta_max) # Clamp to max steering angle possible
    
    # 5. Get reference velocity
    # Look slightly ahead for the velocity reference to anticipate braking
    v_idx = (closest_idx + 3) % len(racetrack.centerline)
    v_ref = ref_velocities[v_idx]
    
    return np.array([delta_ref, v_ref])