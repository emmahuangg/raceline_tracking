import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# =============================================================================
# TUNING PARAMETERS — These values control how the car behaves on the track.
# =============================================================================

# Steering and acceleration gains for the low-level proportional controller.
# A proportional controller compares "what we want" with "what we have" and
# multiplies the difference (the error) by a constant gain.
# Higher gains → stronger reaction to error → faster response but more risk of oscillation.
K_P_STEER = 6.0
K_P_ACCEL = 8.0

# Parameters that control how far ahead on the track the car looks.
# The "lookahead distance" is the distance ahead of the car where we pick a target point
# for the car to aim toward. A larger lookahead makes the motion smoother but can cut
# corners more; a smaller lookahead follows the path more tightly but can be twitchy.
LOOKAHEAD_GAIN = 0.6      # Scales with current speed to get a basic lookahead distance.
LOOKAHEAD_MIN = 15.0      # Minimum distance the car will ever look ahead.
LOOKAHEAD_MAX = 50.0      # Maximum distance the car will ever look ahead.

# Parameters used to create a speed profile along the track.
# The idea is: at each point on the track, we pick a "maximum safe speed"
# based on how sharp the track turns and how quickly the car can speed up or slow down.
LATERAL_ACCEL_MAX = 18.0   # Maximum lateral acceleration (sideways acceleration in turns).
                           # Higher value → car can take corners faster.
LONGITUDINAL_ACCEL = 15.0  # Maximum forward acceleration (how quickly the car speeds up).
LONGITUDINAL_DECEL = 18.0  # Maximum braking deceleration (how quickly the car slows down).
GLOBAL_MAX_SPEED = 100.0   # Hard upper limit on speed, no matter what the track looks like.
MIN_SPEED = 20.0           # Minimum speed to avoid the car going too slowly or stopping.

# Number of points on each side used when smoothing curvature.
# Curvature is sensitive to noise from discrete path points, so we average nearby values.
CURVATURE_SMOOTH_WINDOW = 7

# =============================================================================
# GLOBAL CACHE — This holds the precomputed speed profile for the track.
# We only compute it once, then reuse it every control step.
# =============================================================================
SPEED_PROFILE_CACHE = None

# =============================================================================
# SPEED PROFILE GENERATION
# This function examines the track and decides a safe speed for each point.
# It uses how sharp the track is (curvature) and how fast the car can accelerate
# or brake along the path.
# =============================================================================

def generate_speed_profile(racetrack, parameters):
    """
    Create a speed profile along the track centerline.

    The main idea:
      1. Estimate how sharp the track is at each point (curvature).
      2. From curvature, compute a maximum speed that keeps lateral acceleration
         below a chosen limit.
      3. Do a forward pass around the track, making sure we do not accelerate
         faster than allowed.
      4. Do a backward pass around the track, making sure we can brake in time
         before sharp corners.

    The result is a speed value at each track waypoint that is consistent with
    both turning limits and acceleration/braking limits.
    """
    centerline = racetrack.centerline
    num_points = len(centerline)
    
    # --- Step 1: Compute distance between consecutive waypoints (approximate path length) ---
    # For each pair of neighboring points on the track, we compute the straight-line distance.
    # This gives us an approximation of the local arc length between waypoints.
    segments = np.roll(centerline, -1, axis=0) - centerline
    distances = np.linalg.norm(segments, axis=1)
    distances = np.maximum(distances, 0.01)  # Avoid zero distances that would break formulas.
    
    # --- Step 2: Estimate curvature at each waypoint ---
    # Curvature here represents how sharply the path bends at a point.
    # Higher curvature → sharper turn → we must slow down.
    curvatures = np.zeros(num_points)
    
    for i in range(num_points):
        # Instead of only looking at immediate neighbors, we look a few points behind
        # and ahead, then compare directions. This makes curvature less noisy.
        prev_i = (i - CURVATURE_SMOOTH_WINDOW) % num_points
        next_i = (i + CURVATURE_SMOOTH_WINDOW) % num_points
        
        p_prev = centerline[prev_i]
        p_curr = centerline[i]
        p_next = centerline[next_i]
        
        # Vectors describing where the path is coming from and where it is going next.
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        # If both segments have reasonable length, estimate the direction change.
        if len_v1 > 0.01 and len_v2 > 0.01:
            # Normalize the direction vectors.
            v1_norm = v1 / len_v1
            v2_norm = v2 / len_v2
            
            # Compute the angle between the incoming and outgoing directions.
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Use average distance between the two segments to approximate curvature:
            # curvature ≈ angle change / distance traveled.
            avg_dist = (len_v1 + len_v2) / 2.0
            curvatures[i] = angle / avg_dist if avg_dist > 0.01 else 0.0
    
    # --- Step 3: Smooth curvature so speed does not change abruptly ---
    # We apply a simple moving average. Since the track loops, we pad the data
    # using wrap-around so that smoothing works correctly near the start/end.
    window = 5
    pad_width = window // 2
    
    curvatures_padded = np.concatenate([
        curvatures[-pad_width:], 
        curvatures, 
        curvatures[:pad_width]
    ])
    
    smoothed = np.convolve(
        curvatures_padded, 
        np.ones(window) / window,
        mode='valid'
    )
    
    # Keep exactly one curvature value per waypoint.
    curvatures = smoothed[:num_points]
    curvatures = np.maximum(curvatures, 1e-6)  # Prevent division by zero for very straight parts.
    
    # --- Step 4: Convert curvature into a maximum cornering speed ---
    # Physics idea: for a given lateral acceleration limit and curvature, there is a maximum
    # speed we can safely travel without exceeding that sideways acceleration.
    # v_max ≈ sqrt(a_lateral_max / curvature)
    v_max_corners = np.sqrt(LATERAL_ACCEL_MAX / curvatures)
    v_max_corners = np.clip(v_max_corners, MIN_SPEED, GLOBAL_MAX_SPEED)
    
    # --- Step 5: Forward pass — ensure acceleration limits are respected ---
    # We start with a minimum speed and move forward along the track. At each step,
    # we compute the maximum speed we can reach from the previous speed, given the
    # maximum forward acceleration and the distance between points.
    v_forward = np.zeros(num_points)
    v_forward[0] = MIN_SPEED
    
    for i in range(1, num_points):
        dist = distances[i-1]
        
        # Based on the kinematic relation: v_next^2 = v_prev^2 + 2 * a * Δs.
        v_accel = np.sqrt(v_forward[i-1]**2 + 2 * LONGITUDINAL_ACCEL * dist)
        
        # Limit by both the curvature-based speed and the acceleration-based speed.
        v_forward[i] = min(v_max_corners[i], v_accel)
    
    # --- Step 6: Backward pass — ensure braking limits are respected ---
    # Now we go in the opposite direction around the track. At each point, we ask:
    # given the next point's speed and the braking limit, what is the highest speed
    # we can have here and still be able to slow down in time?
    v_profile = np.zeros(num_points)
    v_profile[-1] = v_forward[-1]
    
    for i in range(num_points - 2, -1, -1):
        dist = distances[i]
        
        # Same kinematic relation, but using braking deceleration.
        v_brake = np.sqrt(v_profile[i+1]**2 + 2 * LONGITUDINAL_DECEL * dist)
        
        # This point's speed must satisfy both forward and backward constraints.
        v_profile[i] = min(v_forward[i], v_brake)
    
    # --- Step 7: Extra passes to smooth the loop closure ---
    # Because the track loops, we run a few extra passes that consider wrap-around,
    # so that speeds near the start and end match smoothly and still respect braking.
    for iteration in range(3):
        for i in range(num_points):
            next_i = (i + 1) % num_points
            dist = distances[i]
            v_brake = np.sqrt(v_profile[next_i]**2 + 2 * LONGITUDINAL_DECEL * dist)
            v_profile[i] = min(v_profile[i], v_brake)
    
    return v_profile

# =============================================================================
# GEOMETRY HELPERS FOR PATH FOLLOWING
# These functions help us find where the car is relative to the path and
# pick a suitable target point ahead on the path.
# =============================================================================

def find_closest_point_idx(position, centerline):
    """
    Find the index of the path point (on the centerline) that is closest to the car's
    current position. This acts as our "current location" on the path.
    """
    distances = np.linalg.norm(centerline - position, axis=1)
    return np.argmin(distances)

def find_lookahead_point(current_idx, centerline, current_pos, lookahead_dist):
    """
    Starting from the current index on the centerline, walk forward along the path
    and accumulate distance until the total distance exceeds the desired lookahead
    distance. The point where this happens is our target point for steering.
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
    
    # If we somehow loop all the way around without reaching the distance,
    # fall back to a point somewhat ahead so that the function always returns something.
    return (current_idx + num_points // 4) % num_points

def normalize_angle(angle):
    """
    Wrap an angle into the range [-pi, pi].
    This keeps angles numerically stable and easy to compare.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

# =============================================================================
# LOW-LEVEL PROPORTIONAL CONTROLLER
# This controller takes desired steering angle and desired speed and turns them
# into steering rate and acceleration commands based on simple feedback.
# =============================================================================

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Given the current state of the car and desired steering and speed, compute
    steering rate and acceleration commands using proportional control.

    Idea:
      - For steering: steering_rate ∝ (desired_steering - current_steering).
      - For speed: acceleration ∝ (desired_velocity - current_velocity).

    Larger errors produce larger corrections. The proportional gains above decide
    how strong those corrections are.
    """
    current_steering = state[2]
    current_velocity = state[3]
    
    desired_steering = desired[0]
    desired_velocity = desired[1]
    
    # Steering feedback: correct the difference between current and desired steering angle.
    steering_error = desired_steering - current_steering
    steering_rate = K_P_STEER * steering_error
    
    # Velocity feedback: correct the difference between current and desired speed.
    velocity_error = desired_velocity - current_velocity
    acceleration = K_P_ACCEL * velocity_error
    
    return np.array([steering_rate, acceleration])

# =============================================================================
# HIGH-LEVEL PATH FOLLOWING CONTROLLER
# This controller:
#   1. Figures out where the car is on the path.
#   2. Picks a target point some distance ahead on the path.
#   3. Computes a steering angle that would drive the car toward that target.
#   4. Chooses a desired speed from the precomputed speed profile.
# =============================================================================

def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    Compute desired steering angle and desired velocity for the car.

    High-level steps:
      - Use the speed profile to know how fast we can safely go at our position.
      - Use lookahead distance to pick a goal point ahead on the centerline.
      - Steer toward this goal point using simple geometry.
      - Return the desired steering angle and desired speed for the low-level controller.
    """
    global SPEED_PROFILE_CACHE
    
    # Compute the speed profile only once. After that, reuse the cached result.
    if SPEED_PROFILE_CACHE is None:
        SPEED_PROFILE_CACHE = generate_speed_profile(racetrack, parameters)
    
    # Unpack the state of the car.
    car_pos = state[0:2]       # [x, y] position
    heading = state[4]         # heading angle
    velocity = max(abs(state[3]), 0.1)  # use a small minimum to avoid division by zero later
    
    wheelbase = parameters[0]  # distance between front and rear axles
    max_steering = parameters[4]
    
    # Step 1: Find the closest point on the centerline to the car.
    closest_idx = find_closest_point_idx(car_pos, racetrack.centerline)
    
    # Step 2: Compute how far ahead to look based on current speed.
    # Faster speed → look further ahead so motion is smoother and more stable.
    lookahead_dist = LOOKAHEAD_GAIN * velocity
    lookahead_dist = np.clip(lookahead_dist, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
    
    # Step 3: Find the index of the lookahead point along the centerline.
    lookahead_idx = find_lookahead_point(
        closest_idx, racetrack.centerline, car_pos, lookahead_dist
    )
    target_point = racetrack.centerline[lookahead_idx]
    
    # Step 4: Compute the steering angle that would drive the car toward the target point.
    # We think of the car following an arc that goes through the target point.
    target_vector = target_point - car_pos
    L = np.linalg.norm(target_vector)  # distance from car to target point
    
    if L < 0.1:
        # If we are already very close to the target point, do not turn.
        desired_steering = 0.0
    else:
        # Angle from the car position to the target point in world coordinates.
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        
        # Heading error: how much we need to rotate from our current heading
        # to face the target point.
        alpha = normalize_angle(target_angle - heading)
        
        # Turning relation: we want a curvature that makes an arc passing through
        # the target point. For a simple bicycle model, the steering angle that
        # approximates this is:
        # steering = atan(2 * wheelbase * sin(alpha) / L)
        desired_steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), L)
        
        # Limit the steering angle to what the car can physically achieve.
        desired_steering = np.clip(desired_steering, -max_steering, max_steering)
    
    # Step 5: Choose a desired speed a little ahead of the closest point.
    # Looking slightly ahead for speed helps the car start braking early for corners.
    speed_lookahead_idx = (closest_idx + 3) % len(racetrack.centerline)
    desired_velocity = SPEED_PROFILE_CACHE[speed_lookahead_idx]
    
    return np.array([desired_steering, desired_velocity])
