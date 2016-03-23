import os
import yaml

default_config_yaml = '''
# Metadata
use_exif_size: yes
default_focal_prior: 0.85

# Params for features
feature_type: SURF           # Feature type (HAHOG AKAZE, SURF, SIFT)
feature_root: 1               # If 1, apply square root mapping to features
feature_min_frames: 4000      # If fewer frames are detected, sift_peak_threshold/surf_hessian_threshold is reduced.
feature_process_size: 2048    # Resize the image if its size is larger than specified. Set to -1 for original size
feature_use_adaptive_suppression: no

# Params for SURF
surf_hessian_threshold: 1000  # 3000 Smaller value -> more features
surf_n_octaves: 4             # See OpenCV doc
surf_n_octavelayers: 2        # See OpenCV doc

# Masks for regions - ignore them for feature extraction
# List of bounding boxes specified as the ratio to image width and height
# masks: [{top: 0.96, bottom: 1.0, left: 0.0, right: 0.15}, {top: 0.95, bottom: 1.0, left: 0, right: 0.05}]

# Params for general matching
lowes_ratio: 0.8             # 0.75 to 0.8 as defined in Hartley and Zisserman 
preemptive_lowes_ratio: 0.6   
matcher_type: FLANN           

# Params for FLANN matching
flann_branching: 16           
flann_iterations: 10          
flann_checks: 200             # Smaller -> Fewer matches but faster

# Params for preemptive matching
matching_gps_distance: 150            # Maximum gps distance between two images for matching
matching_gps_neighbors: 0             # Number of images to match selected by GPS distance. Set to 0 to use no limit
matching_time_neighbors: 0            # Number of images to match selected by time taken. Set to 0 to use no limit
preemptive_max: 200                   # Number of features to use for preemptive matching
preemptive_threshold: 0               # If number of matches passes the threshold -> full feature matching

# Params for geometric estimation
robust_matching_threshold: 0.004      # Outlier threshold for fundamental matrix estimation as portion of image width
robust_matching_min_match: 20         # Minimum number of matches to be considered as an edge in the match grph
five_point_algo_threshold: 0.004      # Outlier threshold (in pixels) for essential matrix estimation
five_point_algo_min_inliers: 20       # Minimum number of inliers for considering a two view reconstruction valid.
triangulation_threshold: 0.006        # Outlier threshold (in pixels) for accepting a triangulated point.
triangulation_min_ray_angle: 1.0
resection_threshold: 0.004            # Outlier threshold (in pixels) for camera resection.
resection_min_inliers: 10             # Minimum number of resection inliers to accept it.
retriangulation: no
retriangulation_ratio: 1.25

# Params for track creation
min_track_length: 2             # Minimum number of features/images per track

# Params for bundle adjustment
loss_function: SoftLOneLoss     # Loss function for the ceres problem (see: http://ceres-solver.org/modeling.html#lossfunction)
loss_function_threshold: 1      # Threshold on the squared residuals.  Usually cost is quadratic for smaller residuals and sub-quadratic above.
reprojection_error_sd: 0.004    # The startard deviation of the reprojection error
exif_focal_sd: 0.01             # The standard deviation of the exif focal length in log-scale
radial_distorsion_k1_sd: 0.01   # The standard deviation of the first radial distortion parameter (mean assumed to be 0)
radial_distorsion_k2_sd: 0.01   # The standard deviation of the second radial distortion parameter (mean assumed to be 0)
bundle_interval: 0              # bundle adjustment after adding 'bundle_interval' cameras
bundle_new_points_ratio: 1.2    # bundle when (new points) / (bundled points) > bundle_outlier_threshold
bundle_outlier_threshold: 0.006

save_partial_reconstructions: no

# Params for GPS aligment
use_altitude_tag: no                  # Use or ignore EXIF altitude tag
align_method: orientation_prior       # orientation_prior or naive
align_orientation_prior: horizontal   # horizontal, vertical or no_roll

# Params for navigation graph
nav_min_distance: 0.01                # Minimum distance for a possible edge between two nodes
nav_step_pref_distance: 6             # Preferred distance between camera centers
nav_step_max_distance: 20             # Maximum distance for a possible step edge between two nodes
nav_turn_max_distance: 15             # Maixmum distance for a possible turn edge between two nodes
nav_step_forward_view_threshold: 15   # Maximum difference of angles in degrees between viewing directions for forward steps
nav_step_view_threshold: 30           # Maximum difference of angles in degrees between viewing directions for other steps
nav_step_drift_threshold: 36          # Maximum motion drift with respect to step directions for steps in degrees
nav_turn_view_threshold: 40           # Maximum difference of angles in degrees with respect to turn directions
nav_vertical_threshold: 20            # Maximum vertical angle difference in motion and viewving direction in degrees
nav_rotation_threshold: 30            # Maximum general rotation in degrees between cameras for steps

# Other params
processes: 1                  # Number of threads to use
'''


def default_config():
    '''Return default configuration
    '''
    return yaml.load(default_config_yaml)


def load_config(filepath):
    '''Load config from a config.yaml filepath
    '''
    config = default_config()

    if os.path.isfile(filepath):
        with open(filepath) as fin:
            new_config = yaml.load(fin)
        if new_config:
            for k, v in new_config.items():
                config[k] = v

    return config
