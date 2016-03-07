import cv2,argparse, os, numpy as np

from progressbar import ProgressBar, Percentage, Bar
from stereovision.calibration import StereoCalibrator
from stereovision.exceptions import BadBlockMatcherArgumentError
from stereovision.stereo_cameras import CalibratedPair
from stereovision.calibration import StereoCalibration
#from stereovision 
import blockmatchers

from matplotlib import pyplot as plt
import math
import argparse as ap
#interactive GUI Tracker for Disparity Map
class StereoBMTuner(object):
    """
    A class for tuning Stereo BM settings.
    Display a normalized disparity picture from two pictures captured with a
    ``CalibratedPair`` and allow the user to manually tune the settings for the
    stereo block matcher.
    """
    #: Window to show results in
    window_name = "Stereo BM Tuner"
    def __init__(self, calibrated_pair, image_pair):
	"""Initialize tuner with a ``CalibratedPair`` and tune given pair."""
        #: Calibrated stereo pair to find Stereo BM settings for
        self.calibrated_pair = calibrated_pair
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("cam_preset", self.window_name,
                           self.calibrated_pair.block_matcher.stereo_bm_preset, 3,
                           self.set_bm_preset)
        cv2.createTrackbar("ndis", self.window_name,
                           self.calibrated_pair.block_matcher.search_range, 160,
                           self.set_search_range)
        cv2.createTrackbar("winsize", self.window_name,
                           self.calibrated_pair.block_matcher.window_size, 21,
                           self.set_window_size)
        #: (left, right) image pair to find disparity between
        self.pair = image_pair
        #self.tune_pair(image_pair)
	while True:
		if cv2.waitKey(1) & 0xFF == 27:
			break
		print cv2.getTrackbarPos('ndis',"Stereo BM Tuner")
		self.set_bm_preset(cv2.getTrackbarPos('cam_preset',self.window_name))
		self.set_search_range(cv2.getTrackbarPos('ndis',self.window_name))
		self.set_window_size(cv2.getTrackbarPos('winsize',self.window_name))
		self.update_disparity_map()

    def set_bm_preset(self, preset):
	print 'setting bm_preset'
        """Set ``search_range`` and update disparity image."""
        try:
            self.calibrated_pair.block_matcher.stereo_bm_preset = preset
        except blockmatchers.InvalidBMPreset:
	    print 'Invalid BM Preset'
            return
        #self.update_disparity_map()
    def set_search_range(self, search_range):
	print 'setting search range'
        """Set ``search_range`` and update disparity image."""
        try:
            self.calibrated_pair.block_matcher.search_range = search_range
        except blockmatchers.InvalidSearchRange:
	    print 'Invalid Search Range'
            return
        #self.update_disparity_map()
    def set_window_size(self, window_size):
        """Set ``window_size`` and update disparity image."""
        try:
            self.calibrated_pair.block_matcher.window_size = window_size
        except blockmatchers.InvalidWindowSize:
	    print 'Invalid Win Size'
            return
        #self.update_disparity_map()
    def update_disparity_map(self):
        print 'update disp map! '
        """Update disparity map in GUI."""
        disparity = self.calibrated_pair.block_matcher.compute_disparity(self.pair)
        cv2.imshow(self.window_name, disparity / 255)
        cv2.waitKey()
    def tune_pair(self, pair):
        """Tune a pair of images."""
        self.pair = pair
        self.update_disparity_map()

def find_files(folder):
    """Discover stereo photos and return them as a pairwise sorted list."""
    files = [i for i in os.listdir(folder) if i.startswith("left")]
    files.sort()
    for i in range(len(files)):
        insert_string = "right{}".format(files[i * 2][4:])
        files.insert(i * 2 + 1, insert_string)
    files = [os.path.join(folder, filename) for filename in files]
    return files

if __name__ == "__main__":

    """Let user tune all images in the input folder and report chosen values."""
    parser = argparse.ArgumentParser(description="Read images taken from a "
                                     "calibrated stereo pair, compute "
                                     "disparity maps from them and show them "
                                     "interactively to the user, allowing the "
                                     "user to tune the stereo block matcher "
                                     "settings in the GUI.")
    parser.add_argument("calibration_folder",
                        help="Directory where calibration files for the stereo "
                        "pair are stored.")
    parser.add_argument("image_folder",
                        help="Directory where input images are stored.")
    args = parser.parse_args()
    calibration = StereoCalibration(
                                        input_folder=args.calibration_folder)
    input_files = find_files(args.image_folder)
    calibrated_pair = CalibratedPair(None, calibration,blockmatchers.StereoBM())
    image_pair = [cv2.imread(image) for image in input_files[:2]]
    rectified_pair = calibration.rectify(image_pair)
    tuner = StereoBMTuner(calibrated_pair, rectified_pair)
    chosen_arguments = []
    while input_files:
        image_pair = [cv2.imread(image) for image in input_files[:2]]
	image_names = [image for image in input_files[:2]]
	print image_names
        rectified_pair = calibration.rectify(image_pair)
        tuner.tune_pair(rectified_pair)
        chosen_arguments.append((calibrated_pair.block_matcher.stereo_bm_preset,
                                 calibrated_pair.block_matcher.search_range,
                                 calibrated_pair.block_matcher.window_size))
        input_files = input_files[2:]
    stereo_bm_presets, search_ranges, window_sizes = [], [], []
    for preset, search_range, size in chosen_arguments:
        stereo_bm_presets.append(preset)
        search_ranges.append(search_range)
        window_sizes.append(size)
    for name, values in (("Stereo BM presets", stereo_bm_presets),
                         ("Search ranges", search_ranges),
                         ("Window sizes", window_sizes)):
        report_variable(name, values)
        print()
