# 
# Offsetting 
# the key: http://stackoverflow.com/questions/6087241/opencv-warpperspective
#

# For the ocean panorama, SIFT found a lot more features. This 
# resulted in a much better stitching. (SURF only found 4 and it
# warped considerably)

# Test cases
# python stitch.py Image1.jpg Image2.jpg -a SIFT
# python stitch.py Image2.jpg Image1.jpg -a SIFT
# python stitch.py ../stitcher/images/image_5.png ../stitcher/images/image_6.png -a SIFT
# python stitch.py ../stitcher/images/image_6.png ../stitcher/images/image_5.png -a SIFT
# python stitch.py ../vashon/01.JPG ../vashon/02.JPG -a SIFT
# python stitch.py panorama_vashon2.jpg ../vashon/04.JPG -a SIFT
# python stitch.py ../books/02.JPG ../books/03.JPG -a SIFT

# coding: utf-8
import cv2,argparse, numpy as np
from stereovision import *
from matplotlib import pyplot as plt
import math
import argparse as ap
import calibrate
DEBUG = False

match = 0
## 1. Extract SURF keypoints and descriptors from an image. [4] ----------
def extract_features(image, surfThreshold=100, algorithm='SURF'):

  # Convert image to grayscale (for SURF detector).
  image_gs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  
  if DEBUG:
      cv2.imwrite("out/gray.jpg", image_gs)
  
  # Detect SURF features and compute descriptors.
  detector = cv2.xfeatures2d.SURF_create() # what happens with SIFT?
  # descriptor = cv2.DescriptorExtractor_create(algorithm) # what happens with SIFT?
  
  # kp = detector.detect(image_gs)
  (keypoints,descriptors) = detector.detectAndCompute(image_gs,None)
  
  ## TODO: (Overwrite the following 2 lines with your answer.)
  # descriptors = np.array([[1,1], [7,5], [5,2], [3,4]], np.float32)
  # keypoints = [cv2.KeyPoint(100 * x, 100 * y, 1) for (x,y) in descriptors]
  print 'no of descriptors: ',len(descriptors)
  raw_input()
  return (keypoints, descriptors)



## 2. Find corresponding features between the images. [2] ----------------
def find_correspondences(img1,img2,keypoints1, descriptors1, keypoints2, descriptors2):
  global match
  ## Find corresponding features.
  match = match_flann(img1,img2,descriptors1, descriptors2)
  print 'No of points matched: ',len(match) #2573
  points1 = np.array([keypoints1[i].pt for (i, j) in match], np.float32)
  points2 = np.array([keypoints2[j].pt for (i, j) in match], np.float32)
  
  ## TODO: Look up corresponding keypoints.
  ## TODO: (Overwrite the following 2 lines with your answer.)
  # points1 = np.array([k.pt for k in keypoints1], np.float32)
  # points2 = np.array([k.pt for k in keypoints1], np.float32)

  return (points1, points2)


## 3. Calculate the size and offset of the stitched panorama. [5] --------



def calculate_size(size_image1, size_image2, homography):
  
  (h1, w1) = size_image1[:2]
  (h2, w2) = size_image2[:2]
  
  #remap the coordinates of the projected image onto the panorama image space
  top_left = np.dot(homography,np.asarray([0,0,1]))
  top_right = np.dot(homography,np.asarray([w2,0,1]))
  bottom_left = np.dot(homography,np.asarray([0,h2,1]))
  bottom_right = np.dot(homography,np.asarray([w2,h2,1]))

  if DEBUG:
    print top_left
    print top_right
    print bottom_left
    print bottom_right
  
  #normalize
  top_left = top_left/top_left[2]
  top_right = top_right/top_right[2]
  bottom_left = bottom_left/bottom_left[2]
  bottom_right = bottom_right/bottom_right[2]

  if DEBUG:
    print np.int32(top_left)
    print np.int32(top_right)
    print np.int32(bottom_left)
    print np.int32(bottom_right)
  
  pano_left = int(min(top_left[0], bottom_left[0], 0))
  pano_right = int(max(top_right[0], bottom_right[0], w1))
  W = pano_right - pano_left
  
  pano_top = int(min(top_left[1], top_right[1], 0))
  pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
  H = pano_bottom - pano_top
  
  size = (W, H)
  
  if DEBUG:
    print 'Panodimensions'
    print pano_top
    print pano_bottom
  
  # offset of first image relative to panorama
  X = int(min(top_left[0], bottom_left[0], 0))
  Y = int(min(top_left[1], top_right[1], 0))
  offset = (-X, -Y)
  
  if DEBUG:
    print 'Calculated size:'
    print size
    print 'Calculated offset:'
    print offset
      
  ## Update the homography to shift by the offset
  # does offset need to be remapped to old coord space?
  # print homography
  # homography[0:2,2] += offset

  return (size, offset)


## 4. Combine images into a panorama. [4] --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):

  ## TODO: Combine the two images into one.
  ## TODO: (Overwrite the following 5 lines with your answer.)
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  
  (ox, oy) = offset
  
  translation = np.matrix([
    [1.0, 0.0, ox],
    [0, 1.0, oy],
    [0.0, 0.0, 1.0]
  ])
  
  if DEBUG:
    print homography
  homography = translation * homography
  # print homography
  
  # draw the transformed image2
  cv2.warpPerspective(image2, homography, size, panorama)
  
  panorama[oy:h1+oy, ox:ox+w1] = image1  
  # panorama[:h1, :w1] = image1  

  ## TODO: Draw the common feature keypoints.

  return panorama

def merge_images_translation(image1, image2, offset):

  ## Put images side-by-side into 'image'.
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  (ox, oy) = offset
  ox = int(ox)
  oy = int(oy)
  oy = 0
  
  image = np.zeros((h1+oy, w1+ox, 3), np.uint8)
  
  image[:h1, :w1] = image1
  image[:h2, ox:ox+w2] = image2
  
  return image


##---- No need to change anything below this point. ----------------------


def match_flann(img1,img2,des1, des2, r_threshold = 0.12): #originally 0.12
  # 'Finds strong corresponding features in the two given vectors.'
  # ## Adapted from <http://stackoverflow.com/a/8311498/72470>.

  # ## Build a kd-tree from the second feature vector.
  # FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
  # flann = cv2.flann_Index(desc2, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 4})

  # ## For each feature in desc1, find the two closest ones in desc2.
  # (idx2, dist) = flann.knnSearch(desc1, 2, params={}) # bug: need empty {}

  # ## Create a mask that indicates if the first-found item is sufficiently
  # ## closer than the second-found, to check if the match is robust.
  # mask = dist[:,0] / dist[:,1] < r_threshold
  
  # ## Only return robust feature pairs.
  # idx1  = np.arange(len(desc1))
  # pairs = np.int32(zip(idx1, idx2[:,0]))
  # return pairs[mask]
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)   # or pass empty dictionary

  flann = cv2.FlannBasedMatcher(index_params,search_params)

  matches = flann.knnMatch(des1,des2,k=2) #returns 2 closest matches
  good = []
  for m,n in matches:
    if m.distance < 0.75 * n.distance:
      good.append([m])

  matchesMask = [[0,0] for i in xrange(len(matches))]
  print 'good matches '
  raw_input()
  print good
  raw_input()
# ratio test as per Lowe's paper
  for i,(m,n) in enumerate(matches):
      if m.distance < 0.7*n.distance:
          matchesMask[i]=[1,0] #take first descriptor

  draw_params = dict(matchColor = (0,255,0),
                     singlePointColor = (255,0,0),
                     matchesMask = matchesMask,
                     flags = 0)

  img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)

  plt.imshow(img3,),plt.show()
  #write kp and match indexes to files
  m = []
  idx1 = np.arange(len(des1))
  idx2 = []
  for a in good:
    if a[0].trainIdx > min(len(des1)-1,len(des2)-1):
      print 'stoppp ',a[0].trainIdx
    else:
      idx2.append(a[0].trainIdx)
  print idx2
  raw_input()
  m = np.int32(zip(idx1,idx2))
  return m

def draw_correspondences(image1, image2, points1, points2):
  'Connects corresponding features in the two images using yellow lines.'

  ## Put images side-by-side into 'image'.
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
  image[:h1, :w1] = image1
  image[:h2, w1:w1+w2] = image2
  
  ## Draw yellow lines connecting corresponding features.
  for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
    cv2.line(image, (x1, y1), (x2+w1, y2), (2555, 0, 255), lineType=cv2.LINE_AA)

  return image

def pano(image1,image2):
  ## Detect features and compute descriptors.
  (keypoints1, descriptors1) = extract_features(image1, algorithm='SURF')
  (keypoints2, descriptors2) = extract_features(image2, algorithm='SURF')
  print len(keypoints1), "features detected in image1"
  print len(keypoints2), "features detected in image2"
  
  ## Find corresponding features.
  (points1, points2) = find_correspondences(image1,image2,keypoints1, descriptors1, keypoints2, descriptors2)
  #print len(points1), "features matched"
  
  ## Visualise corresponding features.
  correspondences = draw_correspondences(image1, image2, points1, points2)
  cv2.imwrite("LR/correspondences.jpg", correspondences)
  #cv2.imshow('correspondences',correspondences)
  #print 'Wrote correspondences.jpg'
  
  ## Find homography between the views.
  (homography, _) = cv2.findHomography(points2, points1)
  
  ## Calculate size and offset of merged panorama.
  (size, offset) = calculate_size(image1.shape, image2.shape, homography)
  ## Finally combine images into a panorama.
  panorama = merge_images(image1, image2, homography, size, offset, (points1, points2))
  #print 'Wrote panorama.jpg'
  #raw_input()
  return panorama

def drawlines(img1,img2,lines,points1,points2):
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,points1,points2):
             color = tuple(np.random.randint(0,255,3).tolist())
             x0,y0 = map(int, [0, -r[2]/r[1] ])
             x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
             img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
             img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
             img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

#P1 is the first projection matrix
#P2 is the second projection matrix
def triangulate(P1,P2,pts1,pts2):
  global match
  print "global match:",len(match)
  raw_input()
  A = np.zeros((4,4))
  U,W,Vt,pts4D = np.array((4,4)),np.array((4,1)),np.array((4,4)),np.zeros((4,len(match)))
  numpts = len(pts1[0])
  pts = [pts1,pts2]
  P = [P1,P2]
  print 'Projection matrices:\n'
  print P1
  #raw_input()
  print P2
  #raw_input()
  #P1,P2 - projection matrices - 3 X 4
  #pts1,pts2 - matching correspondences - x and x' - 41 X 2
  print 'My function starts here'
  #raw_input()
  for i in xrange(numpts):
    for j in xrange(2):
      x = pts[j][0][i] #take each point(x,y) in pts1 or pts2 where 1/2 given by j 
      y = pts[j][1][i]
      print 'x: %f\ty:%f'%(x,y)
        
      for k in xrange(4):
        A[j*2+0][k] = x * P[j][2][k] - P[j][0][k]
        A[j*2+1][k] = y * P[j][2][k] - P[j][1][k]
        # print 'A[j*2+0][k]: %f\tA[j*2+1][k]:%f'%(A[j*2+0][k],A[j*2+1][k])
        
    #solve A using SVD
    U,W,Vt = np.linalg.svd(A)
    # print 'V(3,0):%f\tV(3,1):%f\tV(3,2):%f\tV(3,3):%f' %(Vt[3][0],Vt[3][1],Vt[3][2],Vt[3][3])
    # print 'A-USV\': \n',(A-U*W*Vt)/np.linalg.norm(A)
    #raw_input()
    pts4D[0][i] = Vt[3][0]
    pts4D[1][i] = Vt[3][1]
    pts4D[2][i] = Vt[3][2]
    pts4D[3][i] = Vt[3][3]
  #raw_input()
  return pts4D

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
                           self.calibrated_pair.stereo_bm_preset, 3,
                           self.set_bm_preset)
        cv2.createTrackbar("ndis", self.window_name,
                           self.calibrated_pair.search_range, 160,
                           self.set_search_range)
        cv2.createTrackbar("winsize", self.window_name,
                           self.calibrated_pair.window_size, 21,
                           self.set_window_size)
        #: (left, right) image pair to find disparity between
        self.pair = image_pair
        self.tune_pair(image_pair)
    def set_bm_preset(self, preset):
        """Set ``search_range`` and update disparity image."""
        try:
            self.calibrated_pair.stereo_bm_preset = preset
        except InvalidBMPreset:
            return
        self.update_disparity_map()
    def set_search_range(self, search_range):
        """Set ``search_range`` and update disparity image."""
        try:
            self.calibrated_pair.search_range = search_range
        except InvalidSearchRange:
            return
        self.update_disparity_map()
    def set_window_size(self, window_size):
        """Set ``window_size`` and update disparity image."""
        try:
            self.calibrated_pair.window_size = window_size
        except InvalidWindowSize:
            return
        self.update_disparity_map()
    def update_disparity_map(self):
        """Update disparity map in GUI."""
        disparity = self.calibrated_pair.compute_disparity(self.pair)
        cv2.imshow(self.window_name, disparity / 255.)
        cv2.waitKey()
    def tune_pair(self, pair):
        """Tune a pair of images."""
        self.pair = pair
        self.update_disparity_map()

def find_files(directory):
  import os
  files = []
  print 'dir: ',directory
  # raw_input()
  for file_name in os.listdir("/home/shriya/FYP/TRY STUFF/"+directory):
      if file_name.endswith(".jpg"):
        print(file_name)
        files.append(file_name)
  return files

if __name__ == "__main__":
  im1 = cv2.imread("Image1.jpg")
  im2 = cv2.imread("Image2.jpg")
  (kp1, d1) = extract_features(im1, algorithm='SURF')
  (kp2, d2) = extract_features(im2, algorithm='SURF')
  (points1, points2) = find_correspondences(im1,im2,kp1, d1, kp2, d2)
  correspondences = draw_correspondences(im1, im2, points1, points2)
  cv2.imwrite("LR/correspondences.jpg", correspondences)
  F, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_RANSAC)
  K,dist_coefs = calibrate.calib()   
  print 'Camera matrix:\n',K
  print 'F\n',F
  raw_input()     
  E = np.dot(np.dot(K.T,F),K)
  #F error
  pt1 = np.array([[points1[5][0]], [points1[5][1]], [1]])#3X1
  pt2 = np.array([[points2[5][0], points2[5][1], 1]])#1X3
  print 'fund matr error: ',np.dot(np.dot(pt2,F),pt1)
  #raw_input()
  #find translation vector and rotation matrix from E using svd, V is V'
  U,W,Vt = np.linalg.svd(E)
  print 'len of Vt: ',len(Vt),len(Vt[0])
  #raw_input()
  print 'W:diag(110):\n'
  print W[0],W[1]
  #raw_input()
  
  W = np.reshape([0,-1,0,1,0,0,0,0,1],(3,3))
  print 'W:',len(W),len(W[0]),'\nU: ',len(U),len(U[0]),'\nVt: ',len(Vt),len(Vt[0])
  #raw_input()
  R = U*W*Vt # 2 solns here - UW'V'
  t = U[:,2] # last column
  print 'Camera matrix\n'
  print K
  #raw_input()
  print 'R: \n',R,'\n',len(R),len(R[0])
  #raw_input()
  print 't: ',len(t)
  #raw_input()
  #Projection matrices - P1 at origin and P2 from (R,t)
  P1 = np.array([ [ 1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0] ])
  P2 = np.hstack((R,t.reshape(3,1)))
  print 'P1 as sent to tri_in built:\n',P1,len(P1),len(P1[0])
  #raw_input()
  print 'Points1:\n',points1
  #raw_input()
  points1t = points1.T
  points2t = points2.T
  print 'Built-in Triangulate Function\n'
  #raw_input()
  res = cv2.triangulatePoints(P1,P2,points1t[:2],points2t[:2])
  #raw_input()
  print 'res:\n',res,len(res),len(res[0])
  # raw_input()
  print 'My Triangulate Function\n'
  res1 = triangulate(P1,P2,points1t[:2],points2t[:2])
  res1 /= res[3]
  x1 = np.dot(P1[:3],res1)
  x2 = np.dot(P2[:3],res1)
  x1 /= x1[2]
  x2 /= x2[2]
  # print res1,len(res1),len(res1[0])
  #get px,py,pz for pcl
  #delete the last row - all 1s
  res1 = np.delete(res1,(3),axis=0)
  resT = np.transpose(res1)
  np.set_printoptions(threshold=np.nan) #prevent truncation of numpy array
  print resT
  np.save("val.txt",resT)
  raw_input()
  # #create disparity map
  # """Let user tune all images in the input folder and report chosen values."""
  # parser = argparse.ArgumentParser(description="Read images taken from a "
  #                                "calibrated stereo pair, compute "
  #                                "disparity maps from them and show them "
  #                                "interactively to the user, allowing the "
  #                                "user to tune the stereo block matcher "
  #                                "settings in the GUI.")
  # parser.add_argument("image_folder",
  #                   help="Directory where input images are stored.")
  # args = parser.parse_args()
  # print 'booyah',args
  # raw_input()
  # #already calibrated in K  
  # calibration = K
  # print 'calib done'
  # #write find_files function
  # input_files = find_files(args.image_folder)
  # calibrated_pair = CalibratedPair(None, calibration)
  # image_pair = [cv2.imread(image) for image in input_files[:2]]
  # rectified_pair = calibration.rectify(image_pair)
  # print 'rectified!'
  # raw_input()
  # tuner = StereoBMTuner(calibrated_pair, rectified_pair)
  # chosen_arguments = []
  # while input_files:
  #   image_pair = [cv2.imread(image) for image in input_files[:2]]
  #   rectified_pair = calibration.rectify(image_pair)
  #   tuner.tune_pair(rectified_pair)
  #   chosen_arguments.append((calibrated_pair.stereo_bm_preset,
  #                            calibrated_pair.search_range,
  #                            calibrated_pair.window_size))
  #   input_files = input_files[2:]
  # stereo_bm_presets, search_ranges, window_sizes = [], [], []
  # for preset, search_range, size in chosen_arguments:
  #   stereo_bm_presets.append(preset)
  #   search_ranges.append(search_range)
  #   window_sizes.append(size)
  # for name, values in (("Stereo BM presets", stereo_bm_presets),
  #                    ("Search ranges", search_ranges),
  #                    ("Window sizes", window_sizes)):
  #   report_variable(name, values)
  #   print()
