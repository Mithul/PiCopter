import cv2, numpy as np
import math
import argparse as ap
import time
from threading import Thread

#Extract SURF keypoints and descriptors from an image
def extract_features(image1,image2, surfThreshold=1000, algorithm='SURF'):
  # Convert image to grayscale (for SURF detector).
  try:
    # print 'Type of image for cvtColor: ',type(image1)
    image_gs1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
  except TypeError:
    return
  try:
    # print 'Type of image for cvtColor: ',type(image2)
    image_gs2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
  except TypeError:
    return

  
  # Detect SURF features and compute descriptors.
  detector = cv2.xfeatures2d.SURF_create()
  (keypoints1,descriptors1) = detector.detectAndCompute(image_gs1,None)
  (keypoints2,descriptors2) = detector.detectAndCompute(image_gs2,None)
  #return keypoints as numpy arrays
  keypoints1,keypoints2 = np.float32([kp.pt for kp in keypoints1]), np.float32([kp.pt for kp in keypoints2])
  return (keypoints1, descriptors1,keypoints2, descriptors2)


# Find corresponding features between the images
def find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2):
  # Find corresponding features.
  matches = match_flann(descriptors1, descriptors2)
  points1 = np.float32([keypoints1[i] for (_, i) in matches])
  points2 = np.float32([keypoints2[i] for (i, _) in matches])
  return (points1, points2)


#Calculate the size and offset of the stitched panorama
def calculate_size(size_image1, size_image2, homography):
  
  (h1, w1) = size_image1[:2]
  (h2, w2) = size_image2[:2]
  
  #remap the coordinates of the projected image onto the panorama image space
  top_left = np.dot(homography,np.asarray([0,0,1]))
  top_right = np.dot(homography,np.asarray([w2,0,1]))
  bottom_left = np.dot(homography,np.asarray([0,h2,1]))
  bottom_right = np.dot(homography,np.asarray([w2,h2,1]))

  #normalize
  top_left = top_left/top_left[2]
  top_right = top_right/top_right[2]
  bottom_left = bottom_left/bottom_left[2]
  bottom_right = bottom_right/bottom_right[2]

  pano_left = int(min(top_left[0], bottom_left[0], 0))
  pano_right = int(max(top_right[0], bottom_right[0], w1))
  W = pano_right - pano_left
  
  pano_top = int(min(top_left[1], top_right[1], 0))
  pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
  H = pano_bottom - pano_top
  
  size = (W, H)
  
  # offset of first image relative to panorama
  X = int(min(top_left[0], bottom_left[0], 0))
  Y = int(min(top_left[1], top_right[1], 0))
  offset = (-X, -Y)
  return (size, offset)

def blend(panorama,img1,img2,ox,oy):
    p1,p2 = panorama.shape[:2]
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    i1 = panorama[0:p1,w1-100:w1]
    i2 = panorama[0:p1,w1:w1+100]
    print i1.shape[:2]
    print i2.shape[:2]
    panorama[0:p1,w1:w1+100] = cv2.addWeighted(i1,0.1,i2,0.9,0)
    return panorama

#Combine images into a panorama
def merge_images(image1, image2, homography, size, offset, keypoints):

  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  
  (ox, oy) = offset
  
  translation = np.matrix([
    [1.0, 0.0, ox],
    [0, 1.0, oy],
    [0.0, 0.0, 1.0]
  ])
  
  homography = translation * homography
  
  # draw the transformed image2
  cv2.warpPerspective(image2, homography, size, panorama)
  
  panorama[oy:h1+oy, ox:ox+w1] = image1
  #crop panorama -- remove this for vertical images like 'door' test set
  height, width = panorama.shape[:2]
  crop_h = int(0.05 * height)
  crop_w = int(0.015 * width)
  panorama = panorama[crop_h:-crop_h, crop_w:-crop_w]
  panorama = panorama[int(oy*0.7):,:]

  #blend
  panorama = blend(panorama,image1,image2,ox,oy)
  return panorama


def match_flann(des1, des2,ratio=0.75):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(des1, des2, 2)
    matches = []
 
    # loop over the raw matches
    for m in rawMatches:
      # ensure the distance is within a certain ratio of each
      # other (i.e. Lowe's ratio test)
      if len(m) == 2 and m[0].distance < m[1].distance * ratio:
        matches.append((m[0].trainIdx, m[0].queryIdx))
    return matches

  
def draw_correspondences(image1, image2, points1, points2):
  'Connects corresponding features in the two images using yellow lines.'

  # Put images side-by-side into 'image'.
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
  image[:h1, :w1] = image1
  image[:h2, w1:w1+w2] = image2
  
  # Draw yellow lines connecting corresponding features.
  for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
    cv2.line(image, (x1, y1), (x2+w1, y2), (2555, 0, 255), lineType=cv2.LINE_AA)

  return image
def pano(images,i):
    # Detect features and compute descriptors.
  # print 'image len: ',len(images),type(images[0])
  if (i+1) <= (len(images)-1):
    try:
      # print 'Image name: ',images[i]
      image1 = cv2.imread(images[i])
    except TypeError:
      image1 = images[i]
    try:
      # print 'Image name: ',images[i+1]
      image2 = cv2.imread(images[i+1])
    except TypeError:
      image2 = images[i+1]

  else:
    # print 'OOB'
    return  
  (keypoints1, descriptors1,keypoints2, descriptors2) = extract_features(image1,image2)
  print len(keypoints1), "features detected in image1"
  print len(keypoints2), "features detected in image2"
  
  # Find corresponding features.
  (points1, points2) = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
  print len(points1), "features matched"
  
  # Visualise corresponding features.
  correspondences = draw_correspondences(image1, image2, points1, points2)
  cv2.imwrite("door/correspondences.jpg", correspondences)
  print 'Wrote correspondences.jpg'
  
  try:
  # Find homography between the views.
    (homography, _) = cv2.findHomography(points2,points1,cv2.RANSAC,4)
  except Exception:
    print 'Not enough matches!'
    return -1
  # Calculate size and offset of merged panorama.
  (size, offset) = calculate_size(image1.shape, image2.shape, homography)
  
  # Finally combine images into a panorama.
  images[i] = merge_images(image1, image2, homography, size, offset, (points1, points2))
  if(len(images) == 2):#final panorama
    filename = "door/pano_ multi_final"+str(i)+".jpg"
    print 'pano size: ',images[0].shape[:2]
  else:
    filename = "door/pano_multi"+str(i)+".jpg"
  cv2.imwrite(filename,images[i])
  
if __name__ == "__main__":
  import time
  st = time.time()
  images = ["door/door1.jpg","door/door2.jpg","door/door3.jpg","door/door4.jpg","door/door5.jpg","door/door6.jpg","door/door7.jpg"]
  # images = ["door/door1.jpg","door/door2.jpg","door/door3.jpg","door/door4.jpg"]
  #images = ["bridge/01.jpg","bridge/02.jpg","bridge/03.jpg","bridge/04.jpg","bridge/05.jpg","bridge/06.jpg","bridge/07.jpg","bridge/08.jpg"]
  n = len(images)
  val = int(math.ceil(math.log(len(images),2)))+1
  for q in range(val):
    for i in xrange(0,len(images),1):
          print 'i: ',i,' len: ',len(images)
          t = Thread(target=pano, args=(images,i))
          t.start()
          t.join()
          if i+1 <= len(images)-1:
            del images[i+1]
  et = time.time()
  print 'time: ',et-st,'s' 
   