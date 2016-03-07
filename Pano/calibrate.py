#!/usr/bin/env python

import numpy as np
import cv2
import os
from common import splitfn

USAGE = '''
USAGE: calib.py [--save <filename>] [--debug <output path>] [--square_size] [<image mask>]
'''
def draw(img, corners, imgpts):
         imgpts = np.int32(imgpts).reshape(-1,2)
         # draw ground floor in green
         mask1,mask2,mask3 = img,img,img
         cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
         cv2.add(img,mask1)
     
         # draw pillars in blue color
         for i,j in zip(range(4),range(4,8)):
             cv2.line(mask2, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),3)
             cv2.add(img,mask2)
        # draw top layer in red color
         cv2.drawContours(mask3, [imgpts[4:]],-1,(255,0,0),3)
         cv2.add(img,mask3)
         # cv2.imshow('img',img)
         # cv2.waitKey(0)
         # cv2.destroyAllWindows()

         return img
# def draw(img,corners,image_points):
#   corner = tuple(corners[0].ravel())
#   mask1,mask2,mask3 = img,img,img
#   cv2.line(mask1,corner,tuple(image_points[0].ravel()),(255,0,0),5)
#   cv2.add(img,mask1)
#   #cv2.imshow('img',img)
# # raw_input() 
#   cv2.line(mask2,corner,tuple(image_points[1].ravel()),(0,255,0),5)
#   cv2.add(img,mask2)
#   cv2.line(mask3,corner,tuple(image_points[2].ravel()),(0,0,255),5)
#   cv2.add(img,mask3)
#   cv2.imshow('img',img)
#         cv2.waitKey(0)
#   # cv2.destroyAllWindows()
#         return img

def calib():
    import sys, getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['save=', 'debug=', 'square_size='])
    args = dict(args)
    img_mask = 'calib1/left*.jpg'
    img_names = glob(img_mask)
    debug_dir = args.get('--debug')
    square_size = float(args.get('--square_size', 1.0))

    pattern_size = (9, 6)#10X7
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    print 'image names: ',img_names
    for fn in img_names:
        print 'processing %s...' % fn,
        img = cv2.imread(fn, 0)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        path, name, ext = splitfn(fn)
        #cv2.imshow('corners',vis)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
        cv2.imshow('corners',vis)
        cv2.waitKey(25)
        cv2.destroyAllWindows()
        cv2.imwrite('%s/%s_chess.bmp' % (debug_dir, name), vis)
        if not found:
            print 'chessboard not found'
            continue
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        print 'ok'

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h),None,None)
    print 'dist coefs: ',dist_coefs,len(dist_coefs),len(dist_coefs[0])
    raw_input()
    print w/2-0.5,h/2-0.5
    # axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3);
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
    for fn in img_names:
        print 'processing %s...' % fn,
        img = cv2.imread(fn, 0)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        print 'corners: ',len(corners),len(corners[0])
        if found:
            term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        pattern_points = np.zeros((w*h, 1, 3), np.float32) 
        pattern_points[:,:,:2] = np.mgrid[0:w,  0:h].T.reshape(-1,1,2)
        # print 'patt points: ',len(pattern_points),len(pattern_points[0])
        # print 'corners points: ',len(corners),len(corners[0])
        # raw_input()
        # rv,tv,inliers = cv2.solvePnPRansac(pattern_points,corners,camera_matrix,dist_coefs)
        # img_points,jac = cv2.projectPoints(axis,rv,tv,camera_matrix,dist_coefs)
        # img = draw(img,corners,img_points)    
            
        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            cv2.imwrite('%s/%s_chess.bmp' % (debug_dir, name), vis)
            if not found:
                print 'chessboard not found'
        continue
       # img_points.append(corners.reshape(-1, 2))
       # obj_points.append(pattern_points)
       
        print 'ok'
    

    return camera_matrix,dist_coefs
