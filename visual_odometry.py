import numpy as np
import matplotlib.pyplot as plt

import sys
ros_path = '/opt/ros/lunar/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2

sys.path.append(ros_path)

# Camera calibration matrix
K = np.array([[7.215377000000e+02,0.000000000000e+00,6.095593000000e+02],
              [0.000000000000e+00,7.215377000000e+02,1.728540000000e+02],
              [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])

cv2.__version__
t = 1
# Load ground truth data

gt = np.loadtxt('ground-truth.txt').reshape(-1,3,4)

origin = np.append(np.eye(3),np.zeros((3,1)),axis=1)
originx4 = np.append(origin,np.array([0,0,0,1])).reshape((4,4))

# Detect SIFT features
def detectSIFTfeatures(i):
    img = cv2.imread('images/{:06d}.png'.format(i),0)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points = sift.detect(img)
    pts1 = np.array([x.pt for x in key_points],dtype=np.float32)
    return pts1

# Track the detected features with Lukas-kanade tracker
def tracker(i1, i2):
    img1 = cv2.imread('images/{:06d}.png'.format(i1),0)
    img2 = cv2.imread('images/{:06d}.png'.format(i2),0)
    
    pts1 = detectSIFTfeatures(i1)
    
    pts2, status, _ = cv2.calcOpticalFlowPyrLK(img1,img2,pts1,None)
    
    status = status.reshape(status.shape[0])
    pts1 = pts1[status == 1]
    pts2 = pts2[status == 1]
    return pts1, pts2

def findEandrecoverRt(pts1, pts2):
    # Calculating F, E, R and t
    
    # First, we normalize the data
    T_inv = np.array([[621.0,0,621],
                 [0,-375/2,375/2],
                 [0,0,1]])
    T = np.linalg.inv(T_inv)
    # Test the normalized data
    # T @ np.array([0, 0, 1])
    
    # Get the normalized corresponding SIFT  points
    pts1_h = np.hstack((pts1,np.ones((np.size(pts1,0),1))))
    pts1_normalized_all = np.matmul(T,pts1_h.T)

    pts2_h = np.hstack((pts2,np.ones((np.size(pts2,0),1))))
    pts2_normalized_all = np.matmul(T,pts2_h.T)
    
    # Decide number of trails
    p = 0.99
    e = 0.3
    s = 8
#     trails = int(np.log(1-p)/(np.log(1-(1-e)**s))) #2
    trails = 0
    inliers = []
    
    # RANSAC
    
    # Initialize few values for RANSAC
    err_best = 500 # set some random big value
    err = 0
    F_best = np.ones((3,3))
    F_cap_best = np.ones((3,3))
    F_cap = np.ones((3,3))
    
    
    index = np.empty((trails,0))
    ind = 0
        
    for asd in range(trails):
        # Sample s points
        ind = np.arange(len(pts1))
        np.random.shuffle(ind)
        ind = ind[:s]
        
        # Kronoker formulation to get F_cap
        Kr = np.array([pts2_h[ind,0]*pts1_h[ind,0], 
                       pts2_h[ind,0]*pts1_h[ind,1],
                       pts2_h[ind,0], 
                       pts2_h[ind,1]*pts1_h[ind,0], 
                       pts2_h[ind,1]*pts1_h[ind,1], 
                       pts2_h[ind,1], 
                       pts1_h[ind,0], 
                       pts1_h[ind,1], 
                       np.ones((s,))]).T

        # Use SVD to get the F_cap
        u, d, v = np.linalg.svd(Kr) # Make sure you get (9,9) for v.shape


        #Take the last row. we have to take last column, but SVD gives V transpose.
        f = v.T[:,-1:]
        f = f.flatten()
        F_cap = f.reshape((3,3))        

        # Enforce F_cap to rank 2 using SVD
        u, d, v = np.linalg.svd(F_cap)
        d[2] = 0
        F_cap = u.dot(np.diag(d).dot(v))            
        
        for i in range(len(pts1)):
            err = ((pts2_normalized_all[:,i].T).dot(F_cap.dot(pts1_normalized_all[:,i])))**2
            if(err<0.001):
                inlier_index[asd] = np.append(inlier_index[asd],i)

    Kr1 = np.array([pts2_h[inliers,0]*pts1_h[inliers,0], 
                       pts2_h[inliers,0]*pts1_h[inliers,1],
                       pts2_h[inliers,0], 
                       pts2_h[inliers,1]*pts1_h[inliers,0], 
                       pts2_h[inliers,1]*pts1_h[inliers,0], 
                       pts2_h[inliers,1], 
                       pts1_h[inliers,1], 
                       pts1_h[inliers,1], 
                       np.ones((len(inliers),))]).T

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    E = K.T.dot(F.dot(K))
    
    # Decompose R, t
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return E, F, F_cap, pts1_normalized_all, pts2_normalized_all, ind, R, t

# Compute absolute transformation
def getAbsState(i1, i2):
    # Detect SIFT features and track the detected features
    pts1, pts2 = tracker(i1, i2) # Lukas-kanade trackerer
    

    # Estimate Essential matrix using RANSAC scheme
    E, F, F_cap, pts1_normalized_all, pts2_normalized_all, ind, R, t = findEandrecoverRt(pts1, pts2)
    # Scale using ground truth
    scale = np.matmul(np.linalg.inv(np.append(gt[i1],np.array([0,0,0,1])).reshape((4,4))), (np.append(gt[i2],np.array([0,0,0,1])).reshape((4,4))))

    t = t*np.linalg.norm(scale[:3,-1])

    H = np.append(R.T,-t,axis=1)
    H = np.around(H, decimals=5)
    H = np.append(H,np.array([0,0,0,1])).reshape((4,4))
    return H



def writeYourResult(H_rel_pred, H_abs_pred, N, filename):	
	init_line_to_file = "1.000000e+00 -1.822835e-10 5.241111e-10 -5.551115e-17 -1.822835e-10 9.999999e-01 -5.072855e-10 -3.330669e-16 5.241111e-10 -5.072855e-10 9.999999e-01 2.220446e-16"
	with open (filename, 'a') as f: f.write(init_line_to_file + "\n")

	for i in range(N):
	    H = getAbsState(i,i+1)
	    H_abs_pred = np.matmul(H_abs_pred,H)
	    H_abs_pred = H_abs_pred/H_abs_pred[-1,-1]
	    
	    # pre-process H and write it to a file
	    processH = H_abs_pred[:3]
	    processH = np.around(processH, decimals=5)
	    processH = processH.reshape((1,12))
	    processH = processH.tolist()
	    processH = str(processH).replace('[', '').replace(']', '').replace(',', '')

	    with open ('your-result.txt', 'a') as f: f.write(processH + "\n")
	    print("End of pair:{}".format(i))


if __name__ == "__main__":	
	writefile = 'your-result.txt'
	N = 400
	H_rel_pred = np.ones((N,3,4))
	H_abs_pred = np.append(origin,np.array([0,0,0,1])).reshape((4,4))

	writeYourResult(H_rel_pred, H_abs_pred, N,  writefile)


	# In[499]:


	# # Load ground truth data

	points = np.loadtxt("q2/ground-truth.txt").reshape(-1, 3, 4)
	# points1_1 = np.loadtxt("your-result-LK-1.txt").reshape(-1, 3, 4) # Lukas-kanade tracker
	# points1_2 = np.loadtxt("your-result-LK-2.txt").reshape(-1, 3, 4) # Lukas-kanade tracker
	# points2 = np.loadtxt("your-result-flann.txt").reshape(-1, 3, 4) # Flann based tracker
	points1 = np.loadtxt("your-result.txt").reshape(-1, 3, 4)

	fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
	# fig = plt.figure()

	# plt.subplot(311)
	plt.title("LK tracker [t = 2]")
	plt.scatter(points[:,0,3],points[:,2,3], s=2)
	plt.scatter(points1[:,0,3],points1[:,2,3], s=2)

	plt.show()