import numpy as np

def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    with open(filename, 'r') as f:
        calib_file = f.read()
    
    for line in calib_file.split('\n'):
        if not line.strip():    # removes leading and trailing whitespace -> if empty skip
            continue
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        
        calib[key] = pose
    
    return calib

def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    with open(filename, 'r') as f:
        poses_file = f.read()
    
    poses = []
    
    Tr = calibration["Tr"]  # extrinsic calibration matrix from velodyne to camera
    for line in poses_file.split('\n'):
        if not line.strip():    # removes leading and trailing whitespace -> if empty skip
            continue
        values = [float(v) for v in line.strip().split()]
        
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        
        # right to left: transforms velodyne to camera coordinates and applies pose transformation (in sensor coordinates), the inverse transformation is applied again to bring back to velodyne coords
        poses.append( (np.linalg.inv(Tr) @ pose @ Tr) if not np.array_equal(Tr, np.eye(N=Tr.shape[0], M=Tr.shape[1])) else pose)
        poses 
    return poses