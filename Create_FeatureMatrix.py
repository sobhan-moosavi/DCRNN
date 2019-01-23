'''
Created on January 2019
This is code is to create feature matrix for a trajectory based on idea in DCRNN paper. 
Basic Features: Speed, Speed-Change, Acceleration, Acceleration-Change, AngularSpeed, AngularSpeed-Change, RPM, RPM-Change, 
    Heading, Heading-Change, AccelX, AccelX-Change, AccelY, AccelY-Change, AccelZ, AccelZ-Change
Statistical Features: Mean, Std, Min, Max, Q25, Q50, Q75
@author: Sobhan Moosavi and Pravar Mahajan
'''
import cPickle
import numpy as np
import math
from scipy import stats
import time
import progressbar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', type=str, default='SpdAclRp')  # using suffix, you can specify set of basic features to be used to create feature map
args = parser.parse_args()

# data point of a trajectory
class point:    
    time = 0
    lat = 0
    lng = 0
    speed = 0
    acceleration = 0
    rpm = 0
    heading = 0
    accelX = 0
    accelY = 0
    accelZ = 0
    
    def __init__(self, time, lat, lng, speed, acceleration, rpm, heading, accelX, accelY, accelZ):
        self.time = time
        self.lat = lat
        self.lng = lng
        self.speed = speed/3.6 #converting to m/s
        self.acceleration = acceleration #reported in m/s^2
        self.rpm = rpm #engine RPM
        self.heading = heading #between 0 to 359
        self.accelX = accelX
        self.accelY = accelY
        self.accelZ = accelZ
        

# calculating angular displacement 
def returnAngularDisplacement(fLat, fLon, sLat, sLon):
    #Inspired by: https://www.quora.com/How-do-I-convert-radians-per-second-to-meters-per-second
    
    fLat = np.radians(float(fLat))
    fLon = np.radians(float(fLon))
    sLat = np.radians(float(sLat))
    sLon = np.radians(float(sLon))
    
    dis = np.sqrt((fLat-sLat)**2 + (fLon-sLon)**2)
    return dis


# calculating haversine distance
def haversineDistance(aLat, aLng, bLat, bLng, metric='mi'):
    #From degree to radian
    fLat = math.radians(aLat)
    fLon = math.radians(aLng)
    sLat = math.radians(bLat)
    sLon = math.radians(bLng)
           
    R = 3958.7564 #mi
    
    if metric == 'meters':
        R = 6371000.0 #meters
    elif metric == 'km':
        R = 6371.0 #km
    dLon = sLon - fLon
    dLat = sLat - fLat
    a = math.sin(dLat/2.0)**2 + (math.cos(fLat) * math.cos(sLat) * math.pow(math.sin(dLon/2.0), 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
       
    return R * c
    

def generateStatisticalFeatureMatrix(L1=256, L2=4):
    #load trajectories
    trajectories = {}
    filename = 'data/RandomSample_5_10.csv'
    with open(filename, 'r') as f:
        lines = f.readlines()
        ct = ''
        cd = ''
        tj = []
        bar = progressbar.ProgressBar()        
        for ln in bar(lines):
            pts = ln.replace('\r\n','').split(',')
            if pts[1] != ct:
                if ct == "" and pts[0]=="Driver":
                    continue
                if len(tj) >0:
                    trajectories[cd+"|"+ct] = tj
                tj = []
                #Driver,ID,Time,Lat,Lon,Speed,Acceleration,RPM,Heading,AccelX,AccelY,AccelZ
                tj.append(point(int(pts[2]), float(pts[3]), float(pts[4]), (-1.0 if len(pts[5])==0 else float(pts[5])), (0.0 if len(pts[6])==0 else float(pts[6])), \
                    int(pts[7]), int(pts[8]), float(pts[9]), float(pts[10]), float(pts[11])))
                ct = pts[1]
                cd = pts[0]
            else:
                tj.append(point(int(pts[2]), float(pts[3]), float(pts[4]), (-1.0 if len(pts[5])==0 else float(pts[5])), (0.0 if len(pts[6])==0 else float(pts[6])), \
                    int(pts[7]), int(pts[8]), float(pts[9]), float(pts[10]), float(pts[11])))                
                
        trajectories[cd+"|"+ct] = tj
    print('Raw Trajectory Data is loaded! |Trajectories|:' + str(len(trajectories)))
    
    #Generate Basic Features for each trajectory
    basicFeatures = {}
    bar = progressbar.ProgressBar()
    for t in bar(trajectories):
        points = trajectories[t]
        traj = []
        lastAngleSpeed = -1000
        lastSpeed = lastAccel = 0
        for i in range(1, len(points)):
            tuple = []
            if 'All' in args.suffix or 'Spd' in args.suffix:  # to include speed in basic features 
                tuple.append((lastSpeed if points[i].speed==-1.0 else points[i].speed))
                lastSpeed = (lastSpeed if points[i].speed==-1.0 else points[i].speed)
                
            if 'All' in args.suffix or 'Acl' in args.suffix:  # to include acceleration in basic features 
                tuple.append(points[i].acceleration)
                tuple.append(abs(points[i].acceleration - points[i-1].acceleration))
            
            # calculate gps-based speed and acceleration
            if '_gps' in args.suffix:      # to include gps related features (speed and acceleration) in basic features 
                speed_gps = haversineDistance(points[i-1].lat, points[i-1].lng, points[i].lat, points[i].lng, metric='meters')
                tuple.append(speed_gps)
                accel_gps = speed_gps-lastSpeed
                tuple.append(accel_gps)
                tuple.append(abs(accel_gps - lastAccel))
                lastSpeed = speed_gps
                lastAccel = accel_gps
            
            if 'All' in args.suffix or 'Ang' in args.suffix:  # to include angular speed in basic features 
                angularSpeed = returnAngularDisplacement(points[i-1].lat, points[i-1].lng, points[i].lat, points[i].lng)
                tuple.append(angularSpeed)
                if lastAngleSpeed == -1000: tuple.append(0)
                else: tuple.append(abs(angularSpeed-lastAngleSpeed))
                lastAngleSpeed = angularSpeed
            
            if 'All' in args.suffix or 'Rp' in args.suffix:   # to include engine RPM in basic features 
                tuple.append(points[i].rpm)
                tuple.append(abs(points[i].rpm - points[i-1].rpm))
            
            if 'All' in args.suffix or 'Hd' in args.suffix:  # to include heading in basic features 
                tuple.append(points[i].heading)
                tuple.append(abs(points[i].heading - points[i-1].heading))
            
            if 'All' in args.suffix or 'XYZ' in args.suffix: # to include accelerometer sensor data in basic features 
                tuple.append(points[i].accelX)
                tuple.append(abs(points[i].accelX - points[i-1].accelX))
                tuple.append(points[i].accelY)
                tuple.append(abs(points[i].accelY - points[i-1].accelY))
                tuple.append(points[i].accelZ)
                tuple.append(abs(points[i].accelZ - points[i-1].accelZ))
            
            traj.append(tuple)        
        
        basicFeatures[t] = traj
    
    del trajectories
    print('Basic Features are created!')
    
    
    #Generate Statistical Feature Matrix
    bar = progressbar.ProgressBar()
    start = time.time()
    statisticalFeatureMatrix = {}
    for t in bar(basicFeatures):
        #print 'processing', t      
        matricesForTrajectory = []
        traj= basicFeatures[t]
        LEN = len(traj[0])
        ranges = returnSegmentIndexes(L1, len(traj))        
        for p in ranges:
            if p[1] - p[0] < 256:
                continue
            matrixForSegment = np.empty((129, LEN*7))
            matrixForSegment[0, :] = np.zeros((LEN*7,))
            st = p[0]
            for timestep in range(1, 129):
                en = min(st+L2, p[1])
                column = []
                for fIdx in range(0, LEN):
                    arr = []
                    mean = 0.0
                    for i in range(st, en):            
                        mean += traj[i][fIdx]
                        arr.append(traj[i][fIdx])      
                    arr.sort()
                    mean = mean/len(arr)
                    column.append(mean) #mean
                    column.append(arr[0]) #min
                    column.append(arr[len(arr)-1]) #max                    
                    column.append(stats.scoreatpercentile(arr, 25)) #25% percentile
                    if len(arr)%2 == 0:
                        column.append((arr[len(arr)/2] + arr[(len(arr)/2) -1])/2.0) #50% percentile
                    else:
                        column.append(arr[len(arr)/2]) #50% percentile
                    column.append(stats.scoreatpercentile(arr, 75)) #75% percentile
                    std = 0
                    for a in arr:
                        std += (a-mean)**2
                    column.append(math.sqrt(std)) #standard deviation
                matrixForSegment[timestep, :] = list(column)
                st += L2/2            
            matricesForTrajectory.append(matrixForSegment)
              
        # print (len(matricesForTrajectory))
        statisticalFeatureMatrix[t] = normalizeStatFeatureMatrix(np.array(matricesForTrajectory))        
        
      
    del basicFeatures    
    print("statistical features created")
    keys = [k.split("|") for k, v in statisticalFeatureMatrix.items() for i in range(v.shape[0])]
    cPickle.dump(keys, open('data/RandomSample_5_10.pkl', "wb"))
    del keys    
    np.save('data/RandomSample_5_10.npy', np.vstack(statisticalFeatureMatrix.values()), allow_pickle=False)


def returnSegmentIndexes(L1, leng):
    ranges = []
    start = 0
    while True:        
        end = min(start+L1, leng-1)
        ranges.append([start, end])
        start += L1/2
        if end == leng-1:
            break        
    return ranges

# to apply min-max normalization
def normalizeStatFeatureMatrix(statisticalFeatureMatrix, minimum=0, maximum=10):
    r = float(maximum-minimum)
    mins = statisticalFeatureMatrix.min((0, 1))
    maxs = statisticalFeatureMatrix.max((0, 1))    
    statisticalFeatureMatrix = np.nan_to_num(minimum + ((statisticalFeatureMatrix-mins)/(maxs-mins))*r)
    return statisticalFeatureMatrix
    
if __name__ == '__main__':
    generateStatisticalFeatureMatrix()    
