'''
	This code is to create feature matrix for non-deep learning baselines such as Gradient Boosting Decision Tree (GBDT) model. 
	Feature vector here is the original vector used by Dong et al. (2016), as their non-deep learning baseline. 
	Vector created by this method will be of size 321
	Developed by: Sreeja R. Thoom 
	Updated by: Sobhan Moosavi	
'''

import numpy as np
import math
from scipy import stats
import time
import cPickle

class point:
    lat = 0
    lng = 0
    time = 0
    def __init__(self, time, lat, lng):
        self.lat = lat
        self.lng = lng
        self.time = time


class basicfeatures:
    speedNorm = 0
    accelNorm = 0
    diffSpeedNorm = 0
    diffAccelNorm = 0
    angularSpeed = 0
    def __inti__(self, speedNorm, diffSpeedNorm, accelNorm, diffAccelNorm, angularSpeed):
        self.speedNorm = speedNorm
        self.diffSpeedNorm= diffSpeedNorm
        self.accelNorm= accelNorm
        self.diffAccelNorm = diffAccelNorm
        self.angularSpeed = angularSpeed

# to calculate angular speed
def returnAngularDisplacement(fLat, fLon, sLat, sLon):
    #Inspired by: https://www.quora.com/How-do-I-convert-radians-per-second-to-meters-per-second
    
    fLat = np.radians(float(fLat))
    fLon = np.radians(float(fLon))
    sLat = np.radians(float(sLat))
    sLon = np.radians(float(sLon))
    
    dis = np.sqrt((fLat-sLat)**2 + (fLon-sLon)**2)
    return dis


# a helper function to calculate several statistics
def helper_loacl_stats(arr, mean):
    column = []
    if len(arr)!=0:
        mean = mean/len(arr)
        column.append(mean) #mean
        column.append(arr[0]) #min
        column.append(arr[len(arr)-1]) #max                    
        column.append(stats.scoreatpercentile(arr, 25)) #25% percentile
        if len(arr)%2 == 0:
            column.append((arr[int(len(arr)/2)] + arr[int(len(arr)/2) -1])/2.0) #50% percentile
        else:
            column.append(arr[int(len(arr)/2)]) #50% percentile
        column.append(stats.scoreatpercentile(arr, 75)) #75% percentile
        std = 0
        for a in arr:
            std += (a-mean)**2
        column.append(math.sqrt(std))
    else:
        column.extend([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    return column


# to normalize feature vector created for a trajectory
def normalizeFeatureVector(statisticalFeatureMatrix, minimum=0, maximum=10):
    r = float(maximum-minimum)
    mins = statisticalFeatureMatrix.min((0))
    maxs = statisticalFeatureMatrix.max((0))    
    statisticalFeatureMatrix = np.nan_to_num(minimum + ((statisticalFeatureMatrix-mins)/(maxs-mins))*r)
    return statisticalFeatureMatrix


# to return bin value for heading to create several buckets
def returnBinValue(fLat, fLng, sLat, sLng, tLat, tLng):
    sf = haversineDistance(sLat, sLng, fLat, fLng, metric='meters')
    st = haversineDistance(sLat, sLng, tLat, tLng, metric='meters')
    ft = haversineDistance(fLat, fLng, tLat, tLng, metric='meters')
    
    if sf==0 or st==0 or ft==0:
        angle = 180
    else:
        angle = math.degrees(np.arccos((sf**2 + st**2 - ft**2)/(2.0*sf*st)))
    
    # [0,10),  [10,20),  [20,  30),  [30,45),  [45,60),  [60,90),[90,120),  and [120,180]
    
    if angle >= 0 and angle < 10:
        return 1
    elif angle >= 10 and angle < 20:
        return 2
    elif angle >= 20 and angle < 30:
        return 3
    elif angle >= 30 and angle < 45:
        return 4
    elif angle >= 45 and angle < 60:
        return 5
    elif angle >= 60 and angle < 90:
        return 6
    elif angle >= 90 and angle < 120:
        return 7
    else:
        return 8    


# to calculate haversine distance between two points 
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


# a helper function to calculate rectangle-based features
def helper_rectangle(points):
    result= []
    min_lat = points[0].lat
    max_lat = points[0].lat
    min_lng = points[0].lng
    max_lng = points[0].lng
    for i in range(len(points)):
        if points[i].lat >= max_lat:
            max_lat = points[i].lat
        if points[i].lat < min_lat:
            min_lat = points[i].lat
        if points[i].lng >= max_lng:
            max_lng = points[i].lng
        if points[i].lng < min_lng:
            min_lng = points[i].lng
        
    #area =  abs(max_lat - min_lat) * abs(max_lng - min_lng)
    length = haversineDistance(min_lat, max_lng, max_lat, max_lng, metric='mi')
    bredth = haversineDistance(min_lat, max_lng, min_lat, min_lng, metric='mi')
    area = length * bredth 
    result.append(area)
    result.append(length)
    result.append(bredth)
    return result


# to generate feature vector for a trajectory
def generateFeatureVector():
    trajectories = {}
    filename = 'RandomSample_5_10.csv'
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        ct = ''
        cd = ''
        tj = []
        for ln in lines:
            pts = ln.replace('\r\n','').split(',')
            if pts[1] != ct:
                if ct == "" and pts[0]=="Driver":
                    continue
                if len(tj) >0:
                    trajectories[cd+"|"+ct] = tj
                tj = []
                p = point(int(pts[2]), float(pts[3]), float(pts[4]))
                tj.append(p)                
                ct = pts[1]
                cd = pts[0]
            else:
                tj.append(point(int(pts[2]), float(pts[3]), float(pts[4]))) 
        trajectories[cd+"|"+ct] = tj
  
    print('Raw Trajectory Data is loaded! |Trajectories|:' + str(len(trajectories)))
   
    #Generate Basic Features for each trajectory
    basicFeatures = {}
    for t in trajectories:
        points = trajectories[t]
        traj = []
        lastSpeedNorm = lastAccelNorm = -1
        lastLatSpeed = lastLngSpeed = 0
        length = len(points)
        
        rectangleFeatures = helper_rectangle(points)
        tripDuration = points[length-1].time - points[0].time
        tripLength = haversineDistance(points[length-1].lat,points[length-1].lng,points[0].lat, points[0].lng)
        if tripDuration!=0: 
            averageSpeed = tripLength/tripDuration
        else:
            averageSpeed = 0
        
        
        for i in range(1, len(points)):
            speedNorm = haversineDistance(points[i].lat,points[i].lng,points[i-1].lat, points[i-1].lng) #Time difference is unit
            diffSpeedNorm = 0
            if lastSpeedNorm > -1:
                diffSpeedNorm = np.abs(speedNorm - lastSpeedNorm)
                
            latSpeed = np.abs(points[i].lat-points[i-1].lat)
            lngSpeed = np.abs(points[i].lng-points[i-1].lng)
            accelNorm = np.sqrt((latSpeed - lastLatSpeed)**2 + (lngSpeed - lastLngSpeed)**2) #Time difference is unit
            
            diffAccelNorm = 0
            if lastAccelNorm > -1:
                diffAccelNorm = np.abs(accelNorm - lastAccelNorm)
            
            angularSpeed = returnAngularDisplacement(points[i-1].lat, points[i-1].lng, points[i].lat, points[i].lng)
            if i < len(points)-1:
                binNumber = returnBinValue(points[i-1].lat, points[i-1].lng, points[i].lat, points[i].lng, points[i+1].lat, points[i+1].lng)
            else: 
                binNumber = 1
            lastSpeedNorm = speedNorm
            lastAccelNorm = accelNorm
            lastLatSpeed = latSpeed
            lastLngSpeed = lngSpeed
            
            traj.append([speedNorm, diffSpeedNorm, accelNorm, diffAccelNorm, angularSpeed, binNumber, tripDuration, tripLength, averageSpeed, rectangleFeatures[0], rectangleFeatures[1], rectangleFeatures[2]])        
        basicFeatures[t] = traj
    del trajectories
    print('Basic Features are created!')
    
    ########################## Generate Statistical Feature Matrix ################################
    statisticalFeatureMatrix = {}
    for t in basicFeatures:
        #print 'processing', t      
        matricesForTrajectory = []
        
        #traj - store points in a trajectory
        traj= basicFeatures[t] 
        
        #features are appended into column list in the sequence of each basic feature's statistical feature and then local statistical features    
        column = []
        globalFeatures = traj[0]
        column.append(globalFeatures[6])
        column.append(globalFeatures[7])
        column.append(globalFeatures[8])
        column.append(globalFeatures[9])
        column.append(globalFeatures[10])
        column.append(globalFeatures[11])
        for fIdx in range(0,5):
            arr = []
            mean = 0.0
            for i in range(len(traj)):            
                mean += traj[i][fIdx]
                arr.append(traj[i][fIdx])

            arr.sort()
            mean = mean/len(arr)
            column.append(mean) #mean
            column.append(arr[0]) #min
            column.append(arr[len(arr)-1]) #max                    
            column.append(stats.scoreatpercentile(arr, 25)) #25% percentile
            if len(arr)%2 == 0:
                column.append((arr[int(len(arr)/2)] + arr[(len(arr)/2) -1])/2.0) #50% percentile
            else:
                column.append(arr[int(len(arr)/2)]) #50% percentile
            column.append(stats.scoreatpercentile(arr, 75)) #75% percentile
            std = 0
            for a in arr:
                std += (a-mean)**2
            column.append(math.sqrt(std)) #standard deviation
        #trajStatistics[i] = column
        for i in range(0,5):
            a1 = []
            a2 = []
            a3 = []
            a4 = []
            a5 = []
            a6 = []
            a7 = []
            a8 = []
            sum_1 = 0.0
            sum_2 = 0.0
            sum_3 = 0.0
            sum_4 = 0.0
            sum_5 = 0.0
            sum_6 = 0.0
            sum_7 = 0.0
            sum_8 = 0.0
            for j in range(len(traj)):
                if traj[j][5] == 1:
                    a1.append(traj[j][i])
                    sum_1 = sum_1 + traj[j][i]
                elif traj[j][5] == 2:
                    a2.append(traj[j][i])
                    sum_2 = sum_2 + traj[j][i]
                elif traj[j][5] == 3:
                    a3.append(traj[j][i])
                    sum_3 = sum_3 + traj[j][i]
                elif traj[j][5] == 4:
                    a4.append(traj[j][i])
                    sum_4 = sum_4 + traj[j][i]
                elif traj[j][5] == 5:
                    a5.append(traj[j][i])
                    sum_5 = sum_5 + traj[j][i]
                elif traj[j][5] == 6:
                    a6.append(traj[j][i])
                    sum_6 = sum_6 + traj[j][i]
                elif traj[j][5] == 7:
                    a7.append(traj[j][i])
                    sum_7 = sum_7 + traj[j][i]
                else:
                    a8.append(traj[j][i])
                    sum_8 = sum_8 + traj[j][i]
            a1.sort()
            a2.sort()
            a3.sort()
            a4.sort()
            a5.sort()
            a6.sort()
            a7.sort()
            a8.sort()  
            
            #appending local features bin wise (8 bins in total)
            column.extend(helper_loacl_stats(a1, sum_1))
            column.extend(helper_loacl_stats(a2, sum_2))
            column.extend(helper_loacl_stats(a3, sum_3))
            column.extend(helper_loacl_stats(a4, sum_4))
            column.extend(helper_loacl_stats(a5, sum_5))
            column.extend(helper_loacl_stats(a6, sum_6))
            column.extend(helper_loacl_stats(a7, sum_7))
            column.extend(helper_loacl_stats(a8, sum_8))
                        
        statisticalFeatureMatrix[t] = normalizeFeatureVector(np.array(column))
        
    del basicFeatures
    print("######## statistical features created ###########")
    keys = [k.split("|") for k, v in statisticalFeatureMatrix.items()]
    list_1 = statisticalFeatureMatrix.values()
    print(len(list_1))
    print(len(list_1[0]))    
    
    file_name = 'data/non_deep_features'
    cPickle.dump(keys, open(file_name + '.pkl', "wb"))
    del keys
    np.save(file_name + '.npy', np.vstack(statisticalFeatureMatrix.values()))

if __name__ == '__main__':
    generateFeatureVector()

