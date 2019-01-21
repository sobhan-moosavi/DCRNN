# Driving Style Representation in Convolutional Recurrent Neural Network Models of Driver Identification

In this repository we provide all the implementations for our models and baselines, along with several sample files to reproduce our results.

## Feature Engineering 
* Statistical Feature Map
* Feature Vector V1
* Feature Vector V2

## Models
* D-CRNN
* VRAE
* GBDT
* CNN-model
* RNN-model

## Requirements


## How to Run


## Sample Data
You may find a raw sample file in [data](https://github.com/sobhan-moosavi/DCRNN/tree/master/data) directory. The format of this file is described as follows: 

| Attribute | Description |
|:---------:|-------------|
|Driver| Indicates driver id, which is a string. |
|ID| Indicates trajectory id, which is a string. |
|Time| An integer which indicates the timestep for a datapoint of a trajectory. |
|Lat| Shows the latitude value of GPS coordinate. |
|Lon| Shows the longitude value of GPS coordinate. |
|Speed| Shows the ground velocity of the vehicle as reported by OBD-II port. |
|Acceleration| Shows the rate of change of ground velocity (speed). |
|RPM| Shows the round per minute, as reported by OBD-II port. |
|Heading| Shows the bearing of the vehicle, which is a value between 0 and 359. |
|AccelX| Shows the acceleration sensor reading along with X-axis. |
|AccelY| Shows the acceleration sensor reading along with Y-axis. |
|AccelZ| Shows the acceleration sensor reading along with Z-axis. |

## Acknowledgments 
