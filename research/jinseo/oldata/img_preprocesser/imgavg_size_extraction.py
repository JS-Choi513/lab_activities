import os 
import pathlib
import numpy as np

### Caltehch UCSD Bird-2011
root= '/home/js/Downloads/CUB_200_2011/images'

def imgSize_abstraction(filename):
    width = []
    height = []
    file = open(filename,'r')
    for line in file:
        print(line)
        token = line.split(" ")
        width.append(float(token[3]))
        height.append(float(token[4]))
        
        
    width = np.array(width)
    width_avg = np.average(width)
    height = np.array(height)
    height_avg = np.average(height)
    avg_size = str(width_avg)+ " " + str(height_avg)
    print("average size is: ", avg_size)   
    return avg_size
        

if __name__ == "__main__":
    root= '/home/js/Downloads/CUB_200_2011/bounding_boxes.txt'
    val = imgSize_abstraction(root)
    