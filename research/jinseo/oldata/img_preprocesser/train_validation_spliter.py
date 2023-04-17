import os 
import pathlib
from pickletools import uint8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
### Caltehch UCSD Bird-2011
root= '/home/js/Downloads/CUB_200_2011/images'
### input train_test_split.txt, Images.txt and user path to save train data
data_dir = pathlib.Path(root)
image_count = data_dir.glob('*/*.jpg')
np.set_printoptions(threshold=np.inf, linewidth=np.inf) #release print limit


def file_shuffle(path):
    open()

# return img data to 1row rgb array 
def img_preprocessor(path, width_px, height_px):
    img = Image.open(path)
    img = img.resize((width_px,height_px))
    #img = img.convert('1')
    #plt.imshow(img)
    #plt.show()
    #exit()
    img_val = np.array(img)#  0 to 1 normalization needed?
    print(img_val.dtype)
    #print(img_val.shape)
    single_row_data = img_val.flatten()
    #print(single_row_data)
    #new = single_row_data.reshape(64,64,3)
    #print(new.shape)
    #print(type(new))
    #print(new)
    #img = Image.fromarray(new,'RGB')
    #plt.imshow(img)
    #plt.show()
    #exit()  
    #print(img_val.shape)
    #single_row_data = img_val.flatten()
    #print(single_row_data.shape)
    #print(type(single_row_data))
    #exit()

    #print(img_val)
    return single_row_data
    
    
    
def train_test_spliter(img_data_path, split_id_file, class_label_file, train_test_path):
    flush_count = 0
    test_count = 0
    data_dir = pathlib.Path(img_data_path)
    image_path = data_dir.glob('*/*.jpg')
    os.makedirs(train_test_path+'/train')
    os.makedirs(train_test_path+'/test')
    split_def = open(split_id_file,'r')
    class_label = open(class_label_file,'r')
  
    train_data = open(train_test_path+'/train/bird_train.csv','w') 
    test_data = open(train_test_path+'/test/bird_test.csv','w')

    for img_metadata in zip(image_path, split_def, class_label):
        print(img_metadata[1])
        is_training = str(img_metadata[1].split()[1])
        img_data= img_preprocessor(img_metadata[0], 32, 32)
        if is_training is "1" or test_count<2:
            if is_training is "0": test_count= test_count+1
            print("training")
            single_img_data = str(img_metadata[2].split()[1]) + " " + ' '.join([str(i) for i in img_data.tolist()])+'\n'
            train_data.write(single_img_data)
        elif is_training is "0" and test_count == 2:
            test_count =0
            single_img_data = str(img_metadata[2].split()[1])+ " " + ' '.join([str(i) for i in img_data.tolist()])+'\n'
            test_data.write(single_img_data)
            #np.savetxt(test_data, img_data, delimiter=' ', header=img_metadata[2].split()[1],fmt="%s")
            
        if flush_count == 100:
            train_data.flush()
            test_data.flush()
            flush_count = 0
        flush_count = flush_count+1             
    
    train_data.flush()
    train_data.close()
    test_data.flush()
    test_data.close()            
        

        

if __name__ == "__main__":
    root= '/home/js/Downloads/CUB_200_2011/'
    train_test_spliter(root+'images', 
                       root+'train_test_split.txt', 
                       root+'image_class_labels.txt',
                       '/home/js/Bird')
    #img_preprocessor('/home/js/Downloads/CUB_200_2011/images/084.Red_legged_Kittiwake/Red_Legged_Kittiwake_0068_795430.jpg',255,255)
    