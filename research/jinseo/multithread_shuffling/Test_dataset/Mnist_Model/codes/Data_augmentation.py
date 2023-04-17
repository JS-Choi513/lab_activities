#src_file = open('/media/js/test/fashion_mnist/fashion-mnist_train.csv','r')
#dest_file = open('/media/js/test/fashion_mnist/fashion-mnist_train_1GB.csv','a')
count=1
for i in range(3):
    src_file = open('/home/js/train.csv','r')
    dest_file = open('/home/js/train_aug.csv','a')
    line = src_file.readline() 
    print(line)       
    if count == 1:
        dest_file.write(line)
        count = 0
    while True:
        line = src_file.readline()        
        if not line: break
        dest_file.writelines(line)
    
    src_file.close()
    dest_file.close()