import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
filePath = os.path.join(THIS_FOLDER, '/home/js/converted_raw.txt')
filePath2 = "/home/js/interval_revised.txt"
f = open(filePath,'r')
f1 = open(filePath2,'w')
line = f.readlines()
# input trace: %T %t %d %S %n %d %a
# second, nanosecond, RWBS_1, sector_of_numbers, sector_numbers, RWBS_2, action
count = 0
for i in range(len(line)):
    print(line[i])
    each = line[i].split(' ')
    #each = line[i].split()
    time = each[0].split('.')
    time[0] = count 
    #if count < 3:
    fixline = ''.join(str(time[0])+'.'+str(time[1])+' '+each[1]+' '+each[2]+' '+each[3]+' '+each[4])
    f1.write(fixline)
    count = count+1
    #else : count+1     

    
    

    #fixline = ''.join(each[0]+each[1]+each[2]+each[3]+each[4])
    #f1.write(fixline+'\n')
    print(fixline)


