import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
filePath = os.path.join(THIS_FOLDER, '/home/js/result_trace.txt')
filePath2 = "/home/js/converted_raw.txt"

f = open(filePath,'r')
f2 = open(filePath2,'w')
line = f.readlines()
# input trace: %T %t %d %S %n %d %a
# second, nanosecond, RWBS_1, sector_of_numbers, sector_numbers, RWBS_2, action
'''
count = 0
for i in range(len(line)):
    print(line[i])
    each = line[i].split(' ')
    #each = line[i].split()
    time = each[0].split('.')
    time[0] = count 
    nsec= time[1][0:3]
    sample= line2[i].split(' ')
    #if count < 3:
    fixline = ''.join(str(time[0])+'.'+nsec+' '+each[1]+' '+sample[2]+' '+each[3]+' '+each[4])
    f1.write(fixline)
    count = count+10
    #else : count+1     
    #fixline = ''.join(each[0]+each[1]+each[2]+each[3]+each[4])
    #f1.write(fixline+'\n')
    print(fixline)

'''

for i in range(len(line)):
    print(line[i])
    each = line[i].split(' ')
    #each = line[i].split()
    sec = float(each[0])*float(1000)
    nanosec = float(each[1])/float(1000)/float(1000)
    #nanosec2 = str(nanosec)[0:3]
    #nanosec = float(nanosec2)
    #nanosec = each[1]
    if each[5] == 'W':
        each[5] = '0'
    elif each[5] == 'R':
        each[5] = '1'
    else:
        each[5] = '0'
    #if count < 3:
    #bcount
    if each[4] =='0':
        continue
    #str(sec+nanosec)[0:9]
    fixline = ''.join( each[0]+each[1]+' '+'0'+' '+each[3]+' '+each[4]+' '+each[5]+'\n')
    f2.write(fixline)
    #count = count+5
    #else : count+1     

    
    

    #fixline = ''.join(each[0]+each[1]+each[2]+each[3]+each[4])
    #f1.write(fixline+'\n')
    print(fixline)