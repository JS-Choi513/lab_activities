import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
filePath = os.path.join(THIS_FOLDER, '/home/js/result_trace.txt')
filePath2 = "/home/js/Tracefile/single_groups_io_converted.trace"
f = open(filePath,'r')
f1 = open(filePath2,'w')
line = f.readlines()
# input trace: %T %t %d %S %n %d %a
# second, nanosecond, RWBS_1, sector_of_numbers, sector_numbers, RWBS_2, action
for i in range(len(line)):
    print(line[i])
    each = line[i].split()
    sec = each[0]
    nano = each[1]
    rw = each[2]
    sector = each[3]
    num = each[4]
    rw2 = each[5]
    action = each[6]

    sector = int(each[3])-7 # block count 
    page = int(sector) / 8  # block number 
    remain = int(sector) % 8
    num = int(num) + remain
    page_num = int(num) / 8
    if int(num) % 8:
        int(page_num)+1
    term = sector * 512
    if sector == 0:
        continue
    if page_num == 0:
        continue
    if page_num > 4096:
        continue
    each[0] = str(int(sec)*1000+(int(nano)/1000)/int(1000))# timestamp
    each[0] = each[0][0:10]+' '
    each[1] = '0 ' # devno
    each[2] = str(int(page) * 8)+' ' # blkno
    each[3] = str(int(page_num) * 8)+' ' # bcount 
    if each[5] == 'W' or each[5] == 'WM' or each[5] == 'WS': # rw
        each[4] = '0 ' 
    else:
         each[4] = '1 '
    #each[3] = (rw[0] == 'W') ? 0 : 1
    fixline = ''.join(each[0]+each[1]+each[2]+each[3]+each[4])
    f1.write(fixline+'\n')
    print(fixline)


