import numpy as np 
from matplotlib import pyplot as plt

volume = ["Fio(Randread)","Fio(Randwrite)"]
GroupBthroughput = [531.75,275.35]#IOPS normal FioIOPS GPU+Fio
GroupAthroughput = [519.14,519.37]#
#GroupCthroughput = [531.75,275.35]# latency GPU+Fio ??????
#GroupDthroughput = [519.14,519.37]# latency normal Fio

#plt.plot(volume,GroupAthroughput,marker="*")
#plt.plot(volume,GroupBthroughput,marker="^")
def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(4, 0.8, 1, 2)
value_b_x = create_x(4, 0.8, 2, 2)
#value_c_x = create_x(4, 0.8, 3, 2)
#value_d_x = create_x(4, 0.8, 4, 3)

ax = plt.subplot()
ax.bar(value_a_x, GroupAthroughput,width=0.8)
ax.bar(value_b_x, GroupBthroughput,width=0.8)
#ax.bar(value_c_x, GroupCthroughput,width=0.8)
#ax.bar(value_d_x, GroupDthroughput,width=0.8)

middle_x = [(a+b)/2 for (a,b) in zip(value_a_x,value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(volume)
#plt.bar(volume,GroupAthroughput,width=0.3)
#plt.bar(volume,GroupBthroughput,width=0.3)

#plt.plot(volume,GroupAthroughput,marker='o',color='red')
#plt.xticks(rotation=0)
#plt.grid(True)
#plt.xlabel('')
plt.ylabel('usec')
plt.legend(['Fio','Fio+GPU'],bbox_to_anchor=(1.35, 1.0),loc='upper right',)
plt.title('Fio Latency')
plt.ylim([0,600])
for i, v in enumerate(value_a_x):
    plt.text(v, GroupAthroughput[i], GroupAthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 10, 
             color='black',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)

for i, v in enumerate(value_b_x):
    plt.text(v, GroupBthroughput[i], GroupBthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 10, 
             color='black',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)
             
#for i, v in enumerate(value_c_x):
#    plt.text(v, GroupCthroughput[i], GroupCthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
#             fontsize = 10, 
#             color='black',
#             horizontalalignment='center',  # horizontalalignment (left, center, right)
#             verticalalignment='bottom')    # verticalalignment (top, center, bottom)	

#for i, v in enumerate(value_d_x):
#    plt.text(v, GroupDthroughput[i], GroupDthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
#             fontsize = 10, 
#             color='black',
#             horizontalalignment='center',  # horizontalalignment (left, center, right)
#             verticalalignment='bottom')    # verticalalignment (top, center, bottom)	             
'''
for i, v in enumerate(value_d_x):
    if i==0:
        continue
    else:
        plt.text(v, GroupDthroughput[i], GroupDthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
        fontsize = 10, 
        color='black',
        horizontalalignment='center',  # horizontalalignment (left, center, right)
        verticalalignment='bottom')    # verticalalignment (top, center, bottom)	             	 
'''
plt.show()
'''
volume = ["tf only","Fio(read)+tf","Fio(write)+tf"]
GroupAthroughput = [1301,1373,1348]
GroupBthroughput = [1380,1336,1351]
GroupCthroughput = [1144,1153,1190]
GroupDthroughput = [0,1938,2077]
#plt.plot(volume,GroupAthroughput,marker="*")
#plt.plot(volume,GroupBthroughput,marker="^")
def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(4, 0.8, 1, 3)
value_b_x = create_x(4, 0.8, 2, 3)
value_c_x = create_x(4, 0.8, 3, 3)
value_d_x = create_x(4, 0.8, 4, 3)

ax = plt.subplot()
ax.bar(value_a_x, GroupAthroughput)
ax.bar(value_b_x, GroupBthroughput)
ax.bar(value_c_x, GroupCthroughput)
ax.bar(value_d_x, GroupDthroughput)

middle_x = [(a+b+c+d)/4 for (a,b,c,d) in zip(value_a_x,value_b_x, value_c_x, value_d_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(volume)
#plt.bar(volume,GroupAthroughput,width=0.3)
#plt.bar(volume,GroupBthroughput,width=0.3)

#plt.plot(volume,GroupAthroughput,marker='o',color='red')
#plt.xticks(rotation=0)
#plt.grid(True)
#plt.xlabel('')
plt.ylabel('sec')
plt.legend(['cpuset','cpuset+exclusive_cpu','GPU'],bbox_to_anchor=(1.45, 1.0),loc='upper right',)
plt.title('NVMe SSD multi-queue system Execution Time')
plt.ylim([0,2200])
for i, v in enumerate(value_a_x):
    plt.text(v, GroupAthroughput[i], GroupAthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 10, 
             color='black',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)

for i, v in enumerate(value_b_x):
    plt.text(v, GroupBthroughput[i], GroupBthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 10, 
             color='black',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)
for i, v in enumerate(value_c_x):
    plt.text(v, GroupCthroughput[i], GroupCthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 10, 
             color='black',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)	

for i, v in enumerate(value_d_x):
    plt.text(v, GroupDthroughput[i], GroupDthroughput[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 10, 
             color='black',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)	             	 
plt.show()
'''
"""
####################################
Noop
GroupAthroughput = [87.4, 146, 128, 189, 182, 
			  139.4, 97.38, 113, 130, 135, 
			  148, 162, 176, 117, 176, 
			  185, 189, 189, 191, 186, 
			  185, 191, 187]
GroupBthroughput = [72.7, 113, 63.9, 50, 53, 
			  88.36, 97.14, 90.4, 93, 118, 
			  101, 110, 116, 118, 116,
			  118, 119, 120, 120, 119, 
			  109, 119, 118]

####################################
N=23
GroupA =[87.4,146,128,189,182,139.4,97.38,113,130,135,148,162,176,185,189,189,191,186,185,191,187]
GroupB =[72.7,113,63.9,50,53,88.36,97.14,90.4,93,118,101,110,116,118,119,120,120,119,109,119,118]

//ax.plot([87.4,146,128,189,182,139.4,97.38,113,130,135,148,162,176,185,189,189,191,186,185,191,187],
		'k',label = 'GroupA')
plt.plot(GroupA)
plt.plot(GroupB)
ind = np.arange(N) 
plt.xticks(ind,('100ki', '300ki', '500ki', '700ki', '900ki',
				'1mi','3mi','5mi','7mi','9mi'
				'10mi','30mi','50mi','70mi','90mi'
				'100mi','300mi','500mi','700mi','900mi'
				'1Gi','3Gi','5Gi'))
							

plt.xlabel('Volume(byte)')
plt.ylabel('Read(mb/sec)')
plt.title('Direct I/O ')
plt.legend(['GroupA(cpu weight: 1024)','GroupB(cpu weight: 256)'])
plt.show()
"""


