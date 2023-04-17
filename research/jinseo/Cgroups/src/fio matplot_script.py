from matplotlib import pyplot as plt
import numpy as np
topics = ['kyber', 'bfq', 'mq-deadline']
#plt.title('Scheduler: kyber (write, IOPS)')
plt.ylabel('usec')
plt.title('Multi thread latency per scheduler')
#38457.35,85849.46,88907.38, 81012.34
value_a = [972.09,3323.36,3417.24]# 4(iops)
value_b = [2714.07,2979.94,3823.62]# 8(iops)
value_c = [5602.24,5904.19,8401.83]# 16(iops)
value_d = [11321.56,12642,13799.27]# 32(iops) per scheduler


def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(4, 0.8, 1, 3)
value_b_x = create_x(4, 0.8, 2, 3)
value_c_x = create_x(4, 0.8, 3, 3)
value_d_x = create_x(4, 0.8, 4, 3)
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
ax.bar(value_c_x, value_c)
ax.bar(value_d_x, value_d)
#plt.legend()
plt.legend(labels=['4thread', '8thread', '16thread', '32thread'],bbox_to_anchor=(1.05, 1.0), loc='upper left')
middle_x = [(a+b+c+d)/4 for (a,b,c,d) in zip(value_a_x, value_b_x, value_c_x,value_d_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)
plt.ylim(0,35000)
plt.show()


from matplotlib import pyplot as plt
import numpy as np
topics = ['kyber', 'bfq', 'mq-deadline']
#plt.title('Scheduler: kyber (write, IOPS)')
plt.ylabel('usec')
plt.title('Cgroup latency per scheduler')
#38457.35,85849.46,88907.38, 81012.34
value_a = [972.09,4741.20,4424.31]# 4(iops)
value_b = [7586.07,8749.35,9530.20]# 8(iops)
value_c = [22689.35,24994.01,25642.81]# 16(iops)
value_d = [46291.27,46996.26,45451.57]# 32(iops) per scheduler

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(4, 0.8, 1, 3)
value_b_x = create_x(4, 0.8, 2, 3)
value_c_x = create_x(4, 0.8, 3, 3)
value_d_x = create_x(4, 0.8, 4, 3)
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
ax.bar(value_c_x, value_c)
ax.bar(value_d_x, value_d)
#plt.legend()
plt.legend(labels=['4groups', '8groups', '16groups', '32groups'],bbox_to_anchor=(1.05, 1.0), loc='upper left')
middle_x = [(a+b+c+d)/4 for (a,b,c,d) in zip(value_a_x, value_b_x, value_c_x,value_d_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)
plt.ylim(0,50000)
plt.show()


plt.show()

from matplotlib import pyplot as plt
topics = ['4', '8', '16','32']
plt.title('Scheduler: kyber (write, Latency)')
plt.ylabel('usec')
value_a = [3323.36,2979.94,5904.19,12642.77]


value_b = [4741.20,8749.35,24994.01,46996.26]

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(2, 0.8, 1, 4)
value_b_x = create_x(2, 0.8, 2, 4)
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
plt.legend(labels=['Multi thread', 'Cgroup'])
middle_x = [(a+b)/2 for (a,b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)


plt.show()










from matplotlib import pyplot as plt
topics = ['4', '8', '16','32']
plt.title('Scheduler: bfq (Multi thread, IOPS)')
plt.ylabel('IOPS')
value_a = [12787.66,16870.42,11282.04,14812.43]
value_b = [12781.92,16867.18,11282.70,14811.22]
           

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(2, 0.8, 1, 4)
value_b_x = create_x(2, 0.8, 2, 4)
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
plt.legend(labels=['Read', 'Write'])
middle_x = [(a+b)/2 for (a,b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)


plt.show()

from matplotlib import pyplot as plt
topics = ['4', '8', '16','32']
plt.title('Scheduler: bfq (Multi thread, Bandwidth)')
plt.ylabel('MB/s')
value_a = [52.4,69.1,46.2-
value_b = [52.4,69.1,46.2-

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(2, 0.8, 1, 4)
value_b_x = create_x(2, 0.8, 2, 4)
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
plt.legend(labels=['Read', 'Write'])
middle_x = [(a+b)/2 for (a,b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)


plt.show()

from matplotlib import pyplot as plt
topics = ['4', '8', '16','32']
plt.title('Scheduler: bfq (thread, Latency)')
plt.ylabel('usec')
value_a = [7150.57,9166.88,25157.61-
value_b = [2852.71,6003.62,20223.52-

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(2, 0.8, 1, 4)
value_b_x = create_x(2, 0.8, 2, 4)
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
plt.legend(labels=['Read', 'Write'])
middle_x = [(a+b)/2 for (a,b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)


plt.show()


from matplotlib import pyplot as plt
import numpy as np
topics = ['kyber', 'bfq', 'mq-deadline']
#plt.title('Scheduler: kyber (write, IOPS)')
plt.ylabel('usec')
plt.title('Cgroup Latency per scheduler(write)')
#38457.35,85849.46,88907.38, 81012.34


value_a = [4582,2968,4113]# 4(iops)
value_b = [11550,9352,10565]# 8(iops)
value_c = [40427,25647,28206]# 16(iops)
value_d = [82000,52000,53000]# 32(iops) per scheduler

# value_b = [26936.30,29241.93,20589.69, 23369.03]


def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(4, 0.8, 1, 3)
value_b_x = create_x(4, 0.8, 2, 3)
value_c_x = create_x(4, 0.8, 3, 3)
value_d_x = create_x(4, 0.8, 4, 3)
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
ax.bar(value_c_x, value_c)
ax.bar(value_d_x, value_d)
#plt.legend()
plt.legend(labels=['4groups', '8groups', '16groups', '32groups'],bbox_to_anchor=(1.05, 1.0), loc='upper left')
middle_x = [(a+b+c+d)/4 for (a,b,c,d) in zip(value_a_x, value_b_x, value_c_x,value_d_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)
plt.ylim(0,60000)
plt.show()
