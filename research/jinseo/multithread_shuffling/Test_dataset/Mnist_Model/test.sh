#!bin/bash

python3.8 Mnist_model_tfdata_original_300.py > Cifar_original_log.txt &


PROCESS2=`ps -ef |grep 'mpstat' | grep -v "grep" | awk '{print$2}'`

#echo ${PROCESS2}

while :
do
    RESULT1=`pgrep 'python3.8'`  #`ps -ef |grep 'python3.8' | grep -v "grep" | awk '{print$2}'`

    if [ -z ${RESULT1} ] ; then
            echo "Execution complete"&
	    kill -9 ${PROCESS2}
	    break
    #else
            #echo "running"
    fi
    sleep 3
done    

echo 3 > /proc/sys/vm/drop_caches
sleep 5

python3.8 Mnist_model_tfdata_original_300.py > Cifar_original_log2.txt &


PROCESS2=`ps -ef |grep 'mpstat' | grep -v "grep" | awk '{print$2}'`

#echo ${PROCESS2}

while :
do
    RESULT1=`pgrep 'python3.8'`  #`ps -ef |grep 'python3.8' | grep -v "grep" | awk '{print$2}'`

    if [ -z ${RESULT1} ] ; then
            echo "Execution complete"&
	    kill -9 ${PROCESS2}
	    break
    #else
            #echo "running"
    fi
    sleep 3
done    

echo 3 > /proc/sys/vm/drop_caches
sleep 5

