#!bin/bash


python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_100.py &

mpstat 1 > mnist_100_1.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_100.py &

mpstat 1 > mnist_100_2.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_100.py &

mpstat 1 > mnist_100_3.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_200.py &

mpstat 1 > mnist_200_1.txt&


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
sleep 20


python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_200.py &

mpstat 1 > mnist_200_2.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_200.py &

mpstat 1 > mnist_200_3.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_400.py &

mpstat 1 > mnist_400_1.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_400.py &

mpstat 1 > mnist_400_2.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_400.py &

mpstat 1 > mnist_400_3.txt&


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
sleep 20


python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_800.py &

mpstat 1 > mnist_800_1.txt&


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
sleep 20


python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_800.py &

mpstat 1 > mnist_800_2.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_800.py &

mpstat 1 > mnist_800_3.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_1000.py &

mpstat 1 > mnist_1000_1.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_1000.py &

mpstat 1 > mnist_1000_2.txt&


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
sleep 20

python3.8 Mnist_model_tfdata_mtshuffle_per_epoch_1000.py &

mpstat 1 > mnist_1000_3.txt&


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
sleep 20