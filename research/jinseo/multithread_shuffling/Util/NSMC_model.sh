#!bin/bash




python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_original_10.py > NSMC_original_10.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_original_50.py > NSMC_original_50.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_original_90.py > NSMC_original_90.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_original_300.py > NSMC_original_300.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_original_600.py > NSMC_original_600.txt &

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






python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_numpy_10.py > NSMC_numpy_10.txt &

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
python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_numpy_50.py > NSMC_numpy_50.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_numpy_90.py > NSMC_numpy_90.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_numpy_300.py > NSMC_numpy_300.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_numpy_600.py > NSMC_numpy_600.txt &

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











python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_10_6.py > NSMC_mtshuffle_10_6.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_50_6.py > NSMC_mtshuffle_50_6.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_90_6.py > NSMC_mtshuffle_90_6.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_300_6.py > NSMC_mtshuffle_300_6.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_600_6.py > NSMC_mtshuffle_600_6.txt &

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














python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_10_12.py > NSMC_mtshuffle_10_12.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_50_12.py > NSMC_mtshuffle_50_12.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_90_12.py > NSMC_mtshuffle_90_12.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_300_12.py > NSMC_mtshuffle_300_12.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_600_12.py > NSMC_mtshuffle_600_12.txt &


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













python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_10_24.py > NSMC_mtshuffle_10_24.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_50_24.py > NSMC_mtshuffle_50_24.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_90_24.py > NSMC_mtshuffle_90_24.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_300_24.py > NSMC_mtshuffle_300_24.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_600_24.py > NSMC_mtshuffle_600_24.txt &


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



python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_10_36.py > NSMC_mtshuffle_10_36.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_50_36.py > NSMC_mtshuffle_50_36.txt &

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
python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_90_36.py > NSMC_mtshuffle_90_36.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_300_36.py > NSMC_mtshuffle_300_36.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_600_36.py > NSMC_mtshuffle_600_36.txt &

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





python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_10_48.py > NSMC_mtshuffle_10_48.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_50_48.py > NSMC_mtshuffle_50_48.txt &
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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_90_48.py > NSMC_mtshuffle_90_48.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_300_48.py > NSMC_mtshuffle_300_48.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/NSMC_Model/NSMC_mtshuffle_600_48.py > NSMC_mtshuffle_600_48.txt &


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

