#!bin/bash




python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_original_10.py > mnist_tfdata_original_10.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_original_50.py > mnist_tfdata_original_50.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_original_90.py > mnist_tfdata_original_90.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_original_300.py > mnist_tfdata_original_300.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_original_600.py > mnist_tfdata_original_600.txt &

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





: << "END"

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_numpy_10.py > mnist_numpy_10.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_numpy_50.py > mnist_numpy_50.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_numpy_90.py > mnist_numpy_90.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_numpy_300.py > mnist_numpy_300.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_numpy_300.py > mnist_numpy_300.txt &

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











python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_10_6.py > mnist_mt_6_seed10_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_50_6.py > mnist_mt_6_seed50_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_90_6.py > mnist_mt_6_seed90_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_300_6.py > mnist_mt_6_seed300_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_600_6.py > mnist_mt_6_seed600_original.txt &

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














python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_10_12.py > mnist_mt_12_seed10_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_50_12.py > mnist_mt_12_seed50_original.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_90_12.py > mnist_mt_12_seed90_original.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_300_12.py > mnist_mt_12_seed300_original.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_600_12.py > mnist_mt_12_seed600_original.txt &


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













python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_10_24.py > mnist_mt_24_seed10_original.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_50_24.py > mnist_mt_24_seed50_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_90_24.py > mnist_mt_24_seed90_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_300_24.py > mnist_mt_24_seed300_original.txt &
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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_600_24.py > mnist_mt_24_seed600_original.txt &

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



python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_10_36.py > mnist_mt_36_seed10_original.txt &


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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_50_36.py > mnist_mt_36_seed50_original.txt &

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
python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_90_36.py > mnist_mt_36_seed90_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_300_36.py > mnist_mt_36_seed300_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_600_36.py > mnist_mt_36_seed600_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_10_48.py > mnist_mt_48_seed10_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_50_48.py > mnist_mt_48_seed50_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_90_48.py > mnist_mt_48_seed90_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_300_48.py > mnist_mt_48_seed300_original.txt &

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

python3.8 /home/js/noslab.storage/research/jinseo/tfdata/Mnist_Model/FMnist_model_tfdata_mtshuffle_per_epoch_600_48.py > mnist_mt_48_seed600_original.txt &

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
END
