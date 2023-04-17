#!/bin/sh
cd /home/js; blktrace /dev/nvme0n1p2 -a issue &
cd /home/js; fio test.fio 
#wait until fio exit
sleep 5
#kill blktrace
kill -9 `ps -ef | grep blktrace | awk '{print $2}'`
cd /home/js; blkparse -f "%T %t %d %S %n %d %a\n" -i nvme0n1p2 > result_trace.txt
chmod 777 raw_format_converter.py 
python3 /home/js/Desktop/noslab.storage/research/jinseo/Tracefile/raw_format_converter.py
