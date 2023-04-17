#!/bin/sh
#########################################################
# 스케줄러별 Fio Random Read & Write test 
# Blk-mq layer에서 사용가능한 스케줄러
# kyber->bfq->mq-deadline 순으로 테스트
# 1.스케줄러 변경
# 2.멀티쓰레드->Cgroup 번갈아 가면서 실행 
# 3.4->8->16->32그룹(쓰레드) 순서로 실행
# 4.각 케이스마다 스토리지 포맷 실행

# 실험환경
# cpu		Intel® Core i5-9500 @ 3.0GHz, 6core	    
# NVMe SSD​	Intel® SSDPEKNW512G8 512GB
# OS		​​Ubunbtu 16.04, Linux Kernel 4.15.0​​	
# 테스트 장치명: nvme0n1

# fio Job file 다운로드 
# dependency
# fio
# mkfs

# 세팅값
# fio job파일 file_directory = /media/[사용자명]
# lsblk: 현재장치명 확인 및 테스트 장치 지정 
# fio job파일과 shell스크립트는 같은 디렉터리 내 존재 
#########################################################



echo kyber > /sys/block/nvme0n1/queue/scheduler
cat /sys/block/nvme0n1/queue/scheduler | tee -a kyber.txt
wait
echo -------------kyber randomrw 4thread fio------------------ | tee -a kyber.txt
fio 4thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a kyberrw.txt
echo -------------test storage Formatting...------------------ | tee -a kyber.txt
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------kyber randomrw 4cgroup fio------------------ | tee -a kyber.txt
fio 4cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a kyberrw.txt
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------kyber randomrw 8thread fio------------------ | tee -a kyber.txt
fio 8thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a kyberrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
yes | mkfs -T ext4 /dev/nvme0n1
wait
echo -------------kyber randomrw 8cgroup fio------------------ | tee -a kyber.txt
fio 8cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a kyberrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------kyber randomrw 16thread fio------------------ | tee -a kyber.txt
fio 16thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a kyberrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------kyber randomrw 16cgroup fio------------------ | tee -a kyber.txt
fio 16cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a kyberrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------kyber randomrw 32thread fio------------------ | tee -a kyber.txt
fio 32thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a kyberrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------kyber randomrw 32cgroup fio------------------ | tee -a kyber.txt
fio 32cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a kyberrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait

echo bfq > /sys/block/nvme0n1/queue/scheduler
cat /sys/block/nvme0n1/queue/scheduler | tee -a bfqrw.txt
wait


echo -------------bfq randomrw 4thread fio------------------ | tee -a bfqrw.txt
fio 4thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a bfqrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------bfq randomrw 4cgroup fio------------------ | tee -a bfqrw.txt
fio 4cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a bfqrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------bfq randomrw 8thread fio------------------ | tee -a bfqrw.txt
fio 8thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a bfqrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------bfq randomrw 8cgroup fio------------------ | tee -a bfqrw.txt
fio 8cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a bfqrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------bfq randomrw 16thread fio------------------ | tee -a bfqrw.txt
fio 16thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a bfqrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------bfq randomrw 16cgroup fio------------------ | tee -a bfqrw.txt
fio 16cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a bfqrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------bfq randomrw 32thread fio------------------ | tee -a bfqrw.txt
fio 32thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a bfqrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------bfq randomrw 32cgroup fio------------------ | tee -a bfqrw.txt
fio 32cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a bfqrw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo mq-deadline > /sys/block/nvme0n1/queue/scheduler
cat /sys/block/nvme0n1/queue/scheduler | tee -a mq-deadline.txt
wait

echo -------------mq-deadline randomrw 4thread fio------------------ | tee -a mq-deadlinerw.txt
fio 4thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a mq-deadlinerw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------mq-deadline randomrw 4cgroup fio------------------ | tee -a mq-deadlinerw.txt
fio 4cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a mq-deadlinerw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------mq-deadline randomrw 8thread fio------------------ | tee -a mq-deadlinerw.txt
fio 8thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a mq-deadlinerw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------mq-deadline randomrw 8cgroup fio------------------ | tee -a mq-deadlinerw.txt
fio 8cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a mq-deadlinerw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------mq-deadline randomrw 16thread fio------------------ | tee -a mq-deadlinerw.txt
fio 16thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write" | tee -a mq-deadlinerw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------mq-deadline randomrw 16cgroup fio------------------ | tee -a mq-deadlinrw.txt
fio 16cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write"| tee -a mq-deadlinerw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------mq-deadline randomrw 32thread fio------------------ | tee -a mq-deadlinerw.txt
fio 32thread.fio | grep "iops\|lat (usec)\|BW=\|read\|write"| tee -a mq-deadlinerw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait
echo -------------mq-deadline randomrw 32cgroup fio------------------ | tee -a mq-deadlinerw.txt
fio 32cgroup.fio | grep "iops\|lat (usec)\|BW=\|read\|write"| tee -a mq-deadlinerw.txt
wait
umount /dev/nvme0n1
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1/ /media/js
wait

