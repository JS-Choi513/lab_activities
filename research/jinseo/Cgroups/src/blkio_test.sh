# Test script for Block IO Controller

function print_usage(){
    echo "usage: 0$ <GroupB> <GroupA>"
    exit 1
}

# test file validation check 
flow=/mnt/GroupB.dat
fhigh=/mnt/GroupA.dat
if [ $# != 2 ]; then
    print_usage
fi 

# variable assignment for each cgroup path 
cg_B=$1
cg_A=$2

# cgroup directory validation check
for cg in $cg_B $cg_A; do
    if [ ! -d $cg ]; then   
        echo "$cg does not exist"
        print_usage
    fi
done 

# temp directory  
out_cgB=$(mktemp) 
out_cgA=$(mktemp)

echo -n "sync and drop all caches..."
# save unsaved data to disk form memeory(I/O syncronization)
sync

# clear cache memory: page cache, dentry, inode
echo 3 > /proc/sys/vm/drop_caches
echo "done"

echo -n "reading file..."

# register current process(shell) at each cgroup tasks file 
# measuring reading time and write temp file, if standard error occur, write error message to file though
echo $$ > $cg_B/tasks 
(time dd if=$flow of=/dev/null) > $out_cgB 2>&1 & # '&' means parallel execution 

echo $$ > $cg_A/tasks
(time dd if=$fhigh of=/dev/null) > $out_cgA  2>&1 & 

# waiting for child process done
wait
echo "done"

# print result 
echo "-----------------------------------------------"
echo "dd in $cg_B:"
cat $out_cgB
# echo $out_cgB 2>&1 /home/js/log
echo "-----------------------------------------------"
echo "dd in $cg_A:"
cat $out_cgA
# echo $out_cgA GroupA\n > /home/js/log.txt
# remove temp file 
rm -f $out_cgB $out_cgA

