#!/bin/bash
free | grep Mem > memlog.txt
while : 
do 
	free | grep Mem >> memlog.txt
	sleep 1
done
