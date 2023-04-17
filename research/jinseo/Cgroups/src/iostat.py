#!/usr/bin/python

import subprocess
from datetime import datetime
import time

while True:
        ps = subprocess.Popen(('iostat', '-m'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('grep', 'sdb'), stdin=ps.stdout)
        ps.wait()
        res = output.split(" ")
        res = [ x for x in res if x != '' ]
        value = str(datetime.today())[:19]+','+','.join(res)
        with open('iostat_output.csv','a') as f:
           f.write(value)
        time.sleep(1)