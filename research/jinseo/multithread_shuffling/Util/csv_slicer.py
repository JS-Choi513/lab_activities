# this loads the first file fully into memory
with open('#', 'r') as f:
    csvfile = f.readlines()
linesPerFile = 1000000
filename = 1
# this is better then your former loop, it loops in 1000000 lines a peice,
# instead of incrementing 1000000 times and only write on the millionth one
for i in range(0,len(csvfile),linesPerFile):
    with open(str(filename) + '.csv', 'w+') as f:
        if filename > 1: # this is the second or later file, we need to write the
            f.write(csvfile[0]) # header again if 2nd.... file
        f.writelines(csvfile[i:i+linesPerFile])
    filename += 1