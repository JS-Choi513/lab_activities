from queue import Queue

q = Queue()
q.put(1)
q.put(2)
q.put(3)



while q.empty() is False:
    print(q.get())
    print(q.qsize()) 

print(q.qsize()) 

#for i in q.queue:
#    print(i)
#print(q.qsize())    

    