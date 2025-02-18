
import os

def myAdditions(file):    

    pa = "HARE".encode('utf-8')
    pad = bytearray(pa)
    myBuff = bytearray()

    #Convert size to KB
    fileS = str(os.stat(file).st_size)
    encoded = fileS.encode('utf-8')
    filesize = bytearray(encoded)

    split_tup = os.path.splitext(file)
    file_extension = split_tup[-1]

    encoded=file_extension.encode('utf-8')
    file_ex=bytearray(encoded)
    
    myBuff.extend(filesize)
    myBuff.extend(pad)
    myBuff.extend(file_ex)
    myBuff.extend(pad)

    print(pad)
    for x in pad:
        print(x)

file = input("Enter File:")

myAdditions(file)

