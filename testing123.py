
import os

def myAdditions(file):    

    pa = "HARE".encode('utf-8')
    pad = bytearray("HARE".encode('utf-8'))
    myBuff = bytearray()

    #Convert size to KB
    fileS = str(os.stat(file).st_size)
    encoded = fileS.encode('utf-8')
    filesize = bytearray(encoded)

    split_tup = os.path.splitext(file)
    file_extension = split_tup[-1]

    encoded=file_extension.encode('utf-8')
    file_ex=bytearray(file_extension.encode('utf-8'))
    
    myBuff.extend(filesize)
    myBuff.extend(pad)
    myBuff.extend(file_ex)
    myBuff.extend(pad)

    return myBuff

def unCover(myArray):
    pa = "HARE".encode('utf-8')
    pad = bytearray(pa)

    mySize = bytearray()
    count = 0

    for x in range(len(myArray)):
        if myArray[x] == pad[0]:
            for y in range(x,x+5):
                if count == len(pad):
                    return x
                if myArray[y] != pad[count]:
                    break
                count += 1
    
    return -1



file = input("Enter File:")

mine = myAdditions(file)

myfile = 0

f = open(file,'rb')
try:
    myfile = bytearray(f.read())
except:
    f.close()
    print("File Not Available")
else:
    f.close()

mine.extend(myfile)

#First run grabs file_size
print(mine)
pos = unCover(mine)
str = mine[:pos].decode('utf-8',errors="ignore")
print(str)

#Second run grabs file_type
next = mine[5+4:]
pos = unCover(next)
str = next[:pos].decode('utf-8',errors="ignore")
print(str)
