

"""
data = bytearray(b'\x0f\xf0\xff\x00\xff')
key = bytearray(b'\x0f\xf0')

result = bytearray()

ending = len(data)%len(key)
inside = len(data)//len(key)
low = 0
z = 0
for x in range(inside):
    for y in range(low,low+len(key)):
        #print(f"y is {y}")
        result.append(data[y]^key[z])
        z = z + 1
    low = low + inside
    z = 0

z = 0
for w in range(len(data)-ending,len(data)):
    result.append(data[w]^key[z])
    z = z + 1



for x in range(len(data)) :
    result.append(data[x] ^ key[x])
    print(result[x])

"""


#Start of project here

#Function to handle Two Option Prompt
#Input: A Prompt String
#Output: A Boolean indicating which choice was picked
def confirm_loop(prompt_str):
    intry = False
    while not intry:
        print(prompt_str)
        ans = input("Enter: ")
        match ans:
            case "0":
                return False
            case "1":
                return True
            case _:
                print("Invalid Input\n")


#So weird thing is if you take out an element of bytearray
#it's considered an integer. 
#Input: a "byte" (an int)
#Output: an int list of 0s and 1s representing a binary number
def convert_byte_to_bits(mybyte):
    myint = []
    while mybyte > 0:
        bit_ = mybyte % 2
        myint.insert(0,bit_)
        mybyte = mybyte//2
    return myint


def xoring_key_file(key,file):
    result = bytearray()
    ending = len(file)%len(key)
    inside = len(file)//len(key)
    low = 0
    z = 0
    for x in range(inside):
        for y in range(low,low+len(key)):
            result.append(file[y]^key[z])
            z = z + 1
        low = low + inside
        z = 0
    z = 0
    for w in range(len(file)-ending,len(file)):
        result.append(file[w]^key[z])
        z = z + 1
    return result

print("   ___\n__/ o |\n   |  |_____ V\n   |        |\n   \________/\n")

#if true, encrypt
#if false, decrypt
deen = confirm_loop("Mode:\n\tEncrypt: 1\n\tDecrypt: 0") 

#if true, file key
#if false, string key
keytype = confirm_loop("Key Type:\n\tFile: 1\n\tString: 0")


#get key
key = input("Enter Key (String or Path): ")

mykey = bytearray()

#if TRUE, retrieve File
if keytype:
    size = 1024
    f = open(key,'rb')
    try: 
        mykey = bytearray(f.read(size))
    finally:
        f.close()
else: #if FALSE, convert to bytearray
    mykey = bytearray(key, "utf-8")

if len(mykey) < 1024:
    buf = 1024 - len(mykey)
    mykey.extend(mykey[0:buf])

file = input("Enter File Path: ")
myfile = bytearray()

if deen:
    f = open(file,'rb')
    try:
        myfile = bytearray(f.read())
    finally:
        f.close()
        #TODO: INCRYPT
        #TODO: check that file <= 12mb 
        encrypt_v1 = xoring_key_file(mykey,myfile)
        print(encrypt_v1)
        """
        try:
            with open("xoringEX.txt","wb") as f:
                f.write(encrypt_v1)
        except FileExistsError:
            print("already exists")
        """
else: 
    #TODO: DECRYPT
    print("hi")

