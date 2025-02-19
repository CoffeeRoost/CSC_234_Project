#Start of project here
import pickle
import os

#Pads file with file_size and file_extension
#Padding bytes are the string "HARE"
def myAdditions(file):    
    pad = bytearray("HARE".encode('utf-8'))
    myBuff = bytearray()

    #Convert size to KB
    fileS = str(os.stat(file).st_size)
    encoded = fileS.encode('utf-8')
    filesize = bytearray(encoded)

    split_tup = os.path.splitext(file)
    file_extension = split_tup[-1]

    file_ex=bytearray(file_extension.encode('utf-8'))
    
    myBuff.extend(filesize)
    myBuff.extend(pad)
    myBuff.extend(file_ex)
    myBuff.extend(pad)

    return myBuff


#Grabs the first instance of the bytes "HARE" in a bytearray
#Error: -1 if not found
def unCover(myArray):
    pad = bytearray("HARE".encode('utf-8'))

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

#Function to check size of encrypt file is < 12mb
#Input: file path
#Output: Boolean
def checkFileSize(file):
    if(os.path.exists(file) == False):
        return False

    #Convert size to KB
    filesize = os.stat(file).st_size/1024
    return filesize < 12000


#So weird thing is if you take out an element of bytearray
#it's considered an integer. 
#Input: a "byte" (an int)
#Output: an int list of 0s and 1s representing a binary number
#Expand into loop to output entirety of bytearray
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

#Extends key to 1024 bytes
def extending_key(key):
    result = key
    count = 0
    for x in range(len(key),1024):
        if(count == len(key)):
            count = 0
        result.extend(key[count])

    return result

# Huffman encoding starts here

import heapq                                # Implementing a priority queue (min-heap)
from collections import Counter             # Counts occurrences of each byte in data

# Huffman Tree Node
class Node:
    def __init__(self, byte, freq):
        self.byte = byte                   # Byte/character this node represents
        self.freq = freq                   # Frequency of this specific byte in the data
        self.left = None                   # Left child
        self.right = None                  # Right child

    def __lt__(self, other):               # Min-heap comparison
        return self.freq < other.freq      # Sorting nodes by frequency


# Step 1: Build Frequency Table
def build_freq_table(data):
    return Counter(data)                   # Returns {byte: count}

# Step 2: Build Huffman Tree
def build_huffman_tree(freq_table):
    heap = [Node(byte, freq) for byte, freq in freq_table.items()]
    heapq.heapify(heap)                    # Building Min-heap based on frequency

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)   # Poping two lowest frequency nodes and cfeate a merged node i.e. Internal node (sum of both)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)                  # Push merged node, adding it back into the heap

    return heap[0]                                    # Root of Huffman Tree

# Step 3: Generate Huffman Codes
def generate_codes(root, prefix="", code_map={}):
    if root:
        if root.byte is not None:                            # Leaf node
            code_map[root.byte] = prefix                     # Storing the binary Huffman code for the byte in code_map
        generate_codes(root.left, prefix + "0", code_map)    # Recursively assigning codes - traverse left append "0" to prefix
        generate_codes(root.right, prefix + "1", code_map)   # Recursively assigning codes - traverse right append "1" to prefix
    return code_map

# Step 4: Encode Data Using Huffman Codes
def huffman_encode(data, code_map):                          # Looking up the Huffman code in code_map for each byte in data (bytearray containing the original data)
    return "".join(code_map[byte] for byte in data)          # Concatenating the huffman code into a single binary string

# Step 5: Convert Binary String to Bytearray
def binary_string_to_bytearray(binary_string):
    padded_length = 8 - (len(binary_string) % 8)
    binary_string += "0" * padded_length                                                          # Padding to full bytes
    byte_data = bytearray(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))  # Splitting the binary string into chunks of 8 bits
    return byte_data, padded_length                                                               # Return bytearray and padding info(indicates the number of '0' bits added)

# Main Function: Huffman Encoding (Returning Bytearray)
def huffman_compress(byte_data):
    freq_table = build_freq_table(byte_data)                                    # Step 1: Frequency Table
    huffman_tree = build_huffman_tree(freq_table)                               # Step 2: Huffman Tree
    huffman_codes = generate_codes(huffman_tree)                                # Step 3: Huffman Codes
    encoded_binary = huffman_encode(byte_data, huffman_codes)                   # Step 4: Encode Data
    compressed_bytearray, padding = binary_string_to_bytearray(encoded_binary)  # Step 5: Convert to Bytearray
    
    return compressed_bytearray, huffman_tree, huffman_codes, padding           # Return results

# Example Usage
# if __name__ == "__main__":
#     # Example bytearray (Simulating XOR-encrypted data)
#     result = bytearray(b"Example XOR encrypted data to be compressed!")

#     # Perform Huffman Compression
#     compressed_data, tree, codes, padding = huffman_compress(result)

#     print("Original Size:", len(result), "bytes")
#     print("Compressed Size:", len(compressed_data), "bytes")
#     print("Huffman Codes:", codes)
#     print("Padding Added:", padding)
#     print("Compressed Data (Bytearray):", compressed_data)


"""
Decode functions (idk how you wanna format. i tried to fit the general design)
"""
def bytearray_to_binary_string(byte_data, padding):
    binary_string = "".join(format(byte, '08b') for byte in byte_data)
    if padding > 0:
        binary_string = binary_string[:-padding]
    return binary_string

def huffman_decode(binary_string, tree):
    decoded_bytes = bytearray()
    node = tree
    for bit in binary_string:
        node = node.left if bit == '0' else node.right
        if node.left is None and node.right is None:
            decoded_bytes.append(node.byte)
            node = tree
    return decoded_bytes
"""
End of decode functions
"""

#Input Start
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
    script = key.encode('utf-8')
    mykey = bytearray(script)

if len(mykey) < 1024:
    mykey = extending_key(mykey)

file = input("Enter File Path: ")
myfile = bytearray()

if deen:
    if checkFileSize(file):
        f = open(file,'rb')
        try:
            myfile = bytearray(f.read())
        except:
            f.close()
            print("File Not Available")
        else:
            f.close()
            #TODO: ENCRYPT
            encrypt_v0 = myAdditions(file)
            encrypt_v0.extend(myfile)
            
            encrypt_v1 = xoring_key_file(mykey,encrypt_v0)
            
            """
            I moved huffman here because we are not operating with main(). You guys can adjust however you want.
            """
            compressed_data, tree, codes, padding = huffman_compress(encrypt_v1)
            print("Original Size:", len(encrypt_v1), "bytes")
            print("Compressed Size:", len(compressed_data), "bytes")
            print("Huffman Codes:", codes)
            print("Padding Added:", padding)
            print("Compressed Data (Bytearray):", compressed_data)
            with open("huffmanCompressed.txt","wb") as hc:
                pickle.dump((compressed_data, padding, tree), hc)
                
            #submit_vf = bytes(encrypt_v1)
    else:
        print("File to large to encrypt")
else: 
    """
    Test by: entering the same key/file.
    File: huffmanCompressed.txt
    """
    with open(file, 'rb') as f:
        compressed_data, padding, tree = pickle.load(f)
    
    binary_string = bytearray_to_binary_string(compressed_data, padding)
    xor_encrypted_data = huffman_decode(binary_string, tree)
    
    decrypted_data = xoring_key_file(mykey, xor_encrypted_data)
    
    output_path = input("Where to save??: ").strip()
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    
    print("View your file here:", output_path)



