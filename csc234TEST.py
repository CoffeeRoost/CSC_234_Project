#Start of project here
import pickle
import os
import sys
import random
import math
import numpy as np
import ast

#While loop that checks if a file exists
#If Mode = 0: Checks Key File Existence
#If Mode = 1: Checks File Existence and Size Limit
#Loops until valid Name is Given
def fileCheck(file_name,mode):
    exist = 3
    intry = False
    while not intry:
        exist = checkFileSize(file_name,mode)
        match exist:
            case -1:
                print("File does not exist")
            case 0:
                print("File exceeds 12mb maximum")
            case 1:
                return file_name
        
        if mode == 0:
            file_name = input("Enter Key Path: ")
        else:
            file_name = input("Enter File Path: ")
        


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


#Input: file path, mode representing if key or file check
#Mode = 0 (key), Mode = 1 (file)
#Depending on the Mode, it checks for different things
#Key Mode: If the file exists or not
#File Mode: If the file exists or not, or if it exceeds 12mb
def checkFileSize(file,mode):
    if(os.path.exists(file) == False):
        return -1
    
    if(mode == 0): return 1

    #Convert size to KB
    filesize = os.stat(file).st_size/1024

    if filesize > 12000:
        return 0
    else:
        return 1


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
    count = 0
    for x in file:
        if count == len(key):
            count = 0
        result.append(x ^ key[count])
        count += 1
    return result


#Extends key to 1024 bytes
def extending_key(key):
    result = key
    count = 0
    for x in range(len(key),1024):
        if(count == len(key)):
            count = 0
        result.append(key[count])
        count += 1

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
Cube Shift################################################################
"""
def hex_to_bits(hex_string):
    # Initialising hex string 
    
    # Printing initial string 
    print ("Initial string", hex_string) 
    
    # Code to convert hex to binary 
    n = int(hex_string, 16) 
    bStr = bin(n)[2:] #changed here
    #res = bStr
    return(bStr)
    
def pad_right(text, length, char=' '):
    """Pads the given string on the right with the specified character until it reaches the desired length.

    Args:
        text: The string to pad.
        length: The desired total length of the padded string.
        char: The character to use for padding (default is space).

    Returns:
        The padded string, or the original string if it's already long enough.
    """
    if len(text) >= length:
        return text
    return text + char * (length - len(text))

    

def create_hypercube_and_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    """Creates a hypercube, populates with specific data, and applies bit shifts using a specific key."""

    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = int((2**(total_cube_size))*dimension*(2**bit_shift_size)*4/4)
      print(size)
      if len(key_hex) < size:
        key_temp = pad_right(key_hex, size, "0")
      
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      
      else:
        key_temp = key_hex
      return(key_temp)
    # total_key_length = (2**(total_cube_size))*3*(2**bit_shift_size) one mini key per cell, three dimensions per mini key, then bit shift size per dimension
    #convert str to hex (base 16 int)
    
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str = hex_to_bits(pad_or_truncate_key(key_hex))
    key_str = pad_right(key_str, int((2**(total_cube_size))*dimension*(2**bit_shift_size)), "0")

    print(key_str)
    arbitrary_key = []
    #TODO: see if ravel just works here xd
    for cube_coordinate in range(2**total_cube_size): #loop through each cell of the cube
        key_shifts = []
        for dim in range(dimension): #then each dimension e.g. x, y, z
            cube_offset = cube_coordinate*dimension*(2**bit_shift_size) #increases when cube_coordinate increments
            dimension_offset = dim*(2**bit_shift_size) #'                  ' when dim increments
            key_shifts.append(key_str[cube_offset+dimension_offset:cube_offset+dimension_offset+(2**bit_shift_size)]) #+2**bit_shift_size gives the bits needed to shift
        arbitrary_key.append(key_shifts)

    hypercube_shape = (cube_side_length,) * dimension
    hypercube = np.array(cube_data).reshape(hypercube_shape)


    def rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        """hypercube: original hypercube
        arbitrary_key: key we're rotating with
        shift_dimension: e.g. x y z
        line_index: grabbed via np.ndindex(hypercube.shape); tells us the line in the shift dimension's coordinates
        e.g. if shift dim = 1, would look like [0,0,0], [0,1,0], [0,2,0]..."""
        shifted_hypercube = hypercube.copy() #make a copy so we can do stuff with it
        
        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape) #ravel = un un ravel, i.e. wrap up a line into a cube
        shift_amount = arbitrary_key[key_index][shift_dimension] #find the shift amount (see above)
        
        line_slice = list(line_index) #find the initial coordinate for the line we're slicing
        for cell in range(cube_side_length):
          
          temp_slice = list(line_slice) #make a copy of the slice so it doesn't self reference
          temp_slice[shift_dimension] = cell 
          
          new_coords = list(temp_slice)
          #for demonstration, print before and after some specific times
          if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
            print("Before:")
            print(new_coords)


          new_coords[shift_dimension] = (new_coords[shift_dimension] + int(shift_amount,2)) % cube_side_length
          

          #for demonstration
          if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
            print("After:")
            print(new_coords)
          shifted_hypercube[tuple(new_coords)] = hypercube[tuple(temp_slice)]
          
        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()
    
    shift_history = []

    for line_index in np.ndindex(*hypercube.shape): #loops through each line first
      print(line_index)
      for shift_dimension in range(dimension): #then shifts all dimensions in each line.

        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = arbitrary_key[key_index][shift_dimension]
        #also for demonstration
        if ((shift_amount == "0001") or (shift_amount == "0101")):
          print(f"Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
        
        shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, list(line_index))
        shift_history.append((shift_dimension, line_index, shift_amount, shifted_hypercube.copy(),key_index))
    #prints last two
    for i in range(len(shift_history)-2, len(shift_history)):
      shift_dimension, line_index, shift_amount, shifted_hypercube, key_index = shift_history[i]
      print(f"Shift: Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
      print(shifted_hypercube)

    # print("\nRandom Ahh Key:")
    # print(arbitrary_key)


    return hypercube, shifted_hypercube
##########################################################################
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

def iterate_ndindex_backwards_generator(shape):
    """Iterates through an np.ndindex object backwards using a generator (memory-efficient).

    Args:
        shape: The shape of the ndarray to index. Can be a tuple or a list.

    Yields:
        Tuples representing the indices, in reverse order of np.ndindex.
    """
    shape = tuple(shape)  # Ensure it's a tuple
    total_size = np.prod(shape, dtype=np.intp) # Calculate total number of indices

    for i in range(total_size - 1, -1, -1): # Iterate backwards through the linear index
        index = np.unravel_index(i, shape)  # Convert linear index to multi-dimensional index
        yield index

def reverse_hypercube_and_reverse_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    """Creates a hypercube, populates with specific data, and applies bit shifts using a specific key."""
    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = int((2**(total_cube_size))*dimension*(2**bit_shift_size)*4/4)
      if len(key_hex) < size:
        key_temp = pad_right(key_hex, size, "0")
      
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      
      else:
        key_temp = key_hex
      return(key_temp)

    # total_key_length = (2**(total_cube_size))*3*(2**bit_shift_size) one mini key per cell, three dimensions per mini key, then bit shift size per dimension
    #convert str to hex (base 16 int)
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str_to_flip = hex_to_bits(pad_or_truncate_key(key_hex))
    key_str_to_flip = pad_right(key_str_to_flip, int((2**(total_cube_size))*dimension*(2**bit_shift_size)), "0")



    arbitrary_key = []
    #TODO: see if ravel just works here xd
    for cube_coordinate in range(2**total_cube_size): #loop through each cell of the cube
        key_shifts = []
        for dim in range(dimension): #then each dimension e.g. x, y, z
            cube_offset = cube_coordinate*dimension*(2**bit_shift_size) #increases when cube_coordinate increments
            dimension_offset = dim*(2**bit_shift_size) #'                  ' when dim increments

            bits_to_check = key_str_to_flip[cube_offset+dimension_offset:cube_offset+dimension_offset+bit_shift_size]
            
            new_int_shift = (cube_side_length-int(bits_to_check, 2)) % cube_side_length
            new_bit_shift = bin(new_int_shift)[2:].zfill(bit_shift_size)
            print(new_bit_shift)
            key_shifts.append(new_bit_shift) #+2**bit_shift_size gives the bits needed to shift
        arbitrary_key.append(key_shifts)

    hypercube_shape = (cube_side_length,) * dimension
    hypercube = np.array(cube_data).reshape(hypercube_shape)


    def rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        """hypercube: original hypercube
        arbitrary_key: key we're rotating with
        shift_dimension: e.g. x y z
        line_index: grabbed via np.ndindex(hypercube.shape); tells us the line in the shift dimension's coordinates
        e.g. if shift dim = 1, would look like [0,0,0], [0,1,0], [0,2,0]..."""
        shifted_hypercube = hypercube.copy() #make a copy so we can do stuff with it
        
        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape) #ravel = un un ravel, i.e. wrap up a line into a cube
        shift_amount = arbitrary_key[key_index][shift_dimension] #find the shift amount (see above)
        
        line_slice = list(line_index) #find the initial coordinate for the line we're slicing
        for cell in range(cube_side_length):
          
          temp_slice = list(line_slice) #make a copy of the slice so it doesn't self reference
          temp_slice[shift_dimension] = cell 
          
          new_coords = list(temp_slice)
          #for demonstration, print before and after some specific times
          if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
            print("Before:")
            print(new_coords)


          new_coords[shift_dimension] = (new_coords[shift_dimension] + int(shift_amount,2)) % cube_side_length
          

          #for demonstration
          if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
            print("After:")
            print(new_coords)
          shifted_hypercube[tuple(new_coords)] = hypercube[tuple(temp_slice)]
          
        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()
    
    shift_history = []

    for line_index in iterate_ndindex_backwards_generator(hypercube.shape): #loops through each line backwards
      print(line_index)
      for shift_dimension in range(dimension): #then shifts all dimensions in each line.

        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = arbitrary_key[key_index][shift_dimension]
        #also for demonstration``
        if ((shift_amount == "0001") or (shift_amount == "0101")):
          print(f"Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
        
        shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, list(line_index))
        shift_history.append((shift_dimension, line_index, shift_amount, shifted_hypercube.copy(),key_index))
    #prints last two
    for i in range(len(shift_history)-2, len(shift_history)):
      shift_dimension, line_index, shift_amount, shifted_hypercube, key_index = shift_history[i]
      print(f"Shift: Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
      print(shifted_hypercube)

    # print("\nRandom Ahh Key:")
    # print(arbitrary_key)


    return hypercube, shifted_hypercube

def bytearray_to_bits_array(byte_data):
    return [int(bit) for byte in byte_data for bit in format(byte, '01b')]
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

    key = fileCheck(key,0)

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

file = fileCheck(file,1)

myfile = bytearray()


if deen:
    f = open(file,'rb')
    try:
        myfile = bytearray(f.read())
    finally:
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
    print(type(compressed_data))  # Should be bytearray or bytes
    print(type(padding))          # Should be int
    print(type(tree))   
    with open("huffmanCompressed.txt","wb") as hc:
       pickle.dump((compressed_data, padding, tree), hc, protocol=4)
    
    submit_vf = bytes(encrypt_v1)

    """
    try:
        with open("xoringEX.txt","wb") as f:
            f.write(submit_vf)
    except FileExistsError:
            print("already exists")
    """
    #submit_vf = bytes(encrypt_v1)
    # with open(file, 'rb') as f:
    #     compressed_data, padding, tree = pickle.load(f)
    
    # dimension = 3
    # total_cube_size = 18#2^3
    # bit_shift_size = 2 #2^2 shifting by 0 to 3
    # cube_data = bytearray_to_bits_array(compressed_data) #total 2^3 = 8 entries 
    # if len(cube_data) < 262144:
    #     cube_data.extend([0] * (262144 - len(cube_data)))
    # # key_hex = ""
    # for i in range(12):
    #     hypercube, shifted_hypercube = create_hypercube_and_shift(
    #     dimension, total_cube_size, mykey, bit_shift_size, cube_data
    # )

    # print(hypercube)

    # with open("cube_data.txt", "w") as file:
    #     file.write(str(cube_data))  # Convert the list to a string and write it

    # print("Array written to cube_data.txt")

else: 
    """
    Test by: entering the same key/file.
    File: huffmanCompressed.txt
    """
    # dimension = 3
    # total_cube_size = 18#2^3
    # bit_shift_size = 2 #2^2 shifting by 0 to 3
    # with open("cube_data.txt", "r") as file:
    #     shifted_hypercube = ast.literal_eval(file.read()) 
    # with open(file, 'rb') as f:
    #     compressed_data, padding, tree = pickle.load(f)
        
    # hypercube, og_hypercube = reverse_hypercube_and_reverse_shift(
    # dimension, total_cube_size, mykey, bit_shift_size, shifted_hypercube
    # )

    # print(og_hypercube)
    with open(file, 'rb') as f:
        compressed_data, padding, tree = pickle.load(f)

    binary_string = bytearray_to_binary_string(compressed_data, padding)
    xor_encrypted_data = huffman_decode(binary_string, tree)

    decrypted_data = xoring_key_file(mykey, xor_encrypted_data)
    
    #Uncover file size in bytes
    pos = unCover(decrypted_data)
    fileSize = int(decrypted_data[:pos].decode('utf-8',errors="ignore"))

    next = decrypted_data[pos+4:]

    #Uncover file type
    pos = unCover(next)
    fileType = next[:pos].decode('utf-8',errors="ignore")

    decrypted_data = next[pos+4:]

    output_path = input("Where to save??: ").strip()
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    
    print("View your file here:", output_path)
