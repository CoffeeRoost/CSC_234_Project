#Start of project here
import pickle
import os
import sys
import random
import math
import numpy as np
import ast
import time
import cProfile
import pstats

from mpmath import mp

mp.dps = 9999 # Set precision for desired decimal places
TEN_THOUSAND_PI = "3"+str(mp.pi)[2:]
print(TEN_THOUSAND_PI)

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


"""
def xoring_key_file(key,file):
    result = bytearray()
    count = 0
    for x in file:
        if count == len(key):
            count = 0
        result.append(x ^ key[count])
        count += 1
    
    return result
"""

def xoring_key_file(key,file):
    result = [0] * len(file)
    count = 0
    for x in range(len(file)):
        if count == len(key):
            count = 0
        result[x] = file[x] ^ key[count]
        count += 1

    return bytearray(result)


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

def extending_key(key,size):
    PI_pos = hash_key(key)
    result = [0] * size

    for x in range(size):
        if PI_pos == len(TEN_THOUSAND_PI):
            PI_pos = 0
        result[x] = int(TEN_THOUSAND_PI[PI_pos])
        PI_pos += 1

    return bytearray(result)


#Subsitute for no consistent hash function
#1. Consider the key by every individual byte
#2. Multiply that byte by its position in the key
#3. Add them together to determine the PI_position
#4. Do this for the first (number to be determined) bytes
#5. Mod sum by 10000
def hash_key(key):
    sum = 0
    for i,byte in enumerate(key):
        sum += i * byte
    
    return sum % 10000


def extending_file(file):
    req_pad = 16000000 - len(file)

    my_bytes = np.random.randint(len(file),size=req_pad)
    result = [0] * req_pad

    for x in range(req_pad):
        result[x] = file[my_bytes[x]]

    return bytearray(result)

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

#### CUBE SHIFT ####

def generate_random_data(size):
    return bytearray(random.getrandbits(8) for _ in range(size))

def bitarray_from_bytearray(bytearr):
    return np.unpackbits(np.array(bytearr, dtype=np.uint8))

def create_hypercube_of_squares(bitarr, hypercube_length, square_length, num_dimensions):
    """Creates a hypercube where each cell contains a square (square_length x square_length)."""
    cube_size = hypercube_length ** num_dimensions * (square_length * square_length)
    reshaped = bitarr[:cube_size].reshape((hypercube_length,) * num_dimensions + (square_length, square_length))
    return reshaped

def create_index_cube(hypercube_length, num_dimensions):
    """Creates a hypercube where each cell contains its own multi-dimensional index."""
    indices = np.indices((hypercube_length,) * num_dimensions)
    index_cube = np.stack(indices, axis=-1)
    return index_cube

def apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions):
    """Applies rotations based on the key to the index cube."""
    rotated_index_cube = np.copy(index_cube)  # Avoid modifying the original
    index = 0
    for coords in np.ndindex((hypercube_length,) * num_dimensions):
        for dim in range(num_dimensions):
            shift = key[index] % hypercube_length
            rotated_index_cube = np.roll(rotated_index_cube, shift, axis=dim)
            index += 1
    return rotated_index_cube

def reverse_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions):
    """Applies reverse rotations based on the key to the index cube."""
    rotated_index_cube = np.copy(index_cube)  # Avoid modifying the original
    index = len(key) - 1
    for coords in reversed(list(np.ndindex((hypercube_length,) * num_dimensions))):
        for dim in reversed(range(num_dimensions)):
            shift = key[index] % hypercube_length
            rotated_index_cube = np.roll(rotated_index_cube, -shift, axis=dim)
            index -= 1
    return rotated_index_cube

def encrypt_byte_array(byte_array, key, hypercube_length, square_length, num_dimensions):
    """Encrypts the byte array into a hypercube of squares using the index cube rotation."""
    bit_array = bitarray_from_bytearray(byte_array)
    cube_of_squares = create_hypercube_of_squares(bit_array, hypercube_length, square_length, num_dimensions)
    index_cube = create_index_cube(hypercube_length, num_dimensions)
    rotated_index_cube = apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)
    encrypted_cube = np.zeros_like(cube_of_squares)  # Initialize with zeros, same shape and type

    for coords in np.ndindex((hypercube_length,) * num_dimensions):
        original_coords = tuple(rotated_index_cube[coords])
        encrypted_cube[coords] = cube_of_squares[original_coords]

    return encrypted_cube

def decrypt_hypercube(encrypted_cube, key, hypercube_length, square_length, num_dimensions):
    """Decrypts the hypercube of squares back into a byte array using reversed index cube rotation."""
    index_cube = create_index_cube(hypercube_length, num_dimensions)
    rotated_index_cube = reverse_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)
    decrypted_cube = np.zeros_like(encrypted_cube)  # Initialize with zeros, same shape and type

    for coords in np.ndindex((hypercube_length,) * num_dimensions):
        original_coords = tuple(rotated_index_cube[coords])
        decrypted_cube[coords] = encrypted_cube[original_coords]  # Fill by reverse lookup

    # Flatten the cube back into a bit array
    bit_array = decrypted_cube.flatten()

    # Pack the bit array back into a byte array.  Must be multiple of 8
    bit_array = bit_array[:len(bit_array) - (len(bit_array) % 8)]
    byte_array = np.packbits(bit_array).tobytes()

    return byte_array

def pad_with_pi(data, required_size):
    """Pads the data with digits of pi until it reaches the required size."""
    pi_digits = str(math.pi).replace('.', '')  # Remove decimal point
    padded_data = bytearray(data)

    pi_index = 0
    while len(padded_data) < required_size:
        digit_pair = pi_digits[pi_index:pi_index + 2]
        if len(digit_pair) == 2:
            try:
                padded_data.append(int(digit_pair)) #convert pi digit pairs to bytes
            except ValueError:
                padded_data.append(0) #if there is an error, pad with a zero 
        else:
            padded_data.append(0) #pad with 0 if you can't get 2 digits.

        pi_index = (pi_index + 2) % len(pi_digits) #cycle through the digits of pi
    return padded_data[:required_size]  # Truncate if necessary

def unpad_with_pi(data):
    """Removes padding from the data, assuming it was padded with the digits of pi."""
    pi_digits = str(math.pi).replace('.', '')
    pi_bytes = bytearray()

    # Convert the first 10 digits of pi to bytes, assuming pairs of digits are bytes
    for i in range(0, 20, 2):  # Check first 20 digits because key padding might need more than data padding
        digit_pair = pi_digits[i:i + 2]
        if len(digit_pair) == 2:  # Handle cases where pi_digits has an odd length
            try:
                pi_bytes.append(int(digit_pair))  # Convert pi digit pairs to bytes
            except ValueError:
                return data  # If there's an error, assume no padding and return original data
        else:
            break # Stop when there's not a full digit pair

    # Convert the data to bytearray if it isn't already.
    data = bytearray(data)

    # Find the index of the pi sequence in the data
    try:
      index = data.index(pi_bytes[0])
      if len(data) - index >= len(pi_bytes):
          if data[index:index + len(pi_bytes)] == pi_bytes:
              return data[:index]  # Truncate data at the start of the pi sequence
          else:
              return data #didn't find it so return the data unedited
      else:
        return data #pi sequence too short to match
    except ValueError:  # Substring not found
        return data # didn't find it so return the data unedited


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

def save_array_to_file(array, filename):
    np.save(filename, array)
    print(f"Array saved to {filename}.npy")

def load_array_from_file(filename):
    array = np.load(filename + ".npy")
    return array
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
    #mykey = extending_key(mykey)
    mykey.extend(extending_key(mykey,1024-len(mykey)))


file = input("Enter File Path: ")

if(deen):
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

    #encrypt_v0.extend(extending_file(encrypt_v0))
    #encrypt_v1 = xoring_key_file(mykey,encrypt_v0)

    encrypt_v1 = xoring_key_file(mykey,encrypt_v0)
    encrypt_v1.extend(extending_file(encrypt_v1))
            
    """
    I moved huffman here because we are not operating with main(). You guys can adjust however you want.
    """
    compressed_data, tree, codes, padding = huffman_compress(encrypt_v1)
    print("Original Size:", len(encrypt_v1), "bytes")
    print("Compressed Size:", len(compressed_data), "bytes")
    #print("Huffman Codes:", codes)
    #print("Padding Added:", padding)
    #print("Compressed Data (Bytearray):", compressed_data)
    print(type(compressed_data))  # Should be bytearray or bytes
    print(type(padding))          # Should be int
    print(type(tree))   
    with open("huffmanCompressed.pkl","wb") as hc:
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
    with open("huffmanCompressed.pkl", 'rb') as f:
        compressed_data, padding, tree = pickle.load(f)
    key = mykey
    
    # Constants
    hypercube_length, square_length, num_dimensions = 8, 160, 3

    # Calculate required sizes
    data_size = hypercube_length**num_dimensions * square_length*square_length // 8
    key_size = (hypercube_length**num_dimensions) * num_dimensions

    # Pad with pi
    original_byte_array = pad_with_pi(compressed_data, data_size)
    
    print(original_byte_array[:10])
    key = pad_with_pi(key, key_size)
    encrypted_cube = encrypt_byte_array(original_byte_array, key, hypercube_length, square_length, num_dimensions)
    save_array_to_file(encrypted_cube, "shifted_array")


    # print("diff:")
    # print((np.array(hypercube)-np.array(og_hypercube)))

else: 
    """
    Test by: entering the same key/file.
    File: huffmanCompressed.txt
    """
    shifted_hypercube = load_array_from_file("shifted_array")

    key = mykey
    
    # Constants
    hypercube_length, square_length, num_dimensions = 8, 160, 3

    # Calculate required sizes
    data_size = hypercube_length**num_dimensions * square_length*square_length // 8
    key_size = (hypercube_length**num_dimensions) * num_dimensions

    key = pad_with_pi(key, key_size)
    # decrypt then unpad with pi
    decrypted_byte_array = decrypt_hypercube(shifted_hypercube, key, hypercube_length, square_length, num_dimensions)
    
    unpadded_byte_array = unpad_with_pi(decrypted_byte_array)
    
    print(unpadded_byte_array[:10])

    with open(file, 'rb') as f:
        compressed_data, padding, tree = pickle.load(f)

    binary_string = bytearray_to_binary_string(unpadded_byte_array, padding)
    xor_encrypted_data = huffman_decode(binary_string, tree)

    decrypted_data = xoring_key_file(mykey, xor_encrypted_data)
    
    #Uncover file size in bytes
    pos = unCover(decrypted_data)
    next = decrypted_data[pos+4:]

    #Uncover file type
    pos = unCover(next)
    fileType = next[:pos].decode('utf-8',errors="ignore")

    decrypted_data = next[pos+4:]


    output_path = input("Where to save??: ").strip()
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    
    print("View your file here:", output_path)
