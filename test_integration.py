import pytest
import os
import pickle
import numpy as np

# Import functions and relevant constants from the script to be tested
from csc234TEST import (
    myAdditions,
    extending_key,
    xoring_key_file,
    extending_file,
    huffman_compress,
    pad_with_pi,
    unpad_with_pi,
    encrypt_byte_array,
    decrypt_hypercube,
    bytearray_to_binary_string,
    huffman_decode,
    unCover,
    paddingFile, # Imported constant
    paddingFilelen, # Imported constant
    # Node, build_freq_table, generate_codes, # etc. if needed for more granular tests
)

HYPERCUBE_LENGTH = 8
SQUARE_LENGTH = 160
NUM_DIMENSIONS = 3

DATA_SIZE = HYPERCUBE_LENGTH**NUM_DIMENSIONS * SQUARE_LENGTH*SQUARE_LENGTH // 8
KEY_SIZE_CUBE = (HYPERCUBE_LENGTH**NUM_DIMENSIONS) * NUM_DIMENSIONS

# --- START OF PLACEHOLDER FOR MORE TEST INPUTS ---
# You can add more test cases to this list.
# Each tuple should contain:
# (test_id, input_content, key_content, is_key_file)
# test_id: A unique string identifier for the test case.
# input_content: Bytes for the input file.
# key_content: Bytes for the key (string or file content).
# is_key_file: Boolean, True if key_content is for a file, False if it's a string.
# --- Example Test Cases ---
test_data_params = [
    (
        "binary_data_string_key",
        os.urandom(1024), # 1KB of random binary data
        b"binarykey!@#",
        False,
    ),
    (
        "empty_content_string_key", # Original file is empty
        b"",
        b"keyforempty",
        False,
    ),
    (
        "single_byte_content_string_key",
        b"A",
        b"singlebytekey",
        False,
    ),
    (
        "short_key_string",
        b"Test with a short key.",
        b"k",
        False,
    ),
    (
        "long_key_string_needs_truncation",
        b"Test with a very long key that will be truncated.",
        os.urandom(2048), # Key longer than 1024 bytes
        False,
    ),
    (
        "all_ones_medium_file_string_key",
        b"\xff" * 5000,
        b"anotherkey_ones",
        False,
    ),
    (
        "all_zeros_medium_file_string_key",
        b"\x00" * 5000,
        b"anotherkey_zeros",
        False,
    )
]
# --- END OF PLACEHOLDER FOR MORE TEST INPUTS ---

PREDEFINED_FILES_TO_TEST = {
    "my_special_testfile_txt": {
        "path": "./testFiles/ORV.txt", # Relative or absolute path
        "key_content": b"key_for_testfile_txt",
        "is_key_file": False
    },
    # "data_with_specific_keyfile": {
    #     "path": "./testFiles/ORV.txt",
    #     "key_content": "keyFiles/Fun.png",
    #     "is_key_file": True
    # },
    # "image_testFile": {
    #     "path": "./testFiles/Fun.png",
    #     "key_content": b"key_for_testfile_txt",
    #     "is_key_file": False
    # },
    "video_testFile": {
        "path": "./testFiles/testVideo.mp4", 
        "key_content": b"key_for_testfile_txt",
        "is_key_file": False
    }
    # "audio_testFile": {
    #     "path": "./testFiles/sound.wav",
    #     "key_content": b"key_for_testfile_txt",
    #     "is_key_file": False
    # }

}

for test_id, file_info in PREDEFINED_FILES_TO_TEST.items():
    file_path_to_read = file_info["path"]
    if os.path.exists(file_path_to_read) and os.path.isfile(file_path_to_read):
        try:
            with open(file_path_to_read, "rb") as f:
                content_bytes = f.read()
            
            key_data = file_info["key_content"]
            is_k_file = file_info["is_key_file"]

            if is_k_file and isinstance(key_data, str) and os.path.exists(key_data):
              
                pass 
            elif is_k_file and isinstance(key_data, bytes):
               
                pass 

            test_data_params.append(
                (
                    test_id,
                    content_bytes,
                    key_data,
                    is_k_file
                )
            )
            print(f"  Added predefined file test: {test_id} from {file_path_to_read}")
        except Exception as e:
            print(f"Warning: Could not read or process predefined file '{file_path_to_read}'. Error: {e}")
    else:
        print(f"Warning: Predefined test file not found: {file_path_to_read} for test_id '{test_id}'")

def prepare_key_bytes(key_content_bytes, is_key_file, tmp_path):
    mykey_ba = bytearray()
    if is_key_file:
        # Temp key file
        key_file_path = tmp_path / "temp_test_key.key"
        with open(key_file_path, "wb") as kf:
            kf.write(key_content_bytes)
        
        with open(key_file_path, 'rb') as f_key:
            mykey_ba = bytearray(f_key.read(1024)) # Max 1024 for key as per original code
    else:
        # Use key string directly
        # script_encoded = key_content_bytes.decode('utf-8', errors='ignore').encode('utf-8') # if key_content_bytes was string
        mykey_ba = bytearray(key_content_bytes)

    if len(mykey_ba) < 1024:
        mykey_ba.extend(extending_key(mykey_ba, 1024 - len(mykey_ba)))
    elif len(mykey_ba) > 1024:
        mykey_ba = mykey_ba[:1024]
    return mykey_ba

@pytest.mark.parametrize(
    "test_id, input_content_bytes, key_content_bytes, is_key_file",
    test_data_params
)
def test_encryption_decryption_cycle(test_id, input_content_bytes, key_content_bytes, is_key_file, tmp_path):
    """
    Tests the full encryption and decryption cycle by calling the sequence of functions
    """
    original_filename_stem = f"test_input_{test_id}"
    original_file_ext = ".txt" # Assume txt for simplicity in test, content matters more
    original_input_filename = original_filename_stem + original_file_ext
    
    input_file_path = tmp_path / original_input_filename
    encrypted_file_path = tmp_path / f"encrypted_output_{test_id}.enc"

    decrypted_verification_path = tmp_path / f"decrypted_output_{test_id}{original_file_ext}"

    # 1. Create original input file in the temporary directory
    with open(input_file_path, "wb") as f:
        f.write(input_content_bytes)
    original_file_size_os = os.stat(input_file_path).st_size
    assert original_file_size_os == len(input_content_bytes)

    # 2. Prepare key (1024 bytes)
    processed_key_1024b = prepare_key_bytes(key_content_bytes, is_key_file, tmp_path)
    assert len(processed_key_1024b) == 1024, "Key processing failed to produce 1024 byte key"

    # --- ENCRYPTION PROCESS  ---
    
    # Read original file content for encryption
    with open(input_file_path, 'rb') as f_orig_content:
        myfile_bytes = bytearray(f_orig_content.read())

    # E1: myAdditions (metadata: size and filename)
    # myAdditions uses the string path of the file
    encrypt_metadata_added = myAdditions(str(input_file_path)) 
    encrypt_metadata_added.extend(myfile_bytes) # Append original file content

    # E2: Xoring with key and extending file (padding to 16MB)
    encrypt_xor_result = xoring_key_file(processed_key_1024b, encrypt_metadata_added)
    # The extending_file function pads the XORed data using bytes from itself to reach 16,000,000 bytes
    encrypt_xor_extended = bytearray(encrypt_xor_result) # Make a copy before extending
    encrypt_xor_extended.extend(extending_file(encrypt_xor_result)) 
    assert len(encrypt_xor_extended) == 16000000 if len(encrypt_xor_result) < 16000000 else len(encrypt_xor_result)


    # E3: Huffman Compression
    # This compresses the potentially large (up to 16MB) encrypt_xor_extended data
    huffman_compressed_data, huffman_tree, _, huffman_padding_bits = huffman_compress(encrypt_xor_extended)

    # E4: Prepare data for Cube Shift (Pickle Huffman metadata, then Pad all components with Pi)
    pickled_huffman_padding = bytearray(pickle.dumps(huffman_padding_bits))
    pickled_huffman_tree = bytearray(pickle.dumps(huffman_tree))

    # Pad data, huffman padding, and tree to DATA_SIZE for cube shift
    cs_padded_compressed_data = pad_with_pi(huffman_compressed_data, DATA_SIZE)
    cs_padded_pickled_padding = pad_with_pi(pickled_huffman_padding, DATA_SIZE)
    cs_padded_pickled_tree = pad_with_pi(pickled_huffman_tree, DATA_SIZE)

    # Pad the 1024-byte key to KEY_SIZE_CUBE for cube shift operations
    cube_shift_key_padded = pad_with_pi(processed_key_1024b, KEY_SIZE_CUBE)

    # E5: Cube Shift Encryption
    encrypted_cube_main_data = encrypt_byte_array(cs_padded_compressed_data, cube_shift_key_padded, HYPERCUBE_LENGTH, SQUARE_LENGTH, NUM_DIMENSIONS)
    encrypted_cube_padding_meta = encrypt_byte_array(cs_padded_pickled_padding, cube_shift_key_padded, HYPERCUBE_LENGTH, SQUARE_LENGTH, NUM_DIMENSIONS)
    encrypted_cube_tree_meta = encrypt_byte_array(cs_padded_pickled_tree, cube_shift_key_padded, HYPERCUBE_LENGTH, SQUARE_LENGTH, NUM_DIMENSIONS)

    # E6: Save encrypted data bundle (mimicking `pickle.dump` in main)
    with open(encrypted_file_path, "wb") as ef:
        pickle.dump((encrypted_cube_main_data, encrypted_cube_padding_meta, encrypted_cube_tree_meta), ef, protocol=4)

    # --- DECRYPTION PROCESS ---
    # D1: Load encrypted data bundle
    with open(encrypted_file_path, 'rb') as f_enc_load:
        loaded_cube_data, loaded_cube_padding_meta, loaded_cube_tree_meta = pickle.load(f_enc_load)

    # D2: Decrypt Cube Shift (key for decryption is the same padded cube_shift_key_padded)
    decrypted_cs_padded_data_bytes = decrypt_hypercube(loaded_cube_data, cube_shift_key_padded, HYPERCUBE_LENGTH, SQUARE_LENGTH, NUM_DIMENSIONS)
    decrypted_cs_padded_padding_bytes = decrypt_hypercube(loaded_cube_padding_meta, cube_shift_key_padded, HYPERCUBE_LENGTH, SQUARE_LENGTH, NUM_DIMENSIONS)
    decrypted_cs_padded_tree_bytes = decrypt_hypercube(loaded_cube_tree_meta, cube_shift_key_padded, HYPERCUBE_LENGTH, SQUARE_LENGTH, NUM_DIMENSIONS)

    # D3: Unpad Pi from decrypted components
    unpadded_huffman_compressed_data = unpad_with_pi(decrypted_cs_padded_data_bytes)
    unpadded_pickled_padding_bytes = unpad_with_pi(decrypted_cs_padded_padding_bytes)
    unpadded_pickled_tree_bytes = unpad_with_pi(decrypted_cs_padded_tree_bytes)
    
    # D4: Unpickle Huffman metadata
    retrieved_huffman_padding_bits = pickle.loads(unpadded_pickled_padding_bytes)
    retrieved_huffman_tree = pickle.loads(unpadded_pickled_tree_bytes)

    # D5: Huffman Decompression
    binary_str_for_decode = bytearray_to_binary_string(bytearray(unpadded_huffman_compressed_data), retrieved_huffman_padding_bits)
    data_after_huffman_decode = huffman_decode(binary_str_for_decode, retrieved_huffman_tree) # This should be encrypt_xor_extended

    # D6: Reverse Xoring (using the original 1024-byte key)
    data_with_metadata_header = xoring_key_file(processed_key_1024b, data_after_huffman_decode) # This should be encrypt_metadata_added

    # D7: Uncover metadata (original filename and size)
    pos_size_end = unCover(data_with_metadata_header)
    assert pos_size_end != -1, f"Padding string for size not found in decrypted data for test: {test_id}"
    retrieved_original_size_str = data_with_metadata_header[:pos_size_end].decode('utf-8', errors="ignore")
    retrieved_original_file_size = int(retrieved_original_size_str)

    remaining_data_after_size = data_with_metadata_header[pos_size_end + paddingFilelen:]
    pos_name_end = unCover(remaining_data_after_size)
    assert pos_name_end != -1, f"Padding string for name not found in decrypted data for test: {test_id}"
    retrieved_original_filename_str = remaining_data_after_size[:pos_name_end].decode('utf-8', errors="ignore")

    # D8: Extract final decrypted content
    final_decrypted_content_bytes = remaining_data_after_size[pos_name_end + paddingFilelen:]
    # Truncate to the original file size stored in the metadata
    final_decrypted_content_bytes = final_decrypted_content_bytes[:retrieved_original_file_size]


    # 3. Write decrypted content to a new file for verification (optional, but good for manual check if needed)
    with open(decrypted_verification_path, "wb") as f_dec_verify:
        f_dec_verify.write(final_decrypted_content_bytes)

    # 4. Assertions: Verify correctness
    assert retrieved_original_filename_str == str(input_file_path), \
        f"Filename mismatch for test {test_id}: expected '{str(input_file_path)}', got '{retrieved_original_filename_str}'"
    
    assert retrieved_original_file_size == original_file_size_os, \
        f"Original file size metadata mismatch for test {test_id}: expected {original_file_size_os}, got {retrieved_original_file_size}"

    assert len(final_decrypted_content_bytes) == len(input_content_bytes), \
        f"Decrypted content size mismatch for test {test_id}: expected {len(input_content_bytes)}, got {len(final_decrypted_content_bytes)}"
    
    assert final_decrypted_content_bytes == input_content_bytes, \
        f"Decrypted content does not match original content for test {test_id}"

