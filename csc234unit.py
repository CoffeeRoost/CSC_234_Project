import pytest
import os
import csc234TEST as mod

myTestingFile = 'myTestingFile.txt'
myFileStringLength = len(myTestingFile)

myStringPI_size = 10000

myMaxFileSize = 16000000

myPadding = '?>*:||:*<?'
myPaddingStringLength = len(myPadding)

myKeySize = 1024

myInRangeFileSize = 1024
myFileSizeStringLength = len(str(myInRangeFileSize))

def stripped_fileCheck(file_name,mode):
    exist = mod.checkFileSize(file_name,mode)
    match exist:
        case -2:
            return "File is empty"
        case -1:
            return "File does not exist"
        case 0:
            return "File exceeds 12mb maximum"
        case 1:
            return file_name
        
def stripped_confirm_loop(ans):
    match ans:
        case "0":
            return False
        case "1":
            return True
        case _:
            return -1

#creates a myKeySize bytearray
@pytest.fixture
def myKeySize_byte_array():
    my_bytes = mod.np.random.randint(255,size=myKeySize)
    result = bytearray()
    for byte in my_bytes:
        result.append(byte)
    return result

@pytest.fixture
def zero_byte_array():
    result = bytearray()
    for i in range(myInRangeFileSize):
        result.append(0)
    return result

#creates a myInRangeFileSize bytearray
@pytest.fixture
def myFileSize_byte_array():
    my_bytes = mod.np.random.randint(255,size=myInRangeFileSize)
    result = bytearray()
    for byte in my_bytes:
        result.append(byte)
    return result

#creates a 16mb bytearray
@pytest.fixture
def sixteen_mega_byte_array():
    my_bytes = mod.np.random.randint(255,size=myMaxFileSize)
    result = bytearray()
    for byte in my_bytes:
        result.append(byte)
    return result

#creates an empty file
@pytest.fixture
def emptyFile():
    with open(myTestingFile,'wb+') as fp:
        pass
    return myTestingFile

#deletes the file
@pytest.fixture
def nonexistentFile():
    os.remove(myTestingFile)
    return myTestingFile

#creates a file full of 16mb
@pytest.fixture
def exceedFile(sixteen_mega_byte_array):
    with open(myTestingFile, "wb+") as f:
        f.write(sixteen_mega_byte_array)
    return myTestingFile

#creates a file with myFileSize bytes
@pytest.fixture
def in_range_file(myFileSize_byte_array):
    with open(myTestingFile, "wb+") as f:
        f.write(myFileSize_byte_array)
    return myTestingFile

@pytest.fixture
def myAddition_padding(in_range_file):
    pad = bytearray(myPadding.encode('utf-8'))
    myBuff = bytearray()

    #Grab size
    fileS = str(myInRangeFileSize)
    encoded = fileS.encode('utf-8')
    filesize = bytearray(encoded)

    file_ex=bytearray(in_range_file.encode('utf-8'))
    
    myBuff.extend(filesize)
    myBuff.extend(pad)
    myBuff.extend(file_ex)
    myBuff.extend(pad)

    return myBuff


@pytest.mark.parametrize(
        ('input_fc','expected_fc','mode_fc'),
        (
            ('emptyFile','File is empty',1),
            ('emptyFile','File is empty',2),
            ('exceedFile','File exceeds 12mb maximum',1),
            ('nonexistentFile','File does not exist',2),
            ('exceedFile','exceedFile',2),
            ('in_range_file','in_range_file',1),
            ('nonexistentFile','File does not exist',1),
            ('in_range_file','in_range_file',0),
            ('nonexistentFile','File does not exist',0),
        ),
)


#fileCheck Tests
def test_fileCheck(input_fc,expected_fc,mode_fc,request):
    if(input_fc == expected_fc):
        expected_fc = request.getfixturevalue(expected_fc)

    input_fc = request.getfixturevalue(input_fc)
    assert stripped_fileCheck(input_fc,mode_fc) == expected_fc


#myAdditions test
def test_myAdditions(in_range_file,myAddition_padding):
    assert mod.myAdditions(in_range_file) == myAddition_padding

#unCover test
def test_unCover_fileSizeIndex_end(myAddition_padding):
    assert mod.unCover(myAddition_padding) == myFileSizeStringLength

def test_unCover_fileName_end(myAddition_padding):
    i = myFileStringLength
    pos = mod.unCover(myAddition_padding)
    next = myAddition_padding[pos+myPaddingStringLength:]
    assert mod.unCover(next) == i

#confirm_loop test
@pytest.mark.parametrize(
        ('input_cl','expected_cl'),
        (
            ('0',False),
            ('1',True),
            ('Hello',-1),
            ('\n',-1),
            ('999999',-1),
            ('0123',-1),
            ('1234',-1),
        ),
)

def test_confirm_loop(input_cl,expected_cl):
    assert stripped_confirm_loop(input_cl) == expected_cl

#checkFileSize test

@pytest.mark.parametrize(
        ('input_cFS','expected_cFS','mode_cFS'),
        (
            ('emptyFile',-2,1),
            ('emptyFile',-2,2),
            ('exceedFile',0,1),
            ('in_range_file',1,1),
            ('nonexistentFile',-1,1),
            ('in_range_file',1,0),
            ('nonexistentFile',-1,0),
            ('exceedFile',1,2),
            ('nonexistentFile',-1,2),
        ),
)

def test_checkFileSize(input_cFS,expected_cFS,mode_cFS,request):
    input_cFS = request.getfixturevalue(input_cFS)
    assert mod.checkFileSize(input_cFS,mode_cFS) == expected_cFS


#xoring_key_file test
#TODO: test for mismatched sizes (file > key) (file < key) (file != key)
def test_xoring_key_file(myFileSize_byte_array,zero_byte_array):
    assert mod.xoring_key_file(myFileSize_byte_array,myFileSize_byte_array) == zero_byte_array

#extending_key test


@pytest.fixture
def zero_len_byte_array():
    return bytearray()

@pytest.fixture
def one_len_byte_array():
    my_bytes = mod.np.random.randint(255,size=1)
    result = bytearray()
    for byte in my_bytes:
        result.append(byte)
    return result

@pytest.fixture
def five_one_two_len_byte_array():
    my_bytes = mod.np.random.randint(255,size=512)
    result = bytearray()
    for byte in my_bytes:
        result.append(byte)
    return result

@pytest.mark.parametrize(
        ('input_eK','expected_eK_size'),
        (
            ('myKeySize_byte_array',0),
            ('zero_len_byte_array',myKeySize-0),
            ('one_len_byte_array',myKeySize-1),
            ('five_one_two_len_byte_array',myKeySize-512),
        ),
)

def test_extending_key(input_eK,expected_eK_size,request):
    input_eK = request.getfixturevalue(input_eK)
    assert len(mod.extending_key(input_eK,1024-len(input_eK))) == expected_eK_size

#determine_pad test
""""
    Test Cases:
        1: generates 100 to 199
        2:
            if next is >= 5: generates 25 to 29
            else: generates 200 to 249
        0: generates 00 to 09
        3 to 9: generates 30 to 99
        negatives: set to PI_size + negative
        illegal: Strings
        
"""

@pytest.mark.parametrize(
        ('input_dp','expected_dp'),
        (
            (0,'31'),
            (9999,'83'),
            (-1,'83'),
            (1,'141'),
            (32,'02'),
            (307,'00'),
            (292,'249'),
            (33,'28'),
        ),
)

def test_determine_pad(input_dp,expected_dp):
    assert mod.determine_pad(input_dp) == expected_dp


#hash_key test
@pytest.mark.parametrize(
        ('input_hK','min_in','max_ex','expected_B'),
        (
            ('myKeySize_byte_array',0,myStringPI_size,True),
            ('zero_len_byte_array',0,myStringPI_size,True),
            ('one_len_byte_array',0,myStringPI_size,True),
            ('five_one_two_len_byte_array',0,myStringPI_size,True),
        ),
)

def test_hask_key(input_hK,min_in,max_ex,expected_B,request):
    input_hK = request.getfixturevalue(input_hK)
    i = mod.hash_key(input_hK)
    assert (i >= min_in and i < max_ex) == expected_B

#extending_file test

@pytest.mark.parametrize(
        ('input_eF','expected_eF_size'),
        (
            ('sixteen_mega_byte_array',myMaxFileSize-16000000),
            ('one_len_byte_array',myMaxFileSize-1),
            ('five_one_two_len_byte_array',myMaxFileSize-512),
            ('myFileSize_byte_array',myMaxFileSize-myInRangeFileSize),
        ),
)

def test_extending_file(input_eF,expected_eF_size,request):
    input_eF = request.getfixturevalue(input_eF)
    assert len(mod.extending_file(input_eF)) == expected_eF_size

# Fixtures providing test data for HUFFMAN start here

@pytest.fixture
def simple_data():
    # Returns a bytearray with a mix of characters.
    # Example: b"aaabbc" -> frequencies: a=3, b=2, c=1
    return bytearray(b"aaabbc")

@pytest.fixture
def single_byte_data():
    # Returns a bytearray with a single repeated character.
    # This is an edge case for Huffman encoding (only one type of symbol).
    return bytearray(b"aaaaaa")

@pytest.fixture
def empty_data():
    # Returns an empty bytearray.
    return bytearray()


# 1. Testing build_freq_table

def test_build_freq_table_simple(simple_data):
    freq = mod.build_freq_table(simple_data)
    # Expecting: 'a' (97): 3, 'b' (98): 2, 'c' (99): 1.
    assert freq[97] == 3
    assert freq[98] == 2
    assert freq[99] == 1

def test_build_freq_table_empty(empty_data):
    freq = mod.build_freq_table(empty_data)
    # Expect an empty frequency table.
    assert freq == {}


# 2. Testing build_huffman_tree

def test_build_huffman_tree_simple(simple_data):
    freq = mod.build_freq_table(simple_data)
    tree = mod.build_huffman_tree(freq)
    # The total frequency at the root should equal the length of the data.
    assert tree.freq == len(simple_data)

def test_build_huffman_tree_single(single_byte_data):
    freq = mod.build_freq_table(single_byte_data)
    tree = mod.build_huffman_tree(freq)
    # For a single unique byte, the tree should be a leaf with no children.
    assert tree.byte is not None
    assert tree.left is None and tree.right is None
    assert tree.freq == len(single_byte_data)


# 3. Testing generate_codes

def test_generate_codes_simple(simple_data):
    freq = mod.build_freq_table(simple_data)
    tree = mod.build_huffman_tree(freq)
    codes = mod.generate_codes(tree)
    # Each unique byte in data should have an associated code.
    for byte in freq:
        assert byte in codes
        # For a multi-symbol data, code should not be empty.
        if len(freq) > 1:
            assert codes[byte] != ""
        assert isinstance(codes[byte], str)

def test_generate_codes_single(single_byte_data):
    freq = mod.build_freq_table(single_byte_data)
    tree = mod.build_huffman_tree(freq)
    codes = mod.generate_codes(tree)
    # Even if only one symbol exists, a code (possibly empty) should be assigned.
    for byte in freq:
        assert byte in codes


# 4. Testing huffman_encode

def test_huffman_encode(simple_data):
    freq = mod.build_freq_table(simple_data)
    tree = mod.build_huffman_tree(freq)
    codes = mod.generate_codes(tree)
    binary_str = mod.huffman_encode(simple_data, codes)
    # The binary string's length should equal the sum of the code lengths for each byte.
    expected_length = sum(len(codes[byte]) for byte in simple_data)
    assert len(binary_str) == expected_length
    # Only binary digits should be present.
    assert set(binary_str) <= {'0', '1'}


# 5. Testing binary_string_to_bytearray

def test_binary_string_to_bytearray_exact():
    # Use a binary string that is a multiple of 8: "01010101" represents 85.
    bin_str = "01010101"
    ba, padding = mod.binary_string_to_bytearray(bin_str)
    # Note: The implementation pads even if the input is a multiple of 8.
    # With 8 bits input, padded_length becomes 8 (appending eight '0's), resulting in 16 bits total.
    assert len(ba) == 2
    # The first byte equals 85; the second should be 0.
    assert ba[0] == 85
    assert ba[1] == 0
    assert padding == 8

def test_binary_string_to_bytearray_non_multiple():
    # Use a binary string whose length is not a multiple of 8, e.g. "101" (length 3).
    bin_str = "101"
    ba, padding = mod.binary_string_to_bytearray(bin_str)
    # Expect padded_length = 8 - 3 = 5, making the total length 8 bits.
    assert len(ba) == 1
    # "101" + "00000" -> "10100000" equals 160 in decimal.
    assert ba[0] == int("10100000", 2)
    assert padding == 5


# 6. Testing huffman_compress and huffman_decode round-trip

def test_huffman_compress_roundtrip(simple_data):
    # Compress the simple_data.
    compressed_data, tree, codes, padding = mod.huffman_compress(simple_data)
    # Validate types.
    assert isinstance(compressed_data, bytearray)
    assert isinstance(padding, int)
    # Recover the binary string from the compressed data.
    binary_str = mod.bytearray_to_binary_string(compressed_data, padding)
    # Decode the binary string back to the original data.
    decoded = mod.huffman_decode(binary_str, tree)
    assert decoded == simple_data

def test_huffman_compress_empty(empty_data):
    # For empty input,expect an IndexError due to an empty frequency table.
    with pytest.raises(IndexError):
        mod.huffman_compress(empty_data)


# 7. Testing behavior of huffman_decode with an incorrect tree

def test_huffman_decode_incorrect_tree(simple_data):
    # Compress the data properly.
    compressed_data, tree, codes, padding = mod.huffman_compress(simple_data)
    binary_str = mod.bytearray_to_binary_string(compressed_data, padding)
    # Create a different Huffman tree from a single-byte dataset.
    single_freq = mod.build_freq_table(bytearray(b"a"))
    wrong_tree = mod.build_huffman_tree(single_freq)
    # Expect an AttributeError when using the wrong tree.
    with pytest.raises(AttributeError):
        mod.huffman_decode(binary_str, wrong_tree)


