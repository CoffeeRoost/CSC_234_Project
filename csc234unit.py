import pytest
import os
import csc234V3 as mod

myTestingFile = 'myTestingFile.txt'
myFileStringLength = len(myTestingFile)

myStringPI_size = 10000

myMaxFileSize = 16000000

myPadding = 'HARE'
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
            ('exceedFile','File exceeds 12mb maximum',1),
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
            ('exceedFile',0,1),
            ('in_range_file',1,1),
            ('nonexistentFile',-1,1),
            ('in_range_file',1,0),
            ('nonexistentFile',-1,0),
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
            (9999,'78'),
            (10000,'83'),
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