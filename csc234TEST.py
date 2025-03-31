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


TEN_THOUSAND_PI = "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989380952572010654858632788659361533818279682303019520353018529689957736225994138912497217752834791315155748572424541506959508295331168617278558890750983817546374649393192550604009277016711390098488240128583616035637076601047101819429555961989467678374494482553797747268471040475346462080466842590694912933136770289891521047521620569660240580381501935112533824300355876402474964732639141992726042699227967823547816360093417216412199245863150302861829745557067498385054945885869269956909272107975093029553211653449872027559602364806654991198818347977535663698074265425278625518184175746728909777727938000816470600161452491921732172147723501414419735685481613611573525521334757418494684385233239073941433345477624168625189835694855620992192221842725502542568876717904946016534668049886272327917860857843838279679766814541009538837863609506800642251252051173929848960841284886269456042419652850222106611863067442786220391949450471237137869609563643719172874677646575739624138908658326459958133904780275900994657640789512694683983525957098258226205224894077267194782684826014769909026401363944374553050682034962524517493996514314298091906592509372216964615157098583874105978859597729754989301617539284681382686838689427741559918559252459539594310499725246808459872736446958486538367362226260991246080512438843904512441365497627807977156914359977001296160894416948685558484063534220722258284886481584560285060168427394522674676788952521385225499546667278239864565961163548862305774564980355936345681743241125150760694794510965960940252288797108931456691368672287489405601015033086179286809208747609178249385890097149096759852613655497818931297848216829989487226588048575640142704775551323796414515237462343645428584447952658678210511413547357395231134271661021359695362314429524849371871101457654035902799344037420073105785390621983874478084784896833214457138687519435064302184531910484810053706146806749192781911979399520614196634287544406437451237181921799983910159195618146751426912397489409071864942319615679452080951465502252316038819301420937621378559566389377870830390697920773467221825625996615014215030680384477345492026054146659252014974428507325186660021324340881907104863317346496514539057962685610055081066587969981635747363840525714591028970641401109712062804390397595156771577004203378699360072305587631763594218731251471205329281918261861258673215791984148488291644706095752706957220917567116722910981690915280173506712748583222871835209353965725121083579151369882091444210067510334671103141267111369908658516398315019701651511685171437657618351556508849099898599823873455283316355076479185358932261854896321329330898570642046752590709154814165498594616371802709819943099244889575712828905923233260972997120844335732654893823911932597463667305836041428138830320382490375898524374417029132765618093773444030707469211201913020330380197621101100449293215160842444859637669838952286847831235526582131449576857262433441893039686426243410773226978028073189154411010446823252716201052652272111660396665573092547110557853763466820653109896526918620564769312570586356620185581007293606598764861179104533488503461136576867532494416680396265797877185560845529654126654085306143444318586769751456614068007002378776591344017127494704205622305389945613140711270004078547332699390814546646458807972708266830634328587856983052358089330657574067954571637752542021149557615814002501262285941302164715509792592309907965473761255176567513575178296664547791745011299614890304639947132962107340437518957359614589019389713111790429782856475032031986915140287080859904801094121472213179476477726224142548545403321571853061422881375850430633217518297986622371721591607716692547487389866549494501146540628433663937900397692656721463853067360965712091807638327166416274888800786925602902284721040317211860820419000422966171196377921337575114959501566049631862947265473642523081770367515906735023507283540567040386743513622224771589150495309844489333096340878076932599397805419341447377441842631298608099888687413260472156951623965864573021631598193195167353812974167729478672422924654366800980676928238280689964004824354037014163149658979409243237896907069779422362508221688957383798623001593776471651228935786015881617557829735233446042815126272037343146531977774160319906655418763979293344195215413418994854447345673831624993419131814809277771038638773431772075456545322077709212019051660962804909263601975988281613323166636528619326686336062735676303544776280350450777235547105859548702790814356240145171806246436267945612753181340783303362542327839449753824372058353114771199260638133467768796959703098339130771098704085913374641442822772634659470474587847787201927715280731767907707157213444730605700733492436931138350493163128404251219256517980694113528013147013047816437885185290928545201165839341965621349143415956258658655705526904965209858033850722426482939728584783163057777560688876446248246857926039535277348030480290058760758251047470916439613626760449256274204208320856611906254543372131535958450687724602901618766795240616342522577195429162991930645537799140373404328752628889639958794757291746426357455254079091451357111369410911939325191076020825202618798531887705842972591677813149699009019211697173727847684726860849003377024242916513005005168323364350389517029893922334517220138128069650117844087451960121228599371623130171144484640903890644954440061986907548516026327505298349187407866808818338510228334508504860825039302133219715518430635455007668282949304137765527939751754613953984683393638304746119966538581538420568533862186725233402830871123282789212507712629463229563989898935821167456270102183564622013496715188190973038119800497340723961036854066431939509790190699639552453005450580685501956730229219139339185680344903982059551002263535361920419947455385938102343955449597783779023742161727111723643435439478221818528624085140066604433258885698670543154706965747458550332323342107301545940516553790686627333799585115625784322988273723198987571415957811196358330059408730681216028764962867446047746491599505497374256269010490377819868359381465741268049256487985561453723478673303904688383436346553794986419270563872931748723320837601123029911367938627089438799362016295154133714248928307220126901475466847653576164773794675200490757155527819653621323926406160136358155907422020203187277605277219005561484255518792530343513984425322341576233610642506390497500865627109535919465897514131034822769306247435363256916078154781811528436679570611086153315044521274739245449454236828860613408414863776700961207151249140430272538607648236341433462351897576645216413767969031495019108575984423919862916421939949072362346468441173940326591840443780513338945257423995082965912285085558215725031071257012668302402929525220118726767562204154205161841634847565169998116141010029960783869092916030288400269104140792886215078424516709087000699282120660418371806535567252532567532861291042487761825829765157959847035622262934860034158722980534989650226291748788202734209222245339856264766914905562842503912757710284027998066365825488926488025456610172967026640765590429099456815065265305371829412703369313785178609040708667114965583434347693385781711386455873678123014587687126603489139095620099393610310291616152881384379099042317473363948045759314931405297634757481193567091101377517210080315590248530906692037671922033229094334676851422144773793937517034436619910403375111735471918550464490263655128162288244625759163330391072253837421821408835086573917715096828874782656995995744906617583441375223970968340800535598491754173818839994469748676265516582765848358845314277568790029095170283529716344562129640435231176006651012412006597558512761785838292041974844236080071930457618932349229279650198751872127267507981255470958904556357921221033346697499235630254947802490114195212382815309114079073860251522742995818072471625916685451333123948049470791191532673430282441860414263639548000448002670496248201792896476697583183271314251702969234889627668440323260927524960357996469256504936818360900323809293459588970695365349406034021665443755890045632882250545255640564482465151875471196218443965825337543885690941130315095261793780029741207665147939425902989695946995565761218656196733786236256125216320862869222103274889218654364802296780705765615144632046927906821207388377814233562823608963208068222468012248261177185896381409183903673672220888321513755600372798394004152970028783076670944474560134556417254370906979396122571429894671543578468788614445812314593571984922528471605049221242470141214780573455105008019086996033027634787081081754501193071412233908663938339529425786905076431006383519834389341596131854347546495569781038293097164651438407007073604112373599843452251610507027056235266012764848308407611830130527932054274628654036036745328651057065874882256981579367897669742205750596834408697350201410206723585020072452256326513410559240190274216248439140359989535394590944070469120914093870012645600162374288021092764579310657922955249887275846101264836999892256959688159205600101655256375678"

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
            case -2:
                print("File is empty")
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

    #Grab size
    fileS = str(os.stat(file).st_size)
    encoded = fileS.encode('utf-8')
    filesize = bytearray(encoded)

    file_ex=bytearray(file.encode('utf-8'))
    
    myBuff.extend(filesize)
    myBuff.extend(pad)
    myBuff.extend(file_ex)
    myBuff.extend(pad)

    return myBuff


#Grabs the first instance of the bytes "HARE" in a bytearray
#Error: -1 if not found
def unCover(myArray):
    pad = bytearray("HARE".encode('utf-8'))
    i = 0

    try:
        i = myArray.index(pad)
    except ValueError:
        i = -1
    
    return i


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
    elif filesize <= 0:
        return -2
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
        result[x] = int(determine_pad(PI_pos))
        PI_pos += 1

    return bytearray(result)

def determine_pad(pos):
    if pos < 0:
        pos = len(TEN_THOUSAND_PI) + pos
    elif pos == len(TEN_THOUSAND_PI):
        pos = 0
    elif pos > len(TEN_THOUSAND_PI):
        pos = pos % 10000
    
    pos1 = pos + 1
    pos2 = pos + 2

    if pos1 == len(TEN_THOUSAND_PI):
        pos1 = 0
        pos2 = 1
    elif pos2 == len(TEN_THOUSAND_PI):
        pos2 = 0

    digit = int(TEN_THOUSAND_PI[pos])
    match digit:
        case 1:
            return TEN_THOUSAND_PI[pos] + "" + TEN_THOUSAND_PI[pos1] + "" + TEN_THOUSAND_PI[pos2]
        case 2:
            if int(TEN_THOUSAND_PI[pos1]) >= 5:
                return TEN_THOUSAND_PI[pos] + "" + TEN_THOUSAND_PI[pos1]
            else: return TEN_THOUSAND_PI[pos] + "" + TEN_THOUSAND_PI[pos1] + "" + TEN_THOUSAND_PI[pos2]
        case _:
            return TEN_THOUSAND_PI[pos] + "" + TEN_THOUSAND_PI[


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
    hypercube_length, square_length, num_dimensions = 8, 512, 3

    # Calculate required sizes
    data_size = hypercube_length**num_dimensions * square_length*square_length // 8
    key_size = (hypercube_length**num_dimensions) * num_dimensions

    # Pad with pi
    original_byte_array = pad_with_pi(submit_vf, data_size)
    
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
    hypercube_length, square_length, num_dimensions = 8, 512, 3

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

    #decrypted_data = decrypted_data[:fileSize]

    output_path = input("Where to save??: ").strip()
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    
    print("View your file here:", output_path)
