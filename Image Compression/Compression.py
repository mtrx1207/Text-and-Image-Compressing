
#######################################################################################################################################
                                                    #多媒体技术期末大程序I: JPEG图像压缩算法#
                                                        #RAYMOND SIDHARTA - 3200300849#
                                                      #(Functions and Classes Definition)#
#######################################################################################################################################
import numpy as np

def BGRtoYCbCr(img):
    '''
    Convert BGR color space to YCbCr color space
    Based on cv2 color convertion. Reference : https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    
    parameter : img , 3-dimension ndarray with shape (height, width, 3)
    return : result, 3-dimension ndarray with shape same as img
    '''
    result = np.zeros(img.shape, np.float32)
    result[:,:,0] = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    result[:,:,1] = (img[:,:,0] - result[:,:,0]) * 0.564 + 128
    result[:,:,2] = (img[:,:,2] - result[:,:,0]) * 0.713 + 128
    return np.round(result).astype(int)

def YCbCrtoBGR(img):
    '''
    Convert YCbCr color space to BGR color space
    Based on cv2 color convertion. Reference : https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    
    parameter : img , 3-dimension ndarray with shape (height, width, 3)
    return : result, 3-dimension ndarray with shape same as img
    '''
    result = np.zeros(img.shape, np.float32)
    result[:,:,0] = img[:,:,0] + 1.773 * (img[:,:,1] - 128)
    result[:,:,1] = img[:,:,0] - 0.714 * (img[:,:,2] - 128) - 0.344 * (img[:,:,1] - 128)
    result[:,:,2] = img[:,:,0] + 1.403 * (img[:,:,2] - 128)
    return np.round(result).astype(int)

def Padding(img):
    '''
    Pad image so its height and width are divisible by 16
    Preparation for chroma down-sampling and blocks separation
    
    parameter : img , 3-dimension ndarray with shape (height, width, 3)
    return : img, 3-dimension ndarray represents padded image
    '''
    left_padding = right_padding = top_padding = bottom_padding = 0
    img_height  = img.shape[0]
    img_width   = img.shape[1]
        
    #Vertical padding
    if img_height % 16 != 0:
        vPad = 16 - (img_height % 16)
        top_padding = vPad // 2
        bottom_padding = vPad - top_padding
        img = np.concatenate((np.repeat(img[:1], top_padding, 0), 
                                img, 
                                np.repeat(img[-1:], bottom_padding, 0)), axis = 0)

    #Horizontal padding
    if img_width % 16 != 0:
        hPad = 16 - (img_width % 16)
        left_padding = hPad // 2
        right_padding = hPad - left_padding
        img = np.concatenate((np.repeat(img[:,:1], left_padding, 1),
                                img,
                                np.repeat(img[:,-1:], right_padding, 1)), axis = 1)
    
    return img

class DCT:
    def C(self, ksi):
        return 1/(2**0.5) if ksi == 0 else 1
        
    def DCT2D(self, x):
        '''
        DCT function
        
        parameter : x, ndarray with shape (image_height_downsampled * image_width_downsampled / 64, 8 ,8) represents list of blocks
        return : res, ndarray with shape same as x
        '''
        res = np.zeros((x.shape))
        
        for b in range(res.shape[0]): #for all blocks in x
            block = np.zeros((8, 8))
            for u in range(8):
                for v in range(8):
                    const = 0.25 * self.C(u) * self.C(v)
                    val = 0.0
                    for i in range(8):
                        for j in range(8):
                            val += x[b, i, j] * np.cos((2*i + 1)*u*np.pi/16) * np.cos((2*j + 1)*v*np.pi/16) #DCT FORMULA
                    block[u, v] = const * val
            res[b] = block
        return np.round(res).astype(int)

    def IDCT2D(self, x):
        '''
        Inverse DCT function
        
        parameter : x, ndarray with shape (image_height_downsampled * image_width_downsampled / 64, 8 ,8) represents list of blocks
        return : res, ndarray with shape same as x
        '''
        res = np.zeros((x.shape))
        
        for b in range(res.shape[0]):
            block = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    val = 0.0
                    for u in range(8):
                        for v in range(8):
                            val += self.C(u) * self.C(v) * x[b, u, v] * np.cos((2*i+1)*u*np.pi/16) * np.cos((2*j+1)*v*np.pi/16)
                    block[i,j] = val / 4
            res[b] = block
        return np.round(res).astype(int)

class DownSampling:
    '''
    Class for color downsampling
    For constructor, pass only either '4:4:4' or '4:2:0' for the parameter
    4:4:4 or no sampling, apply for Luminance component
    4:2:0 apply for Cb and Cr components
    '''
    def __init__(self, ratio = '4:2:0'):
        assert ratio in ('4:4:4', '4:2:0'), "Please choose one of the following {'4:4:4', '4:2:0'}"
        self.ratio = ratio
    
    def Downsample(self, x):
        '''
        Function for chroma down-sampling
        
        parameter : x , 2-dimension ndarray with shape (image_height, image_width) represent a certain image component
        return : res , 2-dimension ndarray with shape half from x (for 4:2:0) or exactly same with x (for 4:4:4)
        '''
        # No down-sampling
        if self.ratio == '4:4:4':
            return x, x.shape[0], x.shape[1]
        
        # 4 : 2 : 0 down-sampling
        else:
            res = np.zeros((x.shape[0]//2, x.shape[1]//2))
            a = 0
            for i in range(0, x.shape[0], 2):
                b = 0
                for j in range(0, x.shape[1], 2):
                    res[a, b] = x[i,j]
                    b += 1
                a += 1
            return np.round(res).astype(int), res.shape[0], res.shape[1]
    
    def Rescale(self, x):
        '''
        Function for rescale down-sampled chroma components
        
        parameter : x , 2-dimension ndarray with shape (comp_height_downsampled, comp_width_downsampled) represent a certain downsampled image component
        return : res , 2-dimension ndarray with shape two times from x (for 4:2:0)
        '''
        if (self.ratio == '4:2:0'):
            res = np.zeros((x.shape[0] * 2, x.shape[1] * 2))
            u = v = 0
            for i in range (x.shape[0]):
                for j in range(x.shape[1]):
                    res[u:u+2, v:v+2] = np.full([2, 2], x[i,j])
                    v += 2
                u += 2
                v = 0
            
        return res.astype(int)
                    
class ImageBlock:
    '''
    Class for dealing with image blocks
    Slice a 2-dimension image component to a list of 8 x 8 blocks
    and merge those blocks into a 2-dimension image component again
    For constructor, pass only either '4:4:4' or '4:2:0' for the parameter
    4:4:4 or no sampling, apply for Luminance component
    4:2:0 apply for Cb and Cr components
    '''
    def __init__(self):
        self.left_padding = self.right_padding = self.top_padding = self.bottom_padding = 0
    
    def splitToBlock(self, img):
        '''
        Function for split image component to a list of 8 x 8 blocks
        
        parameter : img , 2-dimension ndarray with shape (comp_height_downsampled, comp_width_downsampled) represent a certain downsampled image component
        return : (blocks, indices) , a tuple which blocks is a bunch of 8 x 8 blocks and indices is blocks's index in img (will be used for merge block)
        '''
        self.img_height = img.shape[0]
        self.img_width = img.shape[1]
        
        blocks = []
        indices = []
        for i in range (0, self.img_height, 8):
            for j in range (0, self.img_width, 8):
                blocks.append(img[i:i+8, j:j+8])
                indices.append((i,j))
        blocks = np.array(blocks)
        indices = np.array(indices)
        return blocks, indices
    
    def mergeBlock(self, blocks, indices, img_height, img_width):
        '''
        Function for merge the blocks into one single 2-dimension  component
        
        parameter : 
        - blocks, ndarray consists of 8 x 8 blocks
        - indices, ndarray represents index of blocks's block in original image
        - img_height, int represents component's height
        - img_width, int represents component's width
        return : img , an ndarray which is the image component
        '''
        img = np.zeros((img_height, img_width)).astype(int)
        for block, index in zip(blocks, indices):
            i, j = index
            img[i:i+8, j:j+8] = block
        
        return img

class Quantization:
    '''
    Class for quantization process
    '''
    luminanceTable = np.array([[16,11,10,16,24,40,51,61],
                               [12,12,14,19,26,58,60,55],
                               [14,13,16,24,40,57,69,56],
                               [14,17,22,29,51,87,80,62],
                               [18,22,37,56,68,109,103,77],
                               [24,35,55,64,81,104,113,92],
                               [49,64,78,87,103,121,120,101],
                               [72,92,95,98,112,100,103,99]])
    
    chrominanceTable = np.array([[17,18,24,47,99,99,99,99],
                                 [18,21,26,66,99,99,99,99],
                                 [24,26,56,99,99,99,99,99],
                                 [47,66,99,99,99,99,99,99],
                                 [99,99,99,99,99,99,99,99],
                                 [99,99,99,99,99,99,99,99],
                                 [99,99,99,99,99,99,99,99],
                                 [99,99,99,99,99,99,99,99]])
    
    def quantize(self, x, channel_type):
        '''
        Function to divide certain block's elements into corresponding element in quantization table
        
        parameter :
        - x , 8 x 8 ndarray which is our image block
        - channel_type, string either 'lum' for using luminanceTable or 'chr' for using chrominanceTable
        result : res , 8 x 8 ndarray which is image block divided by its quantization table
        '''
        assert channel_type in ('lum', 'chr')
        
        if channel_type == 'lum':
            Q = self.luminanceTable
        else:
            Q = self.chrominanceTable
        
        res = np.round(x/Q).astype(int)
        return res
    
    def dequantize(self, x, channel_type):
        '''
        Function to multiply certain quantized block with corresponding element in quantization table
        
        parameter :
        - x , 8 x 8 ndarray which is our quantized block
        - channel_type, string either 'lum' for using luminanceTable or 'chr' for using chrominanceTable
        result : res , 8 x 8 ndarray which is image block divided by its quantization table
        '''
        assert channel_type in ('lum', 'chr')
    
        if channel_type == 'lum':
            Q = self.luminanceTable
        else:
            Q = self.chrominanceTable
        res = x * Q
        return res
  
class EntropyCoding:
    '''
    Class for implementing lossless compression (entropy coding), included :
    - Zigzag conversion from 8 x 8 block into 64-length 1-dimension list
    - Inverse Zigzag conversion from list back to 8 x 8 block
    - Huffman Coding for DPCM-coded DC coefficients
    - Huffman Coding for RLE-coded AC coefficients
    '''
    class HuffmanTreeNode:
        '''
        Class to create Huffman tree's node, which consists the information of left and right subtrees
        Left and right subtrees can be a int (for DPCMHuffmanTree leaf node), tuple (for RLEHuffmanTree), or another Huffman tree's node
        '''
        def __init__(self, left = None, right = None):
            self.left = left
            self.right = right
            
        def children(self): #return a tuple consists left and right subtrees
            return (self.left, self.right)
    
    def DPCMHuffmanTree(self, node, left=True, binString=''):
        '''
        Function to create DPCM Huffman dictionary from the HuffmanTree by traversing it in preorder style
        If we go left, assign '0' to the binString
        If we go right, assign '1' to the binString
        Do this recursively until leaf node was founded, and then write down in our dictionary
        
        parameter :
        - node , DPCM Huffman tree's root node
        - left , bool to mark if current node is left or right child of its prior parent node
        - binString , string consist of '1' or '0'. Longer string length for deeper node
        result : d , Huffman dictionary in form of {data : Huffman bit}, where data = integer; Huffman bit = string
        '''
        if type(node) is int:
            return {node: binString}
        (l, r) = node.children()
        d = dict()
        d.update(self.DPCMHuffmanTree(l, True, binString + '0'))
        d.update(self.DPCMHuffmanTree(r, False, binString + '1'))
        return d
    
    def RLEHuffmanTree(self, node, left=True, binString=''):
        '''
        Function to create RLE Huffman dictionary from the HuffmanTree by traversing it in preorder style
        If we go left, assign '0' to the binString
        If we go right, assign '1' to the binString
        Do this recursively until leaf node was founded, and then write down in our dictionary
        
        parameter :
        - node , RLE Huffman tree's root node
        - left , bool to mark if current node is left or right child of its prior parent node
        - binString , string consist of '1' or '0'. Longer string length for deeper node
        result : d , Huffman dictionary in form of {data : Huffman bit}, where data = tuple (runlength, size); Huffman bit = string
        '''
        if type(node) is tuple:
            return {node: binString}
        (l, r) = node.children()
        d = dict()
        d.update(self.RLEHuffmanTree(l, True, binString + '0'))
        d.update(self.RLEHuffmanTree(r, False, binString + '1'))
        return d
    
    def DPCM(self, imageBlocks):
        '''
        Function to implement DPCM on DC coefficients and create the DPCM Huffman Table
        DC coefficient represents the average intensity of a block
        There's only 1 DC coefficient in each block, the rest are AC coefficients
        DC coefficient is coded separately from AC coefficients
        Suppose we have 5 DC coeffs from 5 different blocks : DC_1, DC_2, DC_3, DC_4, DC_5
        The idea of DPCM is to create a list which contains dc_i = DC_i+1 - DC_i, and dc_0 = DC_0
        We shall call dc_i an amplitude and represented by binary string
        Positive amplitude represented by normal bit, negative amplitude represented using one's complement bit, zero represented using empty bit
        Each DPCM-coded DC coefficient is represented by a pair of symbols (SIZE, AMPLITUDE) which will be saved in dc_tuple, while SIZE is aplitude bit's length
        Huffman code shall be implemented ONLY for SIZE, that is, we will assign shorter bit for SIZE if it appeared frequently
        We save AMPLITUDE as it is, no Huffman code
        
        parameter : imageBlocks, 8 x 8 shaped ndarray
        result : DPCMhuffmanCodeTable , Huffman dictionary in form of {SIZE : Huffman bit}, where SIZE = integer ; Huffman bit = string
        '''
        
        # DPCM for DC coefficients
        dc_list = []
        for i in range (imageBlocks.shape[0]):
            dc_list.append(imageBlocks[i,0,0])
            
        dc_coef = np.array(dc_list)
        predictor = 0
        
        self.dc_tuple = []
        for i in range(len(dc_coef)):
            diff = dc_coef[i] - predictor
            if diff == 0:
                self.dc_tuple.append((0, ''))
            elif diff < 0:
                diff = bin(abs(dc_coef[i] - predictor))[2:]
                diff = ''.join(['0' if i == '1' else '1' for i in diff])
                self.dc_tuple.append((len(diff), diff))
            else:
                diff = bin(dc_coef[i] - predictor)[2:]
                self.dc_tuple.append((len(diff), diff))
            predictor = dc_coef[i]
        
        # Create Huffman table or dictionary
        size_freqdict = dict()
        for i in self.dc_tuple:
            #count appearance frequency of each SIZE
            if i[0] not in size_freqdict:
                size_freqdict[i[0]] = 1
            else:
                size_freqdict[i[0]] += 1
        
        fd = [(freq, k) for (k, freq) in size_freqdict.items()]
        fd = sorted(fd) #sort based on SIZE's frequency
        nodes = fd.copy()
        while len(nodes) > 1:
            (freq1, k1) = nodes[0]
            (freq2, k2) = nodes[1]
            nodes = nodes[2:]
            node = self.HuffmanTreeNode(k1, k2)
            nodes.append((freq1 + freq2, node))
            nodes = sorted(nodes, key = lambda x : x[0])
        
        DPCMhuffmanCodeTable = self.DPCMHuffmanTree(nodes[0][1])
        return DPCMhuffmanCodeTable
        
    def Zigzag(self, block):
        '''
        Function to implement Zigzag scan algorithm on 8 x 8 image block
        After quantization, elements at right bottom of image block usually have value of 0, because we've thrown away high frequency information
        By applying Zigzag algorithm, those zeros will appear in the end of our resulting list, and by applying RLE we can save more bits 
        
        parameter : block , 8 x 8 ndarray
        return : res , list
        '''
        i = 0
        j = 1
        res = []
        isUp = False
        isDown = True
        while j <= 7 - i:
            res.append(block[i][j])
            if i == 0 and j == 0:
                j += 1
                isDown = True
            elif i == 7 and j == 0:
                j += 1
                isUp = True
            else:
                if isUp:
                    i -= 1
                    j += 1
                elif isDown:
                    i += 1
                    j -= 1
                if i < 0:
                    i = 0
                    isUp = False
                    isDown = True
                elif j < 0:
                    j = 0
                    isUp = True
                    isDown = False  

        while i != 7 or j != 7:
            res.append(block[i][j])
            if isUp:
                    i -= 1
                    j += 1
            elif isDown:
                i += 1
                j -= 1
            
            if i > 7:
                i = 7
                j += 2
                isUp = True
                isDown = False
                
            elif j > 7:
                j = 7
                i += 2
                isUp = False
                isDown = True
        res.append(block[7][7])
        return res
    
    def iZigzag(self, ac_coef):
        '''
        Function to implement inverse Zigzag scan algorithm to fill 8 x 8 block from given list of AC coefficients
        
        parameter : ac_coef , list
        return : newmat , 8 x 8 ndarray
        '''
        newmat = np.zeros((8,8))
        i = 0; j = 1
        isUp = False
        isDown = True

        while j <= 7 - i:
            newmat[i,j] = ac_coef.pop(0)
            if i == 0 and j == 0:
                j += 1
                isDown = True
            elif i == 7 and j == 0:
                j += 1
                isUp = True
            else:
                if isUp:
                    i -= 1
                    j += 1
                elif isDown:
                    i += 1
                    j -= 1
                if i < 0:
                    i = 0
                    isUp = False
                    isDown = True
                elif j < 0:
                    j = 0
                    isUp = True
                    isDown = False  

        while i != 7 or j != 7:
            newmat[i,j] = ac_coef.pop(0)
            if isUp:
                    i -= 1
                    j += 1
            elif isDown:
                i += 1
                j -= 1
            
            if i > 7:
                i = 7
                j += 2
                isUp = True
                isDown = False
                
            elif j > 7:
                j = 7
                i += 2
                isUp = False
                isDown = True
        newmat[7,7] = ac_coef.pop(0)
        return newmat.astype(int)
        
    def RLE(self, imageBlock):
        '''
        Function to implement Run-Length Encoding on AC coefficients and create RLE Huffman Table
        RLE is implemented since most part of our AC coefficients are zero after quantization
        Together with Zigzag scan algorithm, we can group those zeros at the end of the AC coefficients list
        RLE step replaces values by a pair (RUNLENGTH, VALUE)
        RUNLENGTH is the number of zeros we found when we scan the AC coefficients list from left to right until we found a non-zero number, called VALUE
        To further save bits, a special pair (0,0) indicates the end-of-block after the last nonzero AC coefficient is reached
        
        To implement Huffman coding on RLE-coded AC coefficients, we separate the pair further with 2 symbols
        - Symbol 1 : (RUNLENGTH, SIZE)
            RUNLENGTH is number of zero (can be represented from 0 - 15)
            SIZE is AMPLITUDE bit's length
            Since RUNLENGTH can represent only zero-runs of length 0 - 15, we must give special extension code (15 , 0)
        - Symbol 2 : AMPLITUDE
            AMPLITUDE is VALUE, but in binary string
            Positive AMPLITUDE use normal bit
            Negative AMPLITUDE use one's complement bit
            Zero AMPLITUDE use empty bit
        Only Symbol 1 will be Huffman coded
        
        parameter : imageBlock , 8 x 8 ndarray
        return : RLEhuffmanCodeTable , Huffman dictionary in form of {Symbol 1 : Huffman bit}, where Symbol 1 = tuple ; Huffman bit = string
        '''
        ac_coef = self.Zigzag(imageBlock)
        self.ac_tuple = []
        
        # Run-Length Encoding
        runlength = 0
        for i in range(len(ac_coef)):
            if ac_coef[i] == 0: # zero run
                if runlength == 15:
                    self.ac_tuple.append(((15, 0),''))
                    runlength = 0
                else:
                    runlength += 1
            else: # on-zero value founded
                amplitude = ac_coef[i]
                if amplitude < 0:
                    amplitude = bin(abs(amplitude))[2:]
                    amplitude = ''.join(['0' if i == '1' else '1' for i in amplitude])
                    self.ac_tuple.append(((runlength, len(amplitude)), amplitude))
                else:
                    amplitude = bin(amplitude)[2:]
                    self.ac_tuple.append(((runlength, len(amplitude)), amplitude))  
                runlength = 0
        
        self.ac_tuple.append(((0,0), '')) # append special pair to mark the end of our block
        
        # Create Huffman table or dictionary
        symbolone_freqdict = dict()
        for i in self.ac_tuple:
            # count appearance frequency of each Symbol 1
            if i[0] not in symbolone_freqdict:
                symbolone_freqdict[i[0]] = 1
            else:
                symbolone_freqdict[i[0]] += 1
                
        fd = [(freq, k) for (k, freq) in symbolone_freqdict.items()]
        fd = sorted(fd) # sort based on Symbol1's frequency
        nodes = fd.copy()
        while len(nodes) > 1:
            (freq1, k1) = nodes[0]
            (freq2, k2) = nodes[1]
            nodes = nodes[2:]
            node = self.HuffmanTreeNode(k1, k2)
            nodes.append((freq1 + freq2, node))
            nodes = sorted(nodes, key = lambda x : x[0])
        
        RLEhuffmanCodeTable = self.RLEHuffmanTree(nodes[0][1])
        return RLEhuffmanCodeTable
        
    def WriteBitStream(self, image_height, image_width, imageBlocks, mode):
        '''
        Function to encode our compressed image data
        Compressed image data is saved in ./result.txt
        Bit allocation :
        - 16 bits for image component's height
        - 16 bits for image component's width
        - 16 bits for number of blocks of that image component (b)
        
        - 8 bits for DPCM table size (s)
        * 8 bits for original SIZE's binary string
        * 4 bits for SIZE's Huffman bit length (n)
        * n bits for SIZE's Huffman bit
        - repeat '*' s times
        - our DPCM data in form of SIZE's Huffman bit followed by its amplitude
        
        * 4 bits for RLE table size (s)
        ** 4 bits for RUNLENGTH
        ** 4 bits for SIZE
        ** 4 bits for current Symbol 1 Huffman bit length (n)
        ** n bits for Symbol 1 Huffman bit length
        * repeat '**' s times
        * our RLE data in form of Symbol 1 Huffman bit followed by the value/amplitude
        - repeat '*' b times
        
        parameter :
        - image_height, int represents certain image component's height
        - image_width, int represents certain image component's width
        - imageBlocks, 3-dimension ndarray with shape (image_height*image_width/64, 8, 8) represents list of 8 x 8 blocks
        return : void
        '''
        block_num = imageBlocks.shape[0]
        with open('./Image Compression/helloworld_compressed.txt', mode) as fw:
            # Header bits
            DPCMhuffmanCodeTable = self.DPCM(imageBlocks)
            image_height = bin(image_height)[2:].zfill(16); fw.write(image_height)
            image_width = bin(image_width)[2:].zfill(16); fw.write(image_width)
            block_num = bin(block_num)[2:].zfill(16); fw.write(block_num)
            
            # DPCM bits
            DPCMTableSize = bin(len(DPCMhuffmanCodeTable))[2:].zfill(8); fw.write(DPCMTableSize)
            for (size, huffbit) in DPCMhuffmanCodeTable.items():
                oribit = bin(size)[2:].zfill(8)
                bitlen = bin(len(huffbit))[2:].zfill(4)
                frame = oribit + bitlen + huffbit
                fw.write(frame)
                
            for (size, amplitude) in self.dc_tuple:
                fw.write(DPCMhuffmanCodeTable[size] + amplitude)
                
            # RLE bits
            for imageBlock in imageBlocks:
                RLEhuffmanCodeTable = self.RLE(imageBlock)
                RLETableSize = bin(len(RLEhuffmanCodeTable))[2:].zfill(4); fw.write(RLETableSize)
                
                for (symbolone, huffbit) in RLEhuffmanCodeTable.items():
                    runlength = bin(symbolone[0])[2:].zfill(4)
                    size = bin(symbolone[1])[2:].zfill(4)
                    bitlen = bin(len(huffbit))[2:].zfill(4)
                    frame = runlength + size + bitlen + huffbit
                    fw.write(frame)
                
                for (symbolone, amplitude) in self.ac_tuple:
                    fw.write(RLEhuffmanCodeTable[symbolone] + amplitude)                
    
    def ReadBitStream(self):
        '''
        Function to decode our compressed image's data
        
        parameter : void
        return : (YCbCrblocks[0], YCbCrblocks[1], YCbCrblocks[2]) , tuple :
        - YCbCrblocks[0] is a 3-dimensional ndarray represents list of 8 x 8 Luminance blocks
        - YCbCrblocks[1] is a 3-dimensional ndarray represents list of 8 x 8 Cr blocks
        - YCbCrblocks[2] is a 3-dimensional ndarray represents list of 8 x 8 Cb blocks
        '''
        with open('./Image Compression/helloworld_compressed.txt', 'r', encoding = 'utf-8') as fr:
            YCbCrblocks = []
            for _ in range(3): # we have 3 components : Y, Cb, Cr
                
                # Read each component's header bits
                image_height = fr.read(16); image_height = int(image_height, 2)
                image_width = fr.read(16); image_width = int(image_width, 2)
                block_num = fr.read(16); block_num = int(block_num, 2)
                
                # Read DPCM data
                DPCMTableSize = fr.read(8); DPCMTableSize = int(DPCMTableSize, 2)
                DPCMhuffmanDict = dict()
                for _ in range(DPCMTableSize):
                    freq = fr.read(8); freq = int(freq, 2)
                    bitlen = fr.read(4); bitlen = int(bitlen, 2)
                    bit = fr.read(bitlen)
                    DPCMhuffmanDict[bit] = freq
                
                DPCM = []   
                for _ in range(block_num):
                    bits = ''
                    while True:
                        bits += fr.read(1)
                        if bits in DPCMhuffmanDict:
                            size = DPCMhuffmanDict[bits]
                            if size == 0:
                                DPCM.append(0)
                            else:
                                amplitude = fr.read(size)
                                if amplitude[0] == '0':
                                    amplitude = ''.join(['0' if i == '1' else '1' for i in amplitude])
                                    DPCM.append(-1 * int(amplitude, 2))
                                else:
                                    DPCM.append(int(amplitude, 2))
                            break
                dc_coef = []
                diff = 0
                for i in range(len(DPCM)):
                    diff += DPCM[i]
                    dc_coef.append(diff)
                    
                # Read RLE data
                blocks = []
                for b in range(block_num):
                    RLEhuffmanDict = dict()
                    RLETableSize = fr.read(4); RLETableSize = int(RLETableSize, 2) 
                    for _ in range(RLETableSize):
                        runlength = fr.read(4); runlength = int(runlength, 2)
                        size = fr.read(4); size = int(size, 2)
                        bitlen = fr.read(4); bitlen = int(bitlen, 2)
                        bit = fr.read(bitlen)
                        RLEhuffmanDict[bit] = (runlength, size)
                    
                    RLE = []
                    isAnyTuple = True
                    
                    while isAnyTuple: 
                        bits = ''
                        while True:
                            bits += fr.read(1)
                            if bits in RLEhuffmanDict:
                                symbolone = RLEhuffmanDict[bits]
                                if symbolone[1] == 0:
                                    RLE.append((symbolone, 0))
                                    if symbolone[0] == 0:
                                        isAnyTuple = False
                                        break
                                else:
                                    amplitude = fr.read(symbolone[1])
                                    if amplitude[0] == '0':
                                        int_amplitude = ''.join(['0' if i == '1' else '1' for i in amplitude]) 
                                        RLE.append((symbolone, int(int_amplitude, 2) * -1))
                                    else:
                                        RLE.append((symbolone, int(amplitude, 2)))
                                break
                    
                    ac_coef = []
                    for i in RLE:
                        if i[0] == (0,0):
                            ac_coef.extend([0] * (63 - len(ac_coef)))
                            break
                        ac_coef.extend([0] * i[0][0])
                        ac_coef.append(i[1])
                    
                    block = self.iZigzag(ac_coef)
                    block[0,0] = dc_coef[b]
                    
                    blocks.append(block)
                YCbCrblocks.append(blocks)
            return YCbCrblocks[0], YCbCrblocks[1], YCbCrblocks[2]