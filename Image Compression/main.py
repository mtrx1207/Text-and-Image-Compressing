
#######################################################################################################################################
                                                    #多媒体技术期末大程序I: JPEG图像压缩算法#
                                                        #RAYMOND SIDHARTA - 3200300849#
#######################################################################################################################################
'''
Joint Photographic Experts Group or JPEG is an international organization that standardized the format during the late 1980s.
JPEG is a commonly used method of lossy and lossless compression for digital images.
Commonly seen JPEG filename extensions : .jpg, .jpeg, .jpe, .jif, .jfif, .jfi

This program shall apply JPEG image compression algorithms, either lossy and lossless, on BMP images.

Main steps : 
        Read BMP image -> Encoder -> Compressed image's data -> Decoder -> Retrieve the BMP image as similar as possible
    
Detail compression steps :
    1. Read BMP image in BGR format (Python cv2 use BGR, not RGB)
    2. Convert color space from BGR to YCbCr, subtract the result by 128 for recentering (DCT preparation)
    3. Pad the image until its height and weight divisible by 16 (preparation for chroma subsampling and block separation)
    4. !LOSSY STEP! : Chroma down-sampling by ratio 4:2:0 for reduce color information (human eyes are not really good to recognize color)
    5. Separate the image into 8 x 8 blocks
    6. Apply Discrete Cosine Transform (DCT) for each block (ignore high frequencies color difference in image since our eyes also can't recognize it as well)
    7. !LOSSY STEP! : Divide each block with Quantization Table, more data loss
    8. Convert each block into 1 dimensional list with Zigzag method, first value of that list is DC coefficient, others are AC coefficients (preparation for DPCM and RLC)
    9. Apply Differential Pulse Code Modulation (DPCM) on DC coefficients
    10. !LOSSLESS STEP! : Huffman Coding on DPCM-coded DC coefficients
    11. Apply Run-Length Coding (RLC) on AC coefficients
    12. !LOSSLESS STEP! : Huffman Coding on RLC-coded AC coefficients
    13. Write the corresponding bits as compressed image's data

Detail decompression steps :
    1. Read compressed image's data
    2. Decode AC coefficients using the Huffman table
    3. Decode DC coefficients using the Huffman table
    4. Merge DC with AC in a list, using inverse Zigzag method to recreate 8 x 8 quantized block
    5. Multiply each block with Quantization Table, resulting the block after DCT
    6. Apply Inverse DCT to that block, resulting the subsampled Y, Cb, and Cr blocks
    7. Merge the blocks; rescale Cb and Cr; combine Y, Cb, and Cr again
    8. Add 128 back to the corresponding result, and convert the color space from BGR to YCbCr
    9. Save the result, compare it with the original image   
'''

import cv2
import numpy as np
from multiprocessing.pool import Pool
import Compression

np.set_printoptions(edgeitems=8)

#---------------------#
# A. DATA PREPARATION
#---------------------#

# 1. Read BMP image
bgr_img = cv2.imread('./Image Compression/helloworld.bmp', 1)

# 2. BGR to YCbCr, then recenter
ycc_img = Compression.BGRtoYCbCr(bgr_img)
ycc_img = ycc_img.astype(int) - 128

# 3. Image padding
ycc_img = Compression.Padding(ycc_img)

#----------------------#
# B. IMAGE COMPRESSION
#----------------------#

# 4. LOSSY COMPRESSION : Downsampling 4 ：2 : 0 for Cb and Cr; 4:4:4 for Y 
lum_downsample = Compression.DownSampling('4:4:4')
chr_downsample = Compression.DownSampling('4:2:0')
Y,  Y_height,  Y_width = lum_downsample.Downsample(ycc_img[:,:,0])
Cb, Cb_height, Cb_width = chr_downsample.Downsample(ycc_img[:,:,1])
Cr, Cr_height, Cr_width = chr_downsample.Downsample(ycc_img[:,:,2])

# 5. Split Y, Cb, and Cr components into 8 x 8 blocks
imgBlock = Compression.ImageBlock()
Y_blocks, Y_indices  = imgBlock.splitToBlock(Y)
Cb_blocks, Cb_indices = imgBlock.splitToBlock(Cb)
Cr_blocks, Cr_indices = imgBlock.splitToBlock(Cr)

'''
DCT and Quantization runtime are the slowest among other image compression steps (also for IDCT and inverse quantization)
In order to reduce runtime, we shall use Python multiprocessing by implementing starmap function
That's why we use a separate function for DCT and Quantization step
'''
def quantizeBlock(block, type):
    dct = Compression.DCT()
    quant = Compression.Quantization()
    
    # 6. Discrete Cosine Transform
    transformed = dct.DCT2D(block)
    
    # 7. LOSSY COMPRESSION : Quantization
    quantized = quant.quantize(transformed, type)
    
    return quantized

def dequantizeBlock(block, type):
    dct = Compression.DCT()
    quant = Compression.Quantization()
    
    #Dequantization
    dequantized = quant.dequantize(block, type)
    
    #Reverse DCT
    detransformed = dct.IDCT2D(dequantized)
    
    return detransformed


# ENCODE and DECODE will be done using multiprocessing
if __name__ == '__main__':
    with Pool() as p:
        lst1 = [Y_blocks, Cb_blocks, Cr_blocks]
        lst2 = ['lum', 'chr', 'chr']
        
        # DCT and Quantization
        result = list(p.starmap(quantizeBlock, zip(lst1, lst2)))
        Y_quantized  = result[0]
        Cb_quantized = result[1]
        Cr_quantized = result[2]

        #------------#
        # C. ENCODE
        #------------#
        Lossless = Compression.EntropyCoding()
        Lossless.WriteBitStream(Y_height, Y_width, Y_quantized, 'w')
        Lossless.WriteBitStream(Cb_height, Cb_width, Cb_quantized, 'a')
        Lossless.WriteBitStream(Cr_height, Cr_width, Cr_quantized, 'a')

        #-----------------------------#
        # D. DECODE AND SEE THE RESULT
        #-----------------------------#
        
        # 1. Read compressed image's data, decode AC and DC coefficients, use inverse Zigzag method to recreate 8 x 8 quantized block
        Y_decoded, Cb_decoded, Cr_decoded = Lossless.ReadBitStream()

        # 2. Inverse Quantization and Inverse DCT
        lst1 = [Y_decoded, Cb_decoded, Cr_decoded]
        result = list(p.starmap(dequantizeBlock, zip(lst1, lst2)))
        Y_decompressed  = result[0]
        Cb_decompressed = result[1]
        Cr_decompressed = result[2]

        # 3. Merge blocks
        Y_merged = imgBlock.mergeBlock(Y_decompressed, Y_indices, Y_height, Y_width)
        Cb_merged = imgBlock.mergeBlock(Cb_decompressed, Cb_indices, Cb_height, Cb_width)
        Cr_merged = imgBlock.mergeBlock(Cr_decompressed, Cr_indices, Cr_height, Cr_width)

        # 4. Rescale Cb and Cr components
        Cb_rescaled = chr_downsample.Rescale(Cb_merged)
        Cr_rescaled = chr_downsample.Rescale(Cr_merged)

        # 5. Add 128 back to each component
        YCbCr_result = np.zeros((Y_height, Y_width, 3))
        YCbCr_result[:,:,0] = Y_merged + 128
        YCbCr_result[:,:,1] = Cb_rescaled + 128
        YCbCr_result[:,:,2] = Cr_rescaled + 128

        # 6. Convert YCbCr into BGR
        BGR_result = Compression.YCbCrtoBGR(YCbCr_result)
        
        # 7. See the resulting image after decompression
        cv2.imwrite('./Image Compression/helloworld_decompressed.bmp', BGR_result)