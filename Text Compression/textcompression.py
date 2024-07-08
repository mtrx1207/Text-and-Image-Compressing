
#######################################################################################################################################
                                                    #多媒体技术期末大程序II: 哈夫曼编码进行文本压缩#
                                                           #RAYMOND SIDHARTA - 3200300849#
#######################################################################################################################################

'''
This code is to implement Huffman Coding for image compression.
Huffman code is a way to encode information using variable-length strings to represent symbols depending on how frequently they appear.
If we don't use Huffman code, all characters is assigned with 1 byte (8 bits).
If we use Huffman code, characters that appear more frequently will be assigned by shorter bit.
This (maybe) create a lossless compression algorithm for the text, since we reduce the length of bit representation for each characters.
'''

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

#Create a class for Huffman Tree's Node
class TreeNode:
    def __init__(self, left = None, right = None):
        self.left = left #left child
        self.right = right #right child
    
    def children(self): #return a tuple consist of left and right child
        return (self.left, self.right)

def huffman_code_tree(node, left=True, binString=''):
    '''
    Create a Huffman Code Dictionary from Huffman Tree
    node : tree or subtree root
    left : is it left node?
    binString : binary string we've got after traversing the tree, deeper node gives longer binary string.
    '''
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict() #Huffman dictionary that we want to create
    d.update(huffman_code_tree(l, True, binString + '0')) #recursive to left node
    d.update(huffman_code_tree(r, False, binString + '1')) #recursive to right node
    return d

#---------------------#
# 1. DATA PREPARATION
#---------------------#

freqdict = dict()
data_path = "./Text Compression/testcase2.txt" # Change here for test another txt file
encoded_path = "./Text Compression/testcase_compressed2.txt"
result_path = "./Text Compression/testcase_result2.txt"

try:
    f = open(data_path, "r", encoding="utf-8")
except OSError:
    print ("Could not open file.")
    sys.exit()

while True:
    res = f.readline()
    if not res:
        break
    
    #count the frequency for each chars in the file
    for freq in res:
        if freq[0] not in freqdict:
            freqdict[freq[0]] = 1
        else:
            freqdict[freq[0]] += 1
f.close()

# Show chars frequency statistic with bar chart, feel free to turn it on anytime you want!

plt.xlabel('Characters')
plt.ylabel('Frequency of appearance')
plt.bar(freqdict.keys(), freqdict.values(), 1.0, color='g')


#--------------------#
# 2. HUFFMAN CODING
#--------------------#

# Prepare for sorting : flip the key and values, such that {key : values} -> (values, key)
fd = [(freq , k) for (k, freq) in freqdict.items()]
fd = sorted(fd) #sort by value (frequency)
nodes = fd.copy()

while len(nodes) > 1:
    (freq1, k1) = nodes[0] # first smallest frequency char in current nodes list
    (freq2, k2) = nodes[1] # second smallest frequency char in current nodes list
    nodes = nodes[2:] # pop both from nodes list
    node = TreeNode(k1, k2)
    nodes.append((freq1 + freq2, node)) # notice : nodes list contains tuples, either in format (integer, integer) or (integer, TreeNode obj.)
    nodes = sorted(nodes, key=lambda x: x[0])

# Create Huffman Tree (nodes[0][1] will be the tree's root) 
huffmanCodeTable = huffman_code_tree(nodes[0][1])


#-------------#
# 3. ENCODING
#-------------#
with open(encoded_path, "w") as fw, open(data_path, "r", encoding="utf-8") as fr:
    '''
    Bit allocation, respectively:
    8 bits for Huffman Dictionary size
    
    *8 bits for original bit strings of a certain character
    *8 bits for Huffman binary string length (n) of that character
    *n-bit(s) Huffman binary string of that character
    All binary allocation marked by '*' is repeated for all chars in the Huffman dictionary
    
    The rest is the main data, which is bunch of binary strings represent the characters from the text based on the Huffman dictionary
    '''
    dictlength = str(bin(len(huffmanCodeTable))[2:]).zfill(8)
    fw.write(dictlength)
    for (char, huffcodestr) in huffmanCodeTable.items():
        oribit = str(bin(ord(char))[2:]).zfill(8)
        bitlen = str(bin(len(huffcodestr))[2:]).zfill(8)
        frame = oribit + bitlen + huffcodestr
        fw.write(frame)

    while True:
        res = fr.readline()
        if not res:
            break
        else:
            for c in res:
                tmp = huffmanCodeTable[c]
                fw.write(huffmanCodeTable[c])
                
    fw.close(); fr.close()
    
    
#-------------#
# 4. DECODING
#-------------#
with open(result_path, "w") as fw, open(encoded_path, "r") as fr:
    
    dictlength = fr.read(8); dictlength = int(dictlength, 2)
    
    #recreate Huffman dictionary
    huffmanDict = dict()
    for _ in range(dictlength):
        char = fr.read(8); char = chr(int(char, 2))
        bitlen = fr.read(8); bitlen = int(bitlen, 2)
        
        huffmancodestring = fr.read(bitlen)
        huffmanDict[huffmancodestring] = char
    
    #rewrite the text
    tmpbit = ''
    while True:
        bit = fr.read(1)
        
        if not bit:
            break
        
        else:
            tmpbit += bit
            if tmpbit in huffmanDict:
                fw.write(huffmanDict[tmpbit])
                tmpbit = ''
    fw.close(); fr.close()