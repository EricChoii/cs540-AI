import sys
import math
from string import ascii_uppercase

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26
    
    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    for c in list(ascii_uppercase):
        X[c]=0
    
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        while 1:
            char = f.read(1)
            if not char:
                break   
            if not char.isalpha():
                continue  
            X[char.upper()] += 1
    f.close()
    
    return X

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
print("Q1")
X = shred('letter.txt')
for c in list(ascii_uppercase):
    print(c, X[c])
    
print("Q2")
e, s = get_parameter_vectors()
print(format(X['A']*math.log(e[0]), ".4f"))
print(format(X['A']*math.log(s[0]), ".4f"))

print("Q3")
esum = ssum = 0
for i in range(0, 26):
    esum += X[chr(i+65)]*math.log(e[i])
    ssum += X[chr(i+65)]*math.log(s[i])
fey = math.log(.6) + esum
fsy = math.log(.4) + ssum
print(format(fey, ".4f"))
print(format(fsy, ".4f"))

print("Q4")
prob = 0
if fsy - fey >= 100:
    prob = 0
elif fsy - fey <= -100:
    prob = 1
else:    
    prob = 1 / (1 + math.exp(fsy-fey))
print(format(prob, ".4f"))
