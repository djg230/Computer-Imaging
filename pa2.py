import re
import numpy as np
import sys

from collections import Counter
ETAOIN = 'ETAOINSHRDLCUMWFGYPBVKJXQZ'
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
common = open("commonwords.txt")

def blankMap():
  return  {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'F': [], 'G': [], 'H': [], 'I': [], 'J': [], 'K': [], 'L': [], 'M': [], 'N': [], 'O': [], 'P': [], 'Q': [], 'R': [], 'S': [], 'T': [], 'U': [], 'V': [], 'W': [], 'X': [], 'Y': [], 'Z': []}

def oneLetter(message):
  return re.findall(r'\b\w\b',text)

def twoLetters(message):
  return re.findall(r'\b\w{2}\b',text)

def threeLetters(message):
  return re.findall(r'\b\w{3}\b',text)

def mode(list):
  occ = Counter(list)
  return occ.most_common(1)[0][0]

def getWordPattern(word):
  word = word.upper()
  nextNum = 0
  letterNums = {}
  wordPattern = []
  for letter in word:
    if letter != '\n':
      if letter not in letterNums:
        letterNums[letter] = str(nextNum)
        nextNum += 1
      wordPattern.append(letterNums[letter])
  return '.'.join(wordPattern)


#           QTADZHKCJUIYGOMRBSXELVNPFW
#filename = input();
key = [0] * 26
used = []
filename = sys.argv[1]
with open(filename, 'r') as f:
  text = f.read().strip();

freq = {}
oFreq = {}
wFreq = {}
tFreq = {}

for x in text:
  for i in x.upper():
    if i in alphabet:
      if i in freq:
        freq[i] += 1
      else:
        freq[i] = 1
freq = sorted([(x,freq[x]) for x in freq],key = lambda y: y[1], reverse=True)

three = threeLetters(text)
two = twoLetters(text)
one = oneLetter(text)

for o in one:
  o = o.upper()
  if o in alphabet:
    if o in oFreq:
      oFreq[o] += 1
    else:
      oFreq[o] = 1
oFreq = sorted([(x,oFreq[x]) for x in oFreq],key = lambda y: y[1], reverse=True)
print(oFreq)
for w in two:
  w = w.upper()
  #if o in alphabet:
  if w in wFreq:
    wFreq[w] += 1
  else:
    wFreq[w] = 1
wFreq = sorted([(x,wFreq[x]) for x in wFreq],key = lambda y: y[1], reverse=True)

for t in three:
  t = t.upper()
  #if t in alphabet:
  if t in tFreq:
    tFreq[t] += 1
  else:
    tFreq[t] = 1
tFreq = sorted([(x,tFreq[x]) for x in tFreq],key = lambda y: y[1], reverse=True)

the = tFreq[0][0]
an = tFreq[1][0]
of = wFreq[0][0]
key[7] = the[1]
key[13] = an[1]
key[3] = an[2]
key[4]  = freq[0][0] #set E in the key to most freq
key[19] = the[0]
key[0] = oFreq[0][0] # a = most frequent singleletter\
#key[14] = of[0]
#key[5] = of[1]

for k in key:
  used.append(k)

bigMap = blankMap()

with open(filename,'r') as f:
  for line in f:
    for word in line.split():
      common = open("commonwords.txt")
      word = word.upper()
      
      map = blankMap()
      candidates = []
      letterCand = {}
      solvedLetters = [0] * len(word)
      count = 0
      c1 = 0
      c2 = 0
      c3 = 0 
      c4 = 0 
      c5 = 0
      i = 0
      # finds letters already solved in key
      for l in word:
        for k in key:
          if l == k:
            solvedLetters[c1] = alphabet[c2]
            c2 = 0
          c2 += 1
        c1 += 1
        c2 = 0

      # gets list of possible candiates encrypted word can represent based on word patterns
      pattern = getWordPattern(word)
      
      for c in common:
        cword = c.rstrip()
        p = getWordPattern(cword)
        if p == pattern:    
          candidates.append(cword.upper())
          count += 1
      # removes candidates that dont match with data we already have from built key
      trueCand = [x for x in candidates]
      badCand= False
      for c in candidates:
        c3 = 0
        for l in c:
          if solvedLetters[c3] != 0:
            if l != solvedLetters[c3]:
              badCand = True
          c3 += 1 
        if badCand:
          trueCand.remove(c)
        badCand = False
        
      #print(trueCand)
      for w in word:
        for e in trueCand:
          l = e[c4]
          if solvedLetters[c4] != l:
            if w in alphabet:
              if l not in map[w]:
                 map[w].append(l)
        c4 += 1

      for letter in map:
        for l in bigMap:
          if letter == l:
            if len(map[letter]) == 1:
              bigMap[l].append(map[letter][0])



for letter in bigMap:
  counter = 0
  if bigMap[letter]:
    for a in alphabet:
     if a == letter:
        most = mode(bigMap[letter])
        
        for i in alphabet:
          if i == most:
            if(key[counter] == 0):
              key[counter] = a
              used.append(a)
          counter += 1
p = 0

 
#check if solutions found for least common letters, if not, set to lowest frequency
if key[25] == 0:
  key[25] = freq[25][0]   #Z

if key[16] == 0:
  key[16] = freq[24][0]   #Q

if key[9] == 0:
  key[9] = freq[23][0]   #J

if key[23] == 0:
  key[23] =  freq[22][0]  #X

true = "FESCKBPOHRJDNMZYAWVLUXIGTQ" #'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
o = 0
per = 0
for k in key:
  if k == true[o]:
    per += 1
  o += 1

#prints for testing
#print(key)
#print(true)
#print("percent correct: ")
#print(per / 26)

fin = " "
for el in key:
  fin += str(el)

k = open("key.txt","w+")
k.write(fin)
k.close()

