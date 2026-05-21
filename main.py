import nltk

try: 
    nltk.find('corpora/words') 
except LookupError:
    nltk.download('words')

dictionary=set()
words_list=nltk.corpus.words.words()

for word in words_list:
    word=word.lower()
    word=word.strip()
    if len(word)<3 or not word.isalpha():
        continue
    dictionary.add(word)
print("size:",len(dictionary))

text=input("Enter a string to spell check:")
textArr=text.split(' ')


for i in range(len(textArr)):
    textArr[i]=textArr[i].strip()
    textArr[i]=textArr[i].lower()
    if textArr[i]==' ' or textArr[i]=='':
        textArr.pop(i)



print(textArr)
isWordArr=[]
for t in textArr:
    if t in dictionary:
        isWordArr.append(True)
    else:
        isWordArr.append(False)
print(isWordArr)
