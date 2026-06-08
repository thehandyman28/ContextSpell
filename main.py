import nltk
from nltk.metrics import distance
from collections import defaultdict
from wordfreq import word_frequency
from  nltk.corpus import brown
import os
from collections import Counter 
import pickle
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('punkt_tab')


CONFUSION_SET = {

    'affect':    ['effect'],
    'effect':    ['affect'],
    'accept':    ['except'],
    'except':    ['accept'],
    'amoral':    ['immoral'],
    'immoral':   ['amoral'],
    'allusion':  ['illusion'],
    'illusion':  ['allusion'],
    'emigrate':  ['immigrate'],
    'immigrate': ['emigrate'],
    'eminent':   ['imminent'],
    'imminent':  ['eminent'],
    'elicit':    ['illicit'],
    'illicit':   ['elicit'],
    'ensure':    ['insure'],
    'insure':    ['ensure'],
    'advise':    ['devise'],
    'devise':    ['advise'],
    'complement': ['implement'],
    'implement':  ['complement'],
}
lemmatizer = WordNetLemmatizer()
def nltk_data():
    try: 
        nltk.find('corpora/words') 
    except LookupError:
        nltk.download('words')
    try: 
        nltk.find('corpora/brown') 
    except LookupError:
        nltk.download('brown')
    try:
        nltk.find('punkt')
    except:
        nltk.download('punkt')

    try:
        nltk.find('punkt_tab')
    except:
        nltk.download('punkt_tab')
    try:
        nltk.find('corpora/wordnet')        
    except LookupError:
        nltk.download('wordnet')            


def buidBigram():
    name="bigram_counts.pkl"
    if os.path.exists(name):
        with open(name,'rb') as f:
            return pickle.load(f)
    bigram_counts=Counter()
    for sen in brown.sents():
        clean=['<s>']
        for word in sen:
            w=word.lower().strip().strip('.,!?')
            if w.isalpha()  and len(w)>1:
                clean.append(w)
        clean.append('</s>')
        if len(clean)<4:
            continue
        for pair in ngrams(clean,2):
            bigram_counts[pair]+=1
    with open(name,'wb') as f:
        pickle.dump(bigram_counts,f)
    return bigram_counts

def bulidDictionarty():

    dictionary=set()
    dictionary_by_letter=defaultdict(set)
    words_list=nltk.corpus.words.words()

    for word in words_list:
        word=word.lower()
        word=word.strip()
        if len(word)<2 or not word.isalpha():
            continue
        dictionary.add(word)
        dictionary_by_letter[word[0]].add(word)
    return dictionary,dictionary_by_letter
#print("size:",len(dictionary_by_letter["s"]))
#keyList=list(dictionary_by_letter)
#print("keyList:",keyList)


def getCandiates(t):
    setOfLetter=dictionary_by_letter[t[0]] 
    setOfLetterLen=set()
    for s in setOfLetter:
        if abs(len(s)-len(t)) <=2:
            setOfLetterLen.add(s)
    for s in CONFUSION_SET:
        dis=distance.edit_distance(t,s,transpositions=True)
        if dis <=2:
            setOfLetterLen.add(s)
            setOfLetterLen.update(CONFUSION_SET[s])

    
    return setOfLetterLen

def getBigramScore(s,textPosCount,textArr):
    bigram_score=0
    if textPosCount==0:
            bigram_score+=bigramCounts[('<s>',s)]*1.0
    else:
        bigram_score=bigramCounts[(textArr[textPosCount-1],s)]*1.0    
    if textPosCount==len(textArr)-1:
        bigram_score+=bigramCounts[(s,'</s>')]*1.0
    else:
        bigram_score+=bigramCounts[(s,textArr[textPosCount+1])]*1.0
    if textPosCount>=2:
        bigram_score+=bigramCounts[(textArr[textPosCount-2],s)]*0.5
    elif textPosCount==1:
        bigram_score+=bigramCounts[('<s>',s)]*0.5
    if textPosCount<=len(textArr)-3:
        bigram_score+=bigramCounts[(s, textArr[textPosCount+2])] * 0.5
    elif textPosCount==len(textArr)-2:
        bigram_score+=bigramCounts[(s,'</s>')]*0.5
    return  bigram_score

def getSuggestions(t,textPosCount,textArr):
    setOfLetterLen=getCandiates(t)
    sugArray=[]
    for s in setOfLetterLen:
        dis=distance.edit_distance(t,s,transpositions=True)
        if 0<dis and dis<=2:
            tot_freq=word_frequency(s, 'en')
            bigram_score=getBigramScore(s,textPosCount,textArr)
            sugArray.append((s,dis,tot_freq,bigram_score))            
    if not sugArray:
        return []
    max_freq=max(sugArray,key=lambda item: item[2])[2]
    max_bigram=max(sugArray,key=lambda item: item[3])[3]
    final=[]
    for word,dis,freq,bigram in sugArray:
        norm_freq=freq/max_freq if max_freq >0 else 0
        norm_bigram=bigram/max_bigram if max_bigram >0 else 0
        wTotal=(0.4*norm_freq)+(0.6*norm_bigram)
        final.append((word,dis,wTotal))

    
    final=sorted(final,key=lambda item:(item[1],-item[2]))
    return final[:5]
def clean_token(token):
    clean=token.strip('.,!>;:"\'-()[]{}')

    if not clean:
         return None
    if any(c.isdigit() for c in clean):
        return None
    if not any(c.isalpha() for c in clean):
        return None
    if "'" in clean:
        return None
    if len(clean)<2:
        return None
    return (token,clean.lower())
def should_skip(org, clean, pos):
    titles = {'dr', 'mr', 'mrs', 'ms', 'prof', 'rev', 'st'}
    if clean is None:
        return True
    if not clean.isalpha():
        return True
    if org.isupper() and len(org)>2:
        return True
    if org[0].isupper() and pos>0:
        return True
    if clean.lower() in titles:
        return True
    return False
def checkKnownWord(word, i, cleaned_words):
   org_score=getBigramScore(word,i,cleaned_words)
   if org_score>0:
       return None
   cands=getCandiates(word)
   scores=[]
   for cand in cands:
       if cand==word:
           continue
       dis = distance.edit_distance(word, cand, transpositions=True)
       if dis > 2:
            continue
       cand_score = getBigramScore(cand, i, cleaned_words)
       if cand_score > 0:
            scores.append((cand, cand_score))
   if not scores:
        return None
   scores=sorted(scores,key=lambda item:-item[1])
   if scores[0][1]>15:
       return scores[:5]
   
   
def spellCheck(text):
    sentences=sent_tokenize(text)
    for sentence in sentences:
        tokens=word_tokenize(sentence)
        textArr = []
        for token in tokens:
            clean=clean_token(token)
            if clean is not None:
                textArr.append(clean)
        cleaned_words =[c for _, c in textArr]    
        #print(textArr)
        isWordArr=[]
        for i, (org,clean) in enumerate(textArr):
            if should_skip(org,clean,i):
                continue
            lemma_v = lemmatizer.lemmatize(clean, pos='v')
            lemma_n = lemmatizer.lemmatize(clean, pos='n')

            if clean in dictionary:
                word_to_check = clean
            elif lemma_n in dictionary:
                word_to_check = lemma_n
            elif lemma_v in dictionary:
                word_to_check = lemma_v
            else:
                word_to_check = None

            if word_to_check is not None:
                isWordArr.append(True)
                result = checkKnownWord(word_to_check, i, cleaned_words )
                if result is not None:
                    print(f'  {org} -> CONTEXT ERROR did you mean: {result}')

            else:
                isWordArr.append(False)
                suggestions=getSuggestions(clean,i,cleaned_words)
                if suggestions:
                    print(f'  {org} -> UNKNOWN WORD suggested: {suggestions}')
                else:
                    print(f'  {org} -> UNKNOWN WORD no suggestions found')

        print(isWordArr)

if __name__=='__main__':
        nltk_data()
        dictionary,dictionary_by_letter=bulidDictionarty()
        bigramCounts=buidBigram()
        text=input("Enter a string to spell check:")
        spellCheck(text)
        






# #bigram_counts=buid_bigram()
# # print(len(bigram_counts))
# # print(bigram_counts[('the','effect')])
# # print(bigram_counts[('the','affect')])
# isWordArr=[]
# textPosCount=0
# for t in textArr:
#     if t in dictionary:
#         isWordArr.append(True)
#     else:
#         isWordArr.append(False)
#     setOfLetter=dictionary_by_letter[t[0]] 
#     setOfLetterLen=set()
#     for s in setOfLetter:
#         if abs(len(s)-len(t)) <=2:
#             setOfLetterLen.add(s)
#     sugArray=[]
#     for s in setOfLetterLen:
#         dis=distance.edit_distance(t,s,transpositions=True)
#         if 0<dis and dis<=2:
#             tot_freq=word_frequency(s, 'en')
#             if textPosCount==0:
#                 bigram_score=bigram_counts[('<s>',s)]
#                 bigram_score+=bigram_counts[(t,textArr[textPosCount+1])]
#             elif textPosCount==len(words_list)-1:
#                 bigram_score=bigram_counts[(textArr[textPosCount-1],s)]
#                 bigram_score+=bigram_counts[(s,'</s>')]
#             else:
#                 bigram_score=bigram_counts[(textArr[textPosCount-1],s)]
#                 bigram_score=+bigram_counts[s,(textArr[textPosCount+1])]

                


#             sugArray.append((s,dis,tot_freq))
#     sugArray=sorted(sugArray,key=lambda item:(item[1],-item[2]))
#     print(t," suggested words:",sugArray[:5])
#     textPosCount+=1

    
# print(isWordArr)


