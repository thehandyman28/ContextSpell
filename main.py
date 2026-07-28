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
    return final[:3]
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
       return scores[:3]
   
   
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
def spellCheckForWeb(text):
    all_results = []
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
            bigram_score = getBigramScore(clean, i, cleaned_words)
            freq_score   = word_frequency(clean, 'en')


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
                    all_results.append({
                        'word':        org,
                        'type':        'context',
                        'suggestions': [r[0] for r in result[:3]],
                        'scores': {
                            'bigram':    bigram_score,
                            'frequency': freq_score,
                            'combined':  round((0.4 * freq_score) + (0.6 * bigram_score), 4)
                        },
                        'weights': {
                            'frequency': 0.4,
                            'bigram':    0.6
                        }
                    })
                    isWordArr.append(True)

                    #print(f'  {org} -> CONTEXT ERROR did you mean: {result}')

            else:
                isWordArr.append(False)
                suggestions=getSuggestions(clean,i,cleaned_words)
                if suggestions:
                    top = suggestions[0]
                    all_results.append({
                        'word':        org,
                        'type':        'unknown',
                        'suggestions': [s[0] for s in suggestions[:3]],
                        'scores': {
                            'edit_distance': top[1],
                            'combined':      round(top[2], 4),
                            'frequency':     round(word_frequency(top[0], 'en'), 6),
                            'bigram':        getBigramScore(top[0], i, cleaned_words)
                        },
                        'weights': {
                            'frequency': 0.4,
                            'bigram':    0.6
                        },
                        'all_candidates': [
                            {
                                'word':     s[0],
                                'distance': s[1],
                                'score':    round(s[2], 4)
                            }
                            for s in suggestions
                        ]
                    })
        return all_results

def evaluate():
    
    from spellchecker import SpellChecker
    spell = SpellChecker()
    
    # Group 1 - unknown word errors
    unknown_tests = [
        ("I recieved your email this morning",      "recieved",   "received"),
        ("The goverment passed a new law",          "goverment",  "government"),
        ("She beleived everything he said",         "beleived",   "believed"),
        ("We must seperate the two groups",         "seperate",   "separate"),
        ("It occured to me something was wrong",    "occured",    "occurred"),
        ("He was completly wrong about it",         "completly",  "completely"),
        ("The commitee met on Thursday",            "commitee",   "committee"),
        ("She was absolutly certain",               "absolutly",  "absolutely"),
        ("The experment failed last Tuesday",       "experment",  "experiment"),
        ("It was an embarasing moment",             "embarasing", "embarrassing"),
    ]
    
    # Group 2 - context errors
    context_tests = [
        ("The affect on the results was significant", "affect",   "effect"),
        ("Please insure the door is locked",          "insure",   "ensure"),
        ("The elicit trade caused problems",          "elicit",   "illicit"),
        ("She could not accept the advise",           "advise",   "advice"),
        ("The allusion to the event was clear",       "allusion", "illusion"),
        ("The amoral behaviour was shocking",         "amoral",   "immoral"),
    ]
    
    # Group 3 - correct sentences
    correct_tests = [
        "The effect on the results was significant.",
        "Please ensure the door is locked tonight.",
        "I received your email this morning.",
        "The government passed a new law yesterday.",
        "She believed everything he said to her.",
        "We must separate the two groups immediately.",
        "The experiment failed last Tuesday morning.",
        "The illicit trade caused major problems.",
        "He was imminent to arrive any moment.",
        "The immigrant population settled here.",
    ]
    
    print("=" * 70)
    print("CONTEXTSPELL vs HUNSPELL EVALUATION")
    print("=" * 70)
    
    # --------------------------------------------------------
    # GROUP 1 - UNKNOWN WORDS
    # --------------------------------------------------------
    print("\nGROUP 1 — Unknown Word Errors")
    print(f"{'Word':<15} {'Expected':<15} {'CS Top1':<15} {'CS Top5':<6} {'HUN Top1':<15} {'HUN Top5':<6}")
    print("-" * 70)
    
    cs_top1_unknown  = 0
    cs_top5_unknown  = 0
    hun_top1_unknown = 0
    hun_top5_unknown = 0
    
    for sentence, flagged, expected in unknown_tests:
        tokens   = sentence.lower().split()
        position = tokens.index(flagged.lower()) if flagged.lower() in tokens else 0
        
        # ContextSpell suggestions
        cs_suggestions = getSuggestions(flagged.lower(), position, tokens)
        cs_words       = [s[0] for s in cs_suggestions]
        cs_top1_word   = cs_words[0] if cs_words else 'none'
        cs_t1          = cs_top1_word == expected
        cs_t5          = expected in cs_words
        
        if cs_t1: cs_top1_unknown += 1
        if cs_t5: cs_top5_unknown += 1
        
        # Hunspell suggestions
        hun_candidates = list(spell.candidates(flagged.lower()) or [])
        hun_words      = sorted(
            hun_candidates,
            key=lambda w: word_frequency(w, 'en'),
            reverse=True
        )[:5]
        hun_top1_word  = hun_words[0] if hun_words else 'none'
        hun_t1         = hun_top1_word == expected
        hun_t5         = expected in hun_words
        
        if hun_t1: hun_top1_unknown += 1
        if hun_t5: hun_top5_unknown += 1
        
        cs_t1_str  = f'✓ {cs_top1_word}'  if cs_t1  else f'✗ {cs_top1_word}'
        hun_t1_str = f'✓ {hun_top1_word}' if hun_t1 else f'✗ {hun_top1_word}'
        
        print(f"{flagged:<15} {expected:<15} {cs_t1_str:<15} {'✓' if cs_t5 else '✗':<6} {hun_t1_str:<15} {'✓' if hun_t5 else '✗':<6}")
    
    print(f"\n  {'Metric':<30} {'ContextSpell':>12} {'Hunspell':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Top-1 accuracy':<30} {cs_top1_unknown*10:>11}% {hun_top1_unknown*10:>11}%")
    print(f"  {'Top-5 accuracy':<30} {cs_top5_unknown*10:>11}% {hun_top5_unknown*10:>11}%")
    
    # --------------------------------------------------------
    # GROUP 2 - CONTEXT ERRORS
    # --------------------------------------------------------
    print("\nGROUP 2 — Context Errors")
    print(f"{'Word':<15} {'Expected':<15} {'CS caught':<12} {'CS suggestion':<15} {'HUN caught':<12}")
    print("-" * 70)
    
    cs_context_caught   = 0
    cs_context_correct  = 0
    hun_context_caught  = 0
    
    for sentence, flagged, expected in context_tests:
        
        # ContextSpell
        results       = spellCheckForWeb(sentence)
        ctx_results   = [r for r in results if r['type'] == 'context']
        flagged_words = [r['word'].lower() for r in ctx_results]
        cs_caught     = flagged.lower() in flagged_words
        
        cs_sugg = '—'
        if cs_caught:
            cs_context_caught += 1
            match = next((r for r in ctx_results if r['word'].lower() == flagged.lower()), None)
            if match and match['suggestions']:
                cs_sugg = match['suggestions'][0]
                if cs_sugg == expected:
                    cs_context_correct += 1
        
        # Hunspell — will never catch context errors
        # flagged word is valid so Hunspell passes it
        hun_caught = False
        if flagged.lower() not in spell:
            hun_caught = True
            hun_context_caught += 1
        
        print(f"{flagged:<15} {expected:<15} {'✓' if cs_caught else '✗':<12} {cs_sugg:<15} {'✓' if hun_caught else '✗ (valid word)':<12}")
    
    n = len(context_tests)
    print(f"\n  {'Metric':<30} {'ContextSpell':>12} {'Hunspell':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Errors caught':<30} {f'{cs_context_caught}/{n}':<12} {f'{hun_context_caught}/{n}'}")
    print(f"  {'Correct suggestion':<30} {f'{cs_context_correct}/{n}':<12} {'N/A'}")
    print(f"  {'Detection rate':<30} {round(cs_context_caught/n*100):>11}% {round(hun_context_caught/n*100):>11}%")
    
    # --------------------------------------------------------
    # GROUP 3 - FALSE POSITIVES
    # --------------------------------------------------------
    print("\nGROUP 3 — False Positives on Correct Sentences")
    print(f"{'Sentence[:40]':<42} {'CS flags':<10} {'HUN flags':<10}")
    print("-" * 70)
    
    cs_fp  = 0
    hun_fp = 0
    
    for sentence in correct_tests:
        
        # ContextSpell false positives
        cs_results  = spellCheckForWeb(sentence)
        cs_count    = len(cs_results)
        cs_fp      += cs_count
        
        # Hunspell false positives
        tokens      = word_tokenize(sentence.lower())
        hun_flags   = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in spell]
        hun_count   = len(hun_flags)
        hun_fp     += hun_count
        
        cs_str  = f'{cs_count} ({", ".join(r["word"] for r in cs_results)})' if cs_count else '0'
        hun_str = f'{hun_count} ({", ".join(hun_flags)})' if hun_count else '0'
        
        print(f"{sentence[:40]:<42} {cs_str:<10} {hun_str:<10}")
    
    n = len(correct_tests)
    print(f"\n  {'Metric':<30} {'ContextSpell':>12} {'Hunspell':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Total false positives':<30} {cs_fp:>12} {hun_fp:>12}")
    print(f"  {'False positive rate':<30} {round(cs_fp/n*100):>11}% {round(hun_fp/n*100):>11}%")
    
    # --------------------------------------------------------
    # FINAL SUMMARY TABLE
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Metric':<35} {'ContextSpell':>12} {'Hunspell':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Unknown word top-1 accuracy':<35} {cs_top1_unknown*10:>11}% {hun_top1_unknown*10:>11}%")
    print(f"  {'Unknown word top-5 accuracy':<35} {cs_top5_unknown*10:>11}% {hun_top5_unknown*10:>11}%")
    print(f"  {'Context error detection rate':<35} {round(cs_context_caught/len(context_tests)*100):>11}% {0:>11}%")
    print(f"  {'Context correct suggestion':<35} {round(cs_context_correct/len(context_tests)*100):>11}% {'N/A':>12}")
    print(f"  {'False positive rate':<35} {round(cs_fp/len(correct_tests)*100):>11}% {round(hun_fp/len(correct_tests)*100):>11}%")
    print("=" * 70)
if __name__=='__main__':
        nltk_data()
        dictionary,dictionary_by_letter=bulidDictionarty()
        bigramCounts=buidBigram()
        text=input("Enter a string to spell check:")
        evaluate()
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


