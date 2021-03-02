import ast
import re
from nltk.corpus import stopwords
pattern = re.compile(r'(,){2,}')

stop_words = set(stopwords.words('english')) 

f = open("abstracts.txt", "r", encoding='utf8')
for i, l in enumerate(f):
    if l == '\n':
        continue
    if i == 0:
        abs = l
        break
f.close()

dic = {}
id = l.split("----")[0]
inv = "".join(l.split("----")[1:])
res = ast.literal_eval(inv) 
abstract =[ "" for i in range(res["IndexLength"])]
inv_indx=  res["InvertedIndex"]

print(inv_indx)

for i in inv_indx:
    try:
        if (i.isalpha() or i[-1] in (',','.',':')) and i not in stop_words:
            for j in inv_indx[i]:
                if i[-1] in (',',':'):
                    abstract[j] = i[:-1].lower()
                else:
                    abstract[j] = i.lower()
    except IndexError:
        pass
sentences = []
abstract = re.sub(pattern, ',', ",".join(abstract))
abstract = abstract.split('.')
for i, sen in enumerate(abstract):
    if sen == '':
        continue
    if sen[0] == ',':
        sentence = sen[1:]
    else:
        sentence = sen
    sentences.append(sentence.split(','))
    
print(sentences)