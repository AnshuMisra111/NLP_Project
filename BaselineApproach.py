import xlrd
import re
import numpy
from gensim.models import Word2Vec
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report


#Run the Script using python Command

file=xlrd.open_workbook("Project_NLP_twitter_Data-10.xlsx")
sheet = file.sheet_by_index(0)

tweets=[]
data=[]
for i in range(1,sheet.nrows):
    tweets.append(sheet.cell_value(i,0))
    data.append(sheet.cell_value(i, 1).strip().lower())



sentences=[]

for element in tweets:
    wordList = re.sub("[^\w]", " ", element).split()
    sentences.append(wordList)

model= Word2Vec(sentences,min_count=1, size=100)
arr=numpy.zeros(shape=(sheet.nrows-1,100))

ind=0
for words in sentences:
    arr2=numpy.zeros(shape=(1,100))
    for element in words:
        arr2 = arr2 + (model[element] * 1)

    arr2= (arr2*(1.0))/len(sentences)
    arr[ind]=arr2
    ind=ind+1


X=arr
X_train=X[:800]
Y_train=data[:800]
X_test=X[800:]
Y_test=data[800:]

clf=BernoulliNB()
clf.fit(X_train,Y_train)
y1=clf.predict(X_test)

right=0
for i in range(len(y1)):
    if(y1[i]==Y_test[i]):
        right=right+1

names = ['negative', 'neutral', 'positive']

print(classification_report(Y_test, y1,))
print("accuracy= ",right*1.0/len(Y_test))