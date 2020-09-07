import xlrd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import numpy
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import unicodedata


#file=open("/Users/anshuman786/Downloads/TOP50cleaned.xlsx")Project_NLP_twitter_Data.xlsx
#file=xlrd.open_workbook("/Users/anshuman786/Downloads/TOP50cleaned.xlsx")
#file=xlrd.open_workbook("/Users/anshuman786/Downloads/Project_NLP_twitter_Data-2.xlsx")
file=xlrd.open_workbook("/Users/anshuman786/Downloads/Project_NLP_twitter_Data-10.xlsx")
sheet = file.sheet_by_index(0)


data=[]
tweets=[]
mapping={'negative':-1,'neutral':0,'positive':1}

print(sheet.nrows)
counter=0
for i in range(1,sheet.nrows):
    counter=counter+1
    #templist=[]
    #templist.append(sheet.cell_value(i,0))
    tweets.append(sheet.cell_value(i,0))
    #templist.append(sheet.cell_value(i, 1))
    #data.append(templist)
    #print(counter)
    data.append(sheet.cell_value(i, 1).strip().lower())
    #data.append(mapping[sheet.cell_value(i, 1).strip().lower()])


#print(sheet.cell_value(0, 2))
#print(tweets)
#print(data)

sentences=[]
senwords=[]

for element in tweets:
    wordList = re.sub("[^\w]", " ", element).split()
    #senwords.append(wordList)
    sentences.append(wordList)

model=Word2Vec(sentences,min_count=1, size=100)

#print(model['launches'])

arr=numpy.zeros(shape=(sheet.nrows-1,100))

tfidf_vec = TfidfVectorizer(min_df=1)
transformed = tfidf_vec.fit_transform(raw_documents=tweets)
index_value={i[1]:i[0] for i in tfidf_vec.vocabulary_.items()}

fully_indexed = []
for row in transformed:
    fully_indexed.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})



#print(len(fully_indexed))

ind=0
for words in sentences:
    arr2=numpy.zeros(shape=(1,100))
    for element in words:
        try:
          #arr2=arr2+(model[element]*fully_indexed[ind][element])
          arr2 = arr2 + (model[element] * 1)
        except:
            arr2 = arr2 + (model[element] * 1)
        #print(model[element])

    arr2= (arr2*(1.0))/len(sentences)
    #arr2=numpy.array(numpy.mean(model[w] for w in words))
    arr[ind]=arr2
    ind=ind+1



print('for model =', arr.shape)
print(len(data))





vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(tweets)
#print(vectorizer.get_feature_names())


print(type(X))

print(X.shape)


X_train=X[:800].toarray()
Y_train=data[:800]
X_test=X[800:].toarray()
Y_test=data[800:]

#X=arr
#X_train=X[:800]
#Y_train=data[:800]
#X_test=X[800:]
#Y_test=data[800:]

#X_train=X_train.astype(int)

print(X_train.shape)
#print(X_test)
print(Y_train)
print(len(Y_test))
#print(type(X_train))


#for (x,y), value in numpy.ndenumerate(X_train):
#   print(type(X_train[x,y]))

print(type(X_train[1]))

#clf = svm.SVC(C=0.1,kernel='rbf')
#clf=tree.DecisionTreeClassifier(criterion='entropy')
#clf=tree.DecisionTreeClassifier()
#clf=AdaBoostClassifier(n_estimators=500,base_estimator=BernoulliNB())
clf=BernoulliNB()
#clf=MultinomialNB()
#clf=GaussianNB()
#clf=RandomForestClassifier(n_estimators=1000)
#clf=MLPClassifier(hidden_layer_sizes=(150,10))
clf.fit(X_train,Y_train)
y1=clf.predict(X_test)

right=0

#print(len(y1)," ",len(Y_test))


for i in range(len(y1)):
    if(y1[i]==Y_test[i]):
        right=right+1
    #print(y1[i]," ",Y_test[i])

print(right," ",len(Y_test))

names = ['negative', 'neutral', 'positive']

print(classification_report(Y_test, y1,))
print("accuracy= ",right*1.0/len(Y_test))