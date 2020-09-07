import xlrd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC


#Run the Script using python Command

file=xlrd.open_workbook("Project_NLP_twitter_Data-10.xlsx")
sheet = file.sheet_by_index(0)

tweets=[]
data=[]
for i in range(1,sheet.nrows):
    tweets.append(sheet.cell_value(i,0))
    data.append(sheet.cell_value(i, 1).strip().lower())


vectorizer = TfidfVectorizer(stop_words='english',min_df=2)
X = vectorizer.fit_transform(tweets)


X_train=X[:900].toarray()
Y_train=data[:900]
X_test=X[900:].toarray()
Y_test=data[900:]

#clf = SVC(gamma="auto")

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