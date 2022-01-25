import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
 
# loading train data set in dataframe from train_data.csv file
df = pd.read_csv("../dataset/diabetes.csv")

# # # getting train data from the dataframe
train =df.drop(columns = ['Outcome'])
print(train)
 
# # getting target data from the dataframe
target = df["Outcome"]

# Splitting between train data into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.20)

model_1 = LogisticRegression()
model_2 =  GaussianNB()
model_3 = RandomForestClassifier()
 
# Making the final model using voting classifier

final_model =VotingClassifier(
    estimators=[('lr', model_1), ('rf', model_2), ('gnb', model_3)],
    voting='soft', weights=[2, 1, 2])
 
# training all the model on the train dataset
final_model.fit(X_train, y_train)
 
# predicting the output on the test dataset
pred_final = final_model.predict(X_test)
 

for clf, label in zip([model_1, model_2, model_3, final_model], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))