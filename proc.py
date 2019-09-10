from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from flask import Flask,render_template,request


app=Flask(__name__)#initialize flask app
df=pd.read_csv('dataset.csv',usecols=['CGPA','Backlogs','Placed'])#read data from csv file
y=df['Placed']#1-d array output of placed
x=df.drop(columns='Placed')
sm = SMOTE()#create smote object
x,y=sm.fit_resample(x,y)#datasize increase cheeyum
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=1)


clf1=GaussianNB()
clf2=RandomForestClassifier()
#print(pd.value_counts(y_train))
clf1=clf1.fit(x_train,y_train)# training
clf2=clf2.fit(x_train,y_train)
print("Classifier 1 accuracy score:",clf1.score(x_test,y_test))
print("Classifier 2 accuracy score:",clf2.score(x_test,y_test))
x=[['6.72','1']]#sample input
x=np.array(x,dtype='Float32').reshape(1,-1)
print("predicted",clf1.predict(x)[0])#predict output 1 or 0
print("predicted",clf2.predict(x)[0])
from sklearn.metrics import confusion_matrix
y_pred=clf1.predict(x_test)#naive bayes confusion matrix
print(confusion_matrix(y_test,y_pred))
y_pred=clf2.predict(x_test)#random forest confusion matrix
print(confusion_matrix(y_test,y_pred))

#flask
@app.route('/')#127.5.6/
def form():
    return render_template('form.html')
@app.route('/predict', methods=['POST'])#127.5.1/predict
def result():
    if request.method=='POST':
        name=request.form['name']
        cgpa=request.form['cgpa']
        backlogs=request.form['bl']
        x=[[cgpa,backlogs]]
        x=np.array(x,dtype='Float32').reshape(1,-1)
        predicted=clf1.predict(x)[0]
        if(predicted==1):
            pred=" "+name+" is likely to be placed"
        else:
            pred="Try Harder to get placed"      
        return render_template('predict.html',pred=pred)
    
    else:
        print("ERROR")
@app.route('/contribute')
def cntr():
    return render_template('contribute.html')
@app.route('/thanks', methods=['POST'])
def tnks():
    return render_template('thanks.html')
    app.logger.warning(result)
if __name__ == "__main__":
    app.run(debug='True')
#