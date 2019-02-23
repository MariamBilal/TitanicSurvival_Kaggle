"""
Titanic Survival Prediction 
"""




"""  ---------------------- Training -----------------------  """



# -------  importing liabraries  

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree



#  ---------   importing training dataset 

train_dataset = pd.read_csv("E:/Kaggle projects/Titanic_project/train.csv" , index_col=None, na_values=['NA'])



#  --------    Droping Name , Ticket and Cabin columns

x_train = train_dataset.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1)



#  -------  Encoding 'Sex' and 'Embarked' column's data by using LabelEncoder

encoder = LabelEncoder()

Sex_encd = encoder.fit_transform(x_train.Sex)
emb_encd = encoder.fit_transform(x_train.Embarked.astype(str))



#  ------  converting the encoded data fields into Pandas DataFrame 

sex = pd.DataFrame(data = Sex_encd, columns = ['Sex'])
embarked = pd.DataFrame(data = emb_encd , columns = ['Embarked'])



#  ------ Droping Sex and Embraked column from our x_train dataset because 
#         we have encoded these columns and save it into a new variable 
   
x_train = x_train.drop(labels=[ 'Sex','Embarked'], axis=1)
print(x_train.info()) 



#  ------  Concatinating the encoded two columns into our x_train dataset.
#          Now our dataset have the encoded columns

x_train = pd.concat([x_train , sex , embarked], axis = 1)

#  ------  Print ' x_train.info() ' to see that columns are now added in 
#          dataste and how many rows each column contain

print(x_train.info())  



#  -------  Droping the Null (NaN) value rows  and 
#          Print ' x_train.info() ' to check the how many rows our dataset have now.

x_train = x_train.dropna()
print(x_train.info())



#  ------  dividing the x_train dataset into train data ( x_train ) and target data ( y_train )

#    extract the'Survived' column from x_train ( because that our target column ) 
#                   --> make dataframe and save it into y_train

#   drop Survived column from x_train

y_train = pd.DataFrame(data = x_train['Survived'] , columns = ['Survived'])
x_train = x_train.drop(labels = ['Survived'], axis = 1)



#   -----  print info about our final x_train and y_train dataset

print(x_train.info())
print(y_train.info())

#  ------  print frist 20 rows from x_train and y_train dataset

print(x_train.head(20))
print(y_train.head(20))



#  -------  training the classifier 
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(x_train , y_train)
print(classifier)






""" ------------------------------------ testing ------------------------------- """

#  -----  we can going to do the same thing with our testing dataset 


test_dataset = pd.read_csv("E:/Kaggle projects/Titanic_project/test.csv" , index_col=None )


x_test = test_dataset.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1)


Sex_encd = encoder.fit_transform(x_test.Sex)
emb_encd = encoder.fit_transform(x_test.Embarked.astype(str))


sex = pd.DataFrame(data = Sex_encd, columns = ['Sex'])
embarked = pd.DataFrame(data = emb_encd , columns = ['Embarked'])



x_test = x_test.drop(labels=[ 'Sex','Embarked'], axis=1)
print(x_test.info())


x_test = pd.concat([x_test , sex , embarked], axis = 1)
print(x_test.info())


#  ------  Going to fill empty rows with '0'

x_test = x_test.fillna(0)
print(x_test.info())



#  ---- Now our testing dataset is ready , we are going to make predictions

predictions = classifier.predict(x_test)



#  ----- we need Passenger ID and Survived (predicted values) into a file So,
#    savinging Passenger ID and Survived columns into a DataFrame
 
pid = x_test['PassengerId']
passengerID = pd.DataFrame(data = pid , columns = ['PassengerId'])
survived = pd.DataFrame(data = predictions , columns = ['Survived'])



#  ----- concatinating both variables into one final DataFrame 
test_results = pd.concat([passengerID  , survived] , axis = 1)


print(test_results.info())
print(test_results.head(30))


#  ----- changing the data type from float64 to int32 before saving it into a csv file

results = test_results[['PassengerId', 'Survived']].astype(int)
print(results.info())

#  ----- saving into a csv file 

excle_file = results.to_csv("Test_Predictions.csv" , sep = ',' ,  index = False)



