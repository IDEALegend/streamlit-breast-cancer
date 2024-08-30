import pandas as pd
import numpy as np
import pickle

def create_model(data):
        X = data.drop('diagnosis', axis=1)
        y = data['diagnosis']

        #Encoding y variable
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

        #Scale the data
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(X)

        #Train the data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

        #Using Logistic Regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=0)
        model.fit(X_train, y_train)


        #test the model
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = model.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))
        print('Classification report:\n', classification_report(y_test, y_pred))
        return model, sc


def get_clean_data():
        data = pd.read_csv('../data/data.csv')
        data = data.drop('id', axis=1)
        return data


def main ():
        data = get_clean_data()
        model, sc = create_model(data)
        #test_model(model)
        with open('../model/model.pkl', 'wb') as f:
                pickle.dump(model, f)
        with open('../model/scaler.pkl', 'wb') as f:
                pickle.dump(sc, f)

if __name__== '__main__':
        main()