from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):

    data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis="columns", inplace=True)

    categorical_col = []
    for column in data.columns:
        if data[column].dtype == object and len(data[column].unique()) <= 50:
            categorical_col.append(column)

    data['Attrition'] = data.Attrition.astype("category").cat.codes
    
    categorical_col.remove('Attrition')

    label = LabelEncoder()
    for column in categorical_col:
        data[column] = label.fit_transform(data[column])

    print(data.head())

    X = data.drop('Attrition', axis=1)
    Y = data.Attrition

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

    return X_train, X_test, Y_train, Y_test