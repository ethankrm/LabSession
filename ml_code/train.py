from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_evaluate(model,X_train, X_test, Y_train, Y_test):
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(Y_test, predictions)
    cm = confusion_matrix(Y_test, predictions)

    if hasattr(model, "predict_proba"):
        Y_prob = model.predict_proba(X_test)[:,1]
    else:
        Y_prob = model.decision_function(X_test)

    return accuracy, cm, Y_test, Y_prob
    