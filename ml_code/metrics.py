import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

def print_metrics(accuracy, confusion_matrix, Y_test, Y_prob):
    print(f"Accuracy: {accuracy}")
    print ("Confusion Matrix")
    print(confusion_matrix)

    fpr, tpr, _ = roc_curve(Y_test,Y_prob)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()