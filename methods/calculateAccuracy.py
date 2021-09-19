# Import library and methods
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Calculate the accuracy of CNN training result
# Input:
#   result: the variable of CNN train result
# Output:
#   result files will be store in ./results
def calculateAccuracy(result, method):
    # Implement and run calculate accuracy method
    if method == "acc":
        test_acc_score = accuracy_score(result[0], result[1])
    elif method =="f1_macro":
        test_acc_score = f1_score(result[0], result[1], average='macro')
    elif method =="f1_micro":
        test_acc_score = f1_score(result[0], result[1], average='micro')
    elif method =="f1_weighted":
        test_acc_score = f1_score(result[0], result[1], average='weighted')
    elif method =="precision":
        test_acc_score = precision_score(result[0], result[1], average="macro")
    elif method =="recall":
        test_acc_score = recall_score(result[0], result[1], average="macro")
    print(test_acc_score)

