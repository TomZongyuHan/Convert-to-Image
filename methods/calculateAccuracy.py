# Import library and methods
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Calculate the accuracy of CNN training result
# Input:
#   result: the variable of CNN train result
# Output:
#   result files will be store in ./results
def calculateAccuracy(result):
    # Implement and run calculate accuracy method
    test_acc_score = accuracy_score(result[0], result[1])
    test_f1_score = f1_score(result[0], result[1], average='weighted')

    # test_acc_score = f1_score(result[0], result[1], average='macro')
    # test_acc_score = f1_score(result[0], result[1], average='weighted')
    # test_acc_score = precision_score(result[0], result[1], average="macro")
    # test_acc_score = recall_score(result[0], result[1], average="macro")
    # test_f1_score = f1_score(result[0], result[1], average='micro')
    return [test_acc_score, test_f1_score]
