# Import library and methods
import numpy as np
from sklearn.metrics import accuracy_score

# Calculate the accuracy of CNN training result
# Input:
#   result: the variable of CNN train result
# Output:
#   result files will be store in ./results
def calculateAccuracy(result, method = "acc"):
    # Implement and run calculate accuracy method
    if method == "acc":
        test_acc_score = accuracy_score(result[2], result[3])
        print(test_acc_score)
    
    return test_acc_score
