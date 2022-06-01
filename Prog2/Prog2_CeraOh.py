#Cera Oh. CS545 ML. Prog 2. Winter 2022
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split

#Data Path
path = './spambase.data'

#Function that loads the data file
def load_data(file_path):
    with open (file_path, 'r') as data:
        dt = np.genfromtxt(file_path, delimiter=',')
    train_set, test_set = create_sets(dt)
    return train_set, test_set

#Function to create training and test sets
def create_sets(data):
    test_set, train_set = sk.model_selection.train_test_split(data, test_size=0.5, train_size=0.5, random_state=34, shuffle=True, stratify=None)
    return train_set, test_set

#Function to compute the prior probability per class
def prior_prob(data):
    counter = 0
    spam, nspam = 0, 0

    spam_mean = np.zeros(57)#array holding mean value for each feature of spam
    nspam_mean = np.zeros(57)#array holding mean value for each feature of non-spam
    spam_sd = np.zeros(57)#array holding SD value for each feature of spam
    nspam_sd = np.zeros(57)#array holding SD value for each feature of non-spam

    #Calculate Prior Probability & Mean
    for row in data:
        if row[-1] == 1: #spam
            spam += 1
            for index in range(57):
                spam_mean[index] += row[index]
        else: #not spam
            nspam += 1
            for index2 in range(57):
                nspam_mean[index2] += row[index2]
        counter += 1

    prior_spam = spam/counter #P(spam)
    prior_nspam = nspam/counter #P(non-spam)

    spam_mean = spam_mean/spam
    nspam_mean = nspam_mean/nspam

    #Calculate SD
    for row in data:
        if row[-1] == 1: #spam
            for index in range(57):
                spam_sd[index] += (row[index] - spam_mean[index])**2
        else: #not spam
            for index2 in range(57):
                nspam_sd[index2] += (row[index2] - nspam_mean[index2])**2

    spam_sd = np.sqrt(spam_sd/spam)
    nspam_sd = np.sqrt(nspam_sd/nspam)

    #Avoid zero SD value
    count = 0
    for index in spam_sd:
        if index == 0:
            spam_sd[count] = 0.00001
        count += 1

    count = 0
    for index in nspam_sd:
        if index == 0:
            nspam_sd[count] = 0.00001
        count += 1

    return prior_spam, prior_nspam, spam_mean, nspam_mean, spam_sd, nspam_sd

#Gaussian Naive Bayes Algorithm
def gaussian_naive_bayes(test_data, prior_spam, prior_nspam, spam_mean, nspam_mean, spam_sd, nspam_sd):
    tp, fp, tn, fn = 0, 0, 0, 0
    counter = 1    

    for row in test_data:
        prediction = -1
        likelihood_spam = np.zeros(57)#P(xi|spam)
        likelihood_nspam = np.zeros(57)#P(xi|non-spam)
        spam = np.log(prior_spam)
        nspam = np.log(prior_nspam) 

        for index in range(57):
            #Probability density function calculation for spam, broken into parts
            math = 1/(np.sqrt(2 * np.pi) * spam_sd[index])
            p = -((row[index] - spam_mean[index])**2)/(2 * ((spam_sd[index])**2))
            math2 = np.e ** p
            if math2 == 0: #for avoiding log division by 0
                math2 = 10 ** -100
            likelihood_spam[index] = math * math2

            #Probability density function calculation for not spam, broken into parts
            math3 = 1/(np.sqrt(2 * np.pi) * nspam_sd[index])
            p2 = -((row[index] - nspam_mean[index])**2)/(2 * ((nspam_sd[index])**2))
            math4 = np.e ** p2         
            if math4 == 0: #for avoiding log division by 0
                math4 = 10 ** - 100
            likelihood_nspam[index] = math3 * math4

        #Sum for pdf 
            spam += np.log(likelihood_spam[index])
            nspam += np.log(likelihood_nspam[index])

        #Find argmax of the two class to classify 
        if spam > nspam:
            print(f"Test {counter} Prediction: spam.")
            prediction = 1
        else:
            print(f"Test {counter} Prediction: not spam.")
            prediction = 0
        counter += 1

        #Data collection for statistical analysis
        if prediction == 1:
            if row[-1] == 1:
                print(f"Target: {row[-1]}. Correct")
                tp += 1
            if row[-1] == 0:
                fp += 1
                print(f"Target: {row[-1]}. Wrong")
        if prediction == 0:
            if row[-1] == 0:
                tn += 1
                print(f"Target: {row[-1]}. Correct")
            if row[-1] == 1:
                fn += 1
                print(f"Target: {row[-1]}. Wrong")
    
    confusion, accuracy, precision, recall = stats(tp, fp, tn, fn)

    return confusion, accuracy, precision, recall

#Function for computing accuracy, precision, recall, and confusion matrix
def stats(tp, fp, tn, fn):
    confusion = np.zeros((2,2)) #create 2x2 matrix filled with 0s for confusion matrix
    #Fill confusion matrix
    confusion[0,0] = tp
    confusion[0,1] = fn
    confusion[1,0] = fp
    confusion[1,1] = tn
    
    accuracy = (tp + tn)/(tp + fp + tn + fn)
    precision = (tp)/(tp + fp)
    recall = (tp)/(tp + fn)

    return confusion, accuracy, precision, recall


def main():
    train_set, test_set = load_data(path)
    prior_spam, prior_nspam, spam_mean, nspam_mean, spam_sd, nspam_sd = prior_prob(train_set)
    confusion, accuracy, precision, recall = gaussian_naive_bayes(test_set, prior_spam, prior_nspam, spam_mean, nspam_mean, spam_sd, nspam_sd)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print("Confusion Matrix: ")
    print(confusion)



if __name__ == '__main__':
    main()
