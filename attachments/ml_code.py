from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split #for splitting dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from scipy.sparse import coo_matrix, hstack
from sklearn.metrics import hinge_loss
import re
from sklearn.metrics import log_loss
from sklearn.linear_model import Perceptron

#load the data
def load_data(filename, X, y):
    file = open("data/" + filename, 'r')
#file = open('SMSSpamCollection.txt', 'r')
    for entries in file:
        X.append(entries.split('	')[1])  # list of X
        y.append(entries.split('	')[0])  # list of y
    file.close()

#count function
def count_function(list):
    count = 0
    for sentence in list:
            count+= 1
    print(count)

def pre_process_function(X):
    my_features = []

    for i in range(len(X)):
        capital_letters = re.compile(r'[A-Z][A-Z]+')
        exclamation_marks = re.compile(r'!!+')
        question_marks = re.compile(r'\?\?+')
        money_sign = re.compile(r'\$|£|€')
        star_sign = re.compile(r'\*')
        percentage_sign = re.compile(r'%')

        b = [capital_letters, exclamation_marks, question_marks, money_sign, star_sign, percentage_sign]
        a = []
        for pattern in b:
            total = 0
            c = pattern.findall(X[i])
            for x in c:
                total += len(x)
            total_length = float(total*100 / len(X[i]))
            if len(c) != 0:
                average_length = float(total*100/ (len(c) * len(X[i])))
                longest_length = float(100*len(max(c, key=len)) / len(X[i]))
            else:
                average_length = 0.0
                longest_length = 0.0

            a.append(total_length)
            a.append(average_length)
            a.append(longest_length)
        my_features.append(a)
        matrix_ = coo_matrix(my_features)
        X[i] = X[i].lower()
        X[i] = RegexpTokenizer(r'\w+').tokenize(X[i])
        for word in X[i]:
            word = WordNetLemmatizer().lemmatize(word, pos="v")
    return matrix_

def binarize(y):
    for i in range(len(y)):
        if y[i] == 'spam':
            y[i] = 1
        else:
            y[i] = -1

def loss_function(predicted_output, real_output):
    count = 0
    for i in range(len(predicted_output)):
        if (real_output[i] - predicted_output[i]) == -2:
            count = count + 10
        elif (real_output[i] - predicted_output[i]) == 2:
            count = count + 1
        else:
            count = count + 0
    len_set = len(predicted_output)
    error = count * 100 / len_set
    return error

#******* main
X = []
y = []
file = 'SMSSpamCollection.txt'
load_data(file, X, y)

#change spam and ham to -1 and 1 respectively
binarize(y)

#split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#tokenize, lemmatize, lower cased, tokens-to-sentence
vect = CountVectorizer(stop_words = 'english', min_df = 17)
features1_matrix = pre_process_function(X_train) #extract features and lower-case, tokenize and lemmatize X_train
#list of tokens into strings as CountVectorizer only takes them as strings
for i in range(len(X_train)):
    X_train[i] = ' '.join(X_train[i])

# A feature vector matrix rows denote number of samples of training set and columns denote words of vocabulary
word_count_vec = vect.fit_transform(X_train)
features_train = (hstack([word_count_vec, features1_matrix])).toarray()
#print(len(features_matrix[0]))
#print(word_count_vec)

                    ############X_test transformation to array##########
features1_matrix_test = pre_process_function(X_test)
for i in range(len(X_test)):
    X_test[i] = ' '.join(X_test[i])

word_count_vec_test = vect.transform(X_test)
features_test = (hstack([word_count_vec_test, features1_matrix_test])).toarray()


                                #############Naive Bayes#########
model1 = MultinomialNB(alpha = 1e-10)
#parameters = [{'alpha': [1e-10, 1e-5, 0.001, 0.1, 1]}]

#model1 = GridSearchCV(mnb, parameters, cv = 10)
model1.fit(features_train, y_train)
#best_accuracy = model1.best_score_
#best_parameters = model1.best_params_
#print(best_accuracy)
#print(best_parameters)

y_train_pred = model1.predict(features_train)
y_test_pred = model1.predict(features_test)

#result party
error_in = loss_function(y_train_pred, y_train)
print("Naive Bayes in sample error is ", error_in, "%")
error_out = loss_function(y_test_pred, y_test)
print("Naive Bayes out of sample error is ", error_out, "%")

                    #######Perceptron Learning Algorithm#######
pla = Perceptron()
parameters = [{'max_iter': [10000]}]

model2 = GridSearchCV(pla, parameters, cv = 10)
model2.fit(features_train, y_train)
#best_accuracy = model1.best_score_
best_parameters = model2.best_params_
#print(best_accuracy)
print(best_parameters)

y_train_pred = model2.predict(features_train)
y_test_pred = model2.predict(features_test)

#result party
error_in = loss_function(y_train_pred, y_train)
print("PLA in sample error is ", error_in, "%")
error_out = loss_function(y_test_pred, y_test)
print("PLA out of sample error is ", error_out, "%")


                #################SVM##########

from sklearn.svm import SVC
from sklearn.svm import SVC, NuSVC, LinearSVC

model4 = LinearSVC(C = 0.1)
#parameters = [{'C': [0.001, 0.01, 0.1, 1, 10]}]
#model4 = GridSearchCV(svm, parameters, cv = 10)
model4.fit(features_train, y_train)
#best_accuracy = model4.best_score_
#best_parameters = model4.best_params_
#print(best_accuracy)
#print(best_parameters)
#print(model3.classes_)

#spam_detect_model4 = model4.fit(features_train,y_train)


y_train_svc_pred = model4.predict(features_train)
y_test_svc_pred = model4.predict(features_test)

error_in4 = loss_function(y_train, y_train_svc_pred)
print("SVM in sample error is ", error_in4*100, "%")
error_out4 = loss_function(y_test, y_test_svc_pred)
print("SVM out of sample error is ", error_out4*100, "%")


            ##########Neural Networks#########

mlp = MLPClassifier(hidden_layer_sizes = (100,))
parameters = [{'alpha': [0.0001]}]
model3 = GridSearchCV(mlp, parameters, cv = 10)
model3.fit(features_train, y_train)
best_accuracy = model3.best_score_
#best_parameters = model3.best_params_
print(best_accuracy)
#print(best_parameters)
#print(model3.classes_)

spam_detect_model_3 = model3.fit(features_train,y_train)

y_train_mlp_pred = model3.predict_proba(features_train)
y_test_mlp_pred = model3.predict_proba(features_test)

error_in3 = log_loss(y_train, y_train_mlp_pred)
print("Neural Network in sample error is ", error_in3*100, "%")
error_out3 = log_loss(y_test, y_test_mlp_pred)
print("Neural Network out of sample error is ", error_out3*100, "%")


