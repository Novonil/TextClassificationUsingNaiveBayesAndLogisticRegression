import os
import io
import re
import pandas as pd
import numpy as np
import argparse
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV


def unique_words(ham_list, spam_list):
    unique_word_set = set()
    for i in ham_list:
        unique_word_set = unique_word_set.union(i)
    for k in spam_list:
        unique_word_set = unique_word_set.union(k)
    return unique_word_set


def list_words(path):
    word_list = []
    for files in os.walk(path):
        for fi in files:
            list_of_files = fi
        # print(list_of_files)
        #print(len(list_of_files))
    for file_name in list_of_files:
        # print(file_name)
        temp_list = []
        reader = io.open(path + file_name, 'rb')
        data = reader.read()
        temp_list.append(str(data).split())
        word_list_temp = re.findall(r'[A-Za-z]+', str(temp_list))
        word_list.append(word_list_temp)
        #unique_words = refined_word_list
    #print(refined_word_list)
    #print(len(refined_word_list))
    return word_list

def Bag_Of_Words(ham_list, spam_list, unique_set):
    ham_dic_list = []
    spam_dic_list = []
    for lst in ham_list:
        dic_intermediate = dict(zip(list(unique_set), [0]*len(unique_set)))
        for w in lst:
            if w in dic_intermediate.keys():
                dic_intermediate[w] = dic_intermediate[w] + 1
        ham_dic_list.append(dic_intermediate)
    for lst in spam_list:
        dic_intermediate = dict(zip(list(unique_set), [0]*len(unique_set)))
        for w in lst:
            if w in dic_intermediate.keys():
                dic_intermediate[w] = dic_intermediate[w] + 1
        spam_dic_list.append(dic_intermediate)

    ham_matrix = pd.DataFrame(ham_dic_list)
    spam_matrix = pd.DataFrame(spam_dic_list)
    ham_matrix["Y"] = [1 for i in range(len(ham_matrix.index))]
    spam_matrix["Y"] = [0 for i in range(len(spam_matrix.index))]
    ham_spam_matrix = pd.concat([ham_matrix, spam_matrix], ignore_index=True)
    return ham_spam_matrix

def Bernoulli_Model(ham_list, spam_list, unique_set):
    ham_dic_list = []
    spam_dic_list = []
    for lst in ham_list:
        dic_intermediate = dict(zip(list(unique_set), [0] * len(unique_set)))
        for w in lst:
            if w in dic_intermediate.keys() and dic_intermediate[w] != 1:
                dic_intermediate[w] = dic_intermediate[w] + 1
        ham_dic_list.append(dic_intermediate)
    for lst in spam_list:
        dic_intermediate = dict(zip(list(unique_set), [0] * len(unique_set)))
        for w in lst:
            if w in dic_intermediate.keys() and dic_intermediate[w] != 1:
                dic_intermediate[w] = dic_intermediate[w] + 1
        spam_dic_list.append(dic_intermediate)
    ham_matrix = pd.DataFrame(ham_dic_list)
    spam_matrix = pd.DataFrame(spam_dic_list)
    ham_matrix["Y"] = [1 for i in range(len(ham_matrix.index))]
    spam_matrix["Y"] = [0 for i in range(len(spam_matrix.index))]
    hs_df = pd.concat([ham_matrix, spam_matrix], ignore_index=True)
    return hs_df

def Bag_Of_Words_Trainer(bg_of_wrds_list):
    ham_spam_prob = [0, 0]
    prob_ham = np.true_divide(bg_of_wrds_list["Y"].sum(), len(bg_of_wrds_list.index))
    prob_spam = 1 - prob_ham
    ham_spam_prob[0] = np.log2(prob_spam)
    ham_spam_prob[1] = np.log2(prob_ham)
    dic_ham_prob = dict(zip(list(bg_of_wrds_list[:-1]), [0]*len(bg_of_wrds_list[:-1])))
    dic_spam_prob = dict(zip(list(bg_of_wrds_list[:-1]), [0]*len(bg_of_wrds_list[:-1])))
    ham_probability = np.zeros(len(bg_of_wrds_list.columns[:-1]))
    spam_probability = np.zeros(len(bg_of_wrds_list.columns[:-1]))
    temp_ham = 0
    temp_spam = 0
    for i, j in enumerate(bg_of_wrds_list.columns[:-1]):
        x = bg_of_wrds_list[j][bg_of_wrds_list["Y"] == 1].sum() + 1
        y = bg_of_wrds_list[j][bg_of_wrds_list["Y"] == 0].sum() + 1
        spam_probability[i] = y
        ham_probability[i] = x
        temp_ham = temp_ham + x
        temp_spam = temp_spam + y
    ham_probability = np.log2(ham_probability)
    spam_probability = np.log2(spam_probability)
    temp_spam = np.log2(temp_spam)
    temp_ham = np.log2(temp_ham)
    ham_probability = ham_probability - temp_ham
    spam_probability = spam_probability - temp_spam
    for i, j in enumerate(bg_of_wrds_list.columns[:-1]):
        dic_ham_prob[j] = ham_probability[i]
        dic_spam_prob[j] = spam_probability[i]
    return dic_ham_prob, dic_spam_prob, ham_spam_prob

def Bernoulli_dist_train(ham_spam_Bernoulli_model):
    ham_spam_prob = [0, 0]
    prob_ham = np.true_divide(ham_spam_Bernoulli_model["Y"].sum(), len(ham_spam_Bernoulli_model.index))
    prob_spam = 1 - prob_ham
    ham_spam_prob[0] = np.log2(prob_spam)
    ham_spam_prob[1] = np.log2(prob_ham)
    dic_ham_prob = dict(zip(list(ham_spam_Bernoulli_model.columns[:-1]), [0] * len(ham_spam_Bernoulli_model.columns[:-1])))
    dic_spam_prob = dict(zip(list(ham_spam_Bernoulli_model.columns[:-1]), [0] * len(ham_spam_Bernoulli_model.columns[:-1])))
    ham_probability = np.zeros(len(ham_spam_Bernoulli_model.columns[:-1]))
    spam_probability = np.zeros(len(ham_spam_Bernoulli_model.columns[:-1]))
    test_ham = 0
    test_spam = 0
    for i, j in enumerate(ham_spam_Bernoulli_model.columns[:-1]):
        y = ham_spam_Bernoulli_model[j][ham_spam_Bernoulli_model["Y"] == 0].sum() + 1
        x = ham_spam_Bernoulli_model[j][ham_spam_Bernoulli_model["Y"] == 1].sum() + 1
        ham_probability[i] = x
        spam_probability[i] = y
        # dsum_h = dsum_h + (ber_df[col][ber_df["Y"] == 1].sum() + 1)
        # dsum_s = dsum_s + (ber_df[col][ber_df["Y"] == 0].sum() + 1)
    test_ham = np.log2(len(ham_spam_Bernoulli_model[ham_spam_Bernoulli_model["Y"] == 1].index) + 2)
    test_spam = np.log2(len(ham_spam_Bernoulli_model[ham_spam_Bernoulli_model["Y"] == 0].index) + 2)
    ham_probability = np.log2(ham_probability)
    spam_probability = np.log2(spam_probability)
    ham_probability = ham_probability - test_ham
    spam_probability = spam_probability - test_spam
    for i, j in enumerate(list(ham_spam_Bernoulli_model.columns[:-1])):
        dic_ham_prob[j] = ham_probability[i]
        dic_spam_prob[j] = spam_probability[i]
    return dic_ham_prob, dic_spam_prob, ham_spam_prob


def Bag_of_Words_predictor(dic_ham_prob, dic_spam_prob, ham_spam_prob, ham_test, spam_test, ham_spam_bag_of_word):
    #check conditional probability of the examples given Y  = 1
    True_Class = [1 for i in range(len(ham_test))]
    True_Class = True_Class + [0 for i in range(len(spam_test))]
    True_Positives = 0
    True_Negatives = 0
    False_Positives = 0
    False_Negatives = 0
    Predicted_Class = []
    for doc in ham_test + spam_test:
        spam_rate = ham_spam_prob[0]
        ham_rate = ham_spam_prob[1]
        for word in doc:
            if word in dic_spam_prob:
                spam_rate = spam_rate + dic_spam_prob[word]
            if word in dic_ham_prob:
                ham_rate = ham_rate + dic_ham_prob[word]
        if ham_rate > spam_rate:
            Predicted_Class.append(1)
        else:
            Predicted_Class.append(0)
    for i in range(len(Predicted_Class)):
        if True_Class[i] == 1 and Predicted_Class[i] == 1:
            True_Positives = True_Positives + 1
        elif True_Class[i] == 0 and Predicted_Class[i] == 1:
            False_Positives = False_Positives + 1
        elif True_Class[i] == 0 and Predicted_Class[i] == 0:
            True_Negatives = True_Negatives + 1
        else:
            False_Negatives = False_Negatives + 1
    match_result = np.array(True_Class) == np.array(Predicted_Class)
    accuracy_BOW = np.true_divide(match_result.sum(), len(match_result))
    precision = np.true_divide(True_Positives, True_Positives + False_Positives)
    recall = np.true_divide(True_Positives, True_Positives + False_Negatives)
    F1_Score = 2 * (recall * precision) / (recall + precision)
    return (accuracy_BOW, precision, recall, F1_Score)


def Bernoulli_Predictor(dic_ham_prob, dic_spam_prob, ham_spam_prob, ham_test, spam_test, ham_spam_bag_of_word):
    #check conditional probability of the examples given Y  = 1
    True_Class = [1 for i in range(len(ham_test))]
    True_Class = True_Class + [0 for i in range(len(spam_test))]
    True_Positives = 1
    True_Negatives = 1
    False_Positives = 1
    False_Negatives = 1
    Predicted_Class = []
    for i in ham_test + spam_test:
        ham_rate = ham_spam_prob[1]
        spam_rate = ham_spam_prob[0]
        for j in ham_spam_bag_of_word.columns[:-1]:
            if j in i:
                spam_rate = spam_rate + dic_spam_prob[j]
                ham_rate = ham_rate + dic_ham_prob[j]
            else:
                ham_rate = ham_rate + (1 - dic_ham_prob[j])
                spam_rate = spam_rate + (1 - dic_spam_prob[j])
        if spam_rate < ham_rate:
            Predicted_Class.append(1)
        else:
            Predicted_Class.append(0)
    match_result = np.array(True_Class) == np.array(Predicted_Class)
    accuracy_bernoulli = np.true_divide(match_result.sum(), len(match_result))
    for i in range(len(Predicted_Class)):
        if True_Class[i] == 1 and Predicted_Class[i] == 1:
            True_Positives = True_Positives + 1
        elif True_Class[i] == 0 and Predicted_Class[i] == 1:
            False_Positives = False_Positives + 1
        elif True_Class[i] == 0 and Predicted_Class[i] == 0:
            True_Negatives = True_Negatives + 1
        else:
            False_Negatives = False_Negatives + 1
    precision = np.true_divide(True_Positives, True_Positives + False_Positives)
    recall = np.true_divide(True_Positives, True_Positives + False_Negatives)
    F1_Score = 2 * (recall * precision)/(recall + precision)
    return (accuracy_bernoulli, precision, recall, F1_Score)

def SGD_Classifier(ham_spam_bag_of_word, ham_spam_bag_of_word_test):
    X_train = np.array(ham_spam_bag_of_word.iloc[:, :-1])
    Y_train = np.array(ham_spam_bag_of_word.iloc[:, -1])
    X_test = np.array(ham_spam_bag_of_word_test.iloc[:, :-1])
    Y_test = np.array(ham_spam_bag_of_word_test["Y"])
    ep_values = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    jobs_val = np.array([-1, -2, -3])
    Epoch_SGD = np.array([500])
    True_Positives = 1
    True_Negatives = 1
    False_Positives = 1
    False_Negatives = 1

    SGD_class = lm.SGDClassifier()
    grid = GridSearchCV(estimator=SGD_class, param_grid=dict(alpha=ep_values, n_jobs=jobs_val, max_iter=Epoch_SGD))
    grid.fit(X_train, Y_train)

    Predicted_Class = grid.predict(X_test)
    match = Y_test == Predicted_Class
    Accuracy_Pred = np.true_divide(match.sum(), len(match))
    for i in range(len(Predicted_Class)):
        if Y_test[i] == 1 and Predicted_Class[i] == 1:
            True_Positives = True_Positives + 1
        elif Y_test[i] == 0 and Predicted_Class[i] == 1:
            False_Positives = False_Positives + 1
        elif Y_test[i] == 0 and Predicted_Class[i] == 0:
            True_Negatives = True_Negatives + 1
        else:
            False_Negatives = False_Negatives + 1
    precision = np.true_divide(True_Positives, True_Positives + False_Positives)
    recall = np.true_divide(True_Positives, True_Positives + False_Negatives)
    F1_Score = 2 * (recall * precision) / (recall + precision)
    return (Accuracy_Pred, recall, precision, F1_Score)


def Log_Reg_Trainer(ham_spam_bag_of_word, Lambda, Eta):
    Weight_0 = 0
    Weight = np.zeros(len(ham_spam_bag_of_word.columns[:-1]))
    for i in range(500):
        gradient_weight = np.zeros(len(ham_spam_bag_of_word.columns[:-1]))
        gradient_weight_0 = 0
        for k, j in ham_spam_bag_of_word.iloc[:, :-1].iterrows():
            gradient_weight = gradient_weight + np.array(j) * (ham_spam_bag_of_word["Y"][k] - np.true_divide(1, 1 + np.exp(Weight_0 + (Weight * np.array(j)).sum()))) - Lambda * Weight
            gradient_weight_0 = gradient_weight_0 + (ham_spam_bag_of_word["Y"][k] - np.true_divide(1, 1 + np.exp(Weight_0 + (Weight * np.array(j)).sum())))
        Weight = Weight + Eta * gradient_weight
        Weight_0 = Weight_0 + Eta * gradient_weight_0
    return Weight, Weight_0


def Logistic_Reg_Predictor(Weight, Weight_0, ham_spam_test, ham_spam_train):
    test_cols = list(ham_spam_test.columns)
    for col in test_cols:
        if col not in ham_spam_train.columns:
            ham_spam_test = ham_spam_test.drop([col], axis=1)
    Predicted_Y = np.zeros(len(ham_spam_test.iloc[:, :-1].index))
    colms = []
    indexes = []
    for i, j in enumerate(ham_spam_train.columns):
        if j not in ham_spam_test.columns:
            indexes.append(i)
            colms.append(j)
    ham_spam_train = ham_spam_train.drop(colms, axis=1)
    Weight = np.delete(Weight, indexes)
    ham_spam_test[ham_spam_train.columns]
    for i, row in ham_spam_test.iloc[:, :-1].iterrows():
        Predicted_Y[i] = np.true_divide(1, 1 + np.exp(Weight_0 + (Weight * np.array(row)).sum()))
    Y_df = pd.DataFrame(zip(np.array(ham_spam_test["Y"]), np.array(Predicted_Y)), columns=["Y", "pred_Y"])
    True_Positives = Y_df["pred_Y"][Y_df["Y"] == 1]
    True_Positives = True_Positives.values.sum()
    True_Negatives = Y_df["pred_Y"][Y_df["Y"] == 0]
    True_Negatives = len(True_Negatives.index) - True_Negatives.values.sum()
    False_Positives = Y_df["pred_Y"][Y_df["Y"] == 0]
    False_Positives = False_Positives.values.sum()
    False_Negatives = Y_df["pred_Y"][Y_df["Y"] == 1]
    False_Negatives = len(False_Negatives.index) - False_Negatives.values.sum()
    accuracy = np.true_divide(True_Positives + True_Negatives, True_Positives + False_Positives + False_Negatives + True_Negatives)
    precision = np.true_divide(True_Positives, True_Positives + False_Positives)
    recall = np.true_divide(True_Positives, True_Positives + False_Negatives)
    F1_score = 2 * (recall * precision) / (recall + precision)
    return [accuracy, precision, recall, F1_score]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm_number', '--algo_no', type=int)
    parser.add_argument('-train_data_ham', '--train_set_ham', type=str)
    parser.add_argument('-train_data_spam', '--train_set_spam', type=str)
    parser.add_argument('-test_data_ham', '--test_set_ham', type=str)
    parser.add_argument('-test_data_spam', '--test_set_spam', type=str)

    arg = parser.parse_args()
    training_set_ham = arg.train_set_ham
    training_set_spam = arg.train_set_spam
    testing_set_ham = arg.test_set_ham
    testing_set_spam = arg.test_set_spam

    # training_set_ham = "enron4_train/ham/"
    # training_set_spam = "enron4_train/spam/"
    # testing_set_ham = "enron4_test/ham/"
    # testing_set_spam = "enron4_test/spam/"

    ham_list = list_words(training_set_ham)
    spam_list = list_words(training_set_spam)
    ham_test = list_words(testing_set_ham)
    spam_test = list_words(testing_set_spam)
    unique_set = unique_words(ham_list, spam_list)
    unique_test = unique_words(ham_test, spam_test)
    ham_spam_bag_of_word = Bag_Of_Words(ham_list, spam_list, unique_set)
    ham_spam_bag_of_word_test = Bag_Of_Words(ham_test, spam_test, unique_test)

    #Naive Bayes with BOW
    print("Executing Naive Bayes with Bag of Word Model")
    ham_spam_bag_of_word = Bag_Of_Words(ham_list, spam_list, unique_set)
    ham_spam_bag_of_word_test = Bag_Of_Words(ham_test, spam_test, unique_test)
    dic_ham_prob, dic_spam_prob, ham_spam_prob = Bag_Of_Words_Trainer(ham_spam_bag_of_word)
    Accuracy_BOW, precision, recall, F1_score = Bag_of_Words_predictor(dic_ham_prob, dic_spam_prob, ham_spam_prob,
                                                                       ham_test, spam_test, ham_spam_bag_of_word_test)
    print("Bag_Of_Words_Accuracy using Naive bayes")
    print(Accuracy_BOW)
    print("Recall")
    print(recall)
    print("Precision")
    print(precision)
    print("F1_score ")
    print(F1_score)

    #Naive Bayes with Bernoulli
    print("Executing Naive Bayes with Bernoulli Model")
    ham_spam_Bernoulli_model = Bernoulli_Model(ham_list, spam_list, unique_set)
    dic_ham_bernoulli, dic_spam_bernoulli, prob_ham_spam = Bernoulli_dist_train(ham_spam_Bernoulli_model)
    Accuracy_Bernoulli, recall, precision, F1_score = Bernoulli_Predictor(dic_ham_bernoulli, dic_spam_bernoulli, prob_ham_spam, ham_test, spam_test, ham_spam_Bernoulli_model)
    print("Bernoulli_Accuracy using Naive Bayes")
    print(Accuracy_Bernoulli)
    print("Recall")
    print(recall)
    print("Precision")
    print(precision)
    print("F1_score ")
    print(F1_score)

    #SGD_Classifier with BOW
    print("SGD_Classifier using Bag of Words")
    Acc_SGD, recall, precision , F1_score = SGD_Classifier(ham_spam_bag_of_word, ham_spam_bag_of_word)
    print("SGD_Classifier_Accuracy_BOW")
    print(Acc_SGD)
    print("Recall")
    print(recall)
    print("Precision")
    print(precision)
    print("F1_score ")
    print(F1_score)

    #SGD Classifier with Bernoulli
    print("SGD_Classifier using Bernoulli Model")
    ham_spam_bag_of_word = Bernoulli_Model(ham_list, spam_list, unique_set)
    ham_spam_bag_of_word_test = Bernoulli_Model(ham_test, spam_test, unique_test)
    Accuracy_SGD, recall, precision , F1_score = SGD_Classifier(ham_spam_bag_of_word, ham_spam_bag_of_word)
    print("SGD_Classifier_Accuracy_Bernoulli")
    print(Accuracy_SGD)
    print("Recall")
    print(recall)
    print("Precision")
    print(precision)
    print("F1_score ")
    print(F1_score)

    #LR with BOW
    print("Logistic Regression using Bag Of Words Model")
    Lambda = 0.05
    Eta = 0.1
    ham_spam_bag_of_word = Bag_Of_Words(ham_list, spam_list, unique_set)
    ham_spam_bag_of_word_test = Bag_Of_Words(ham_test, spam_test, unique_test)
    Weight, Weight_0 = Log_Reg_Trainer(ham_spam_bag_of_word, Lambda, Eta)
    Accuracy_Logistic, Precision, Recall, F1_Score = Logistic_Reg_Predictor(Weight, Weight_0, ham_spam_bag_of_word_test,ham_spam_bag_of_word)
    print("Logistic Accuracy")
    print(Accuracy_Logistic)
    print("Precision")
    print(Precision)
    print("Recall")
    print(Recall)
    print("F1_Score")
    print(F1_Score)

    # LR with Bernoulli
    print("Logistic Regression using Bernoulli Model")
    Lambda = 0.05
    Eta = 0.1
    ham_spam_bernoulli = Bernoulli_Model(ham_list, spam_list, unique_set)
    ham_spam_bernoulli_test = Bernoulli_Model(ham_test, spam_test, unique_test)
    Weight, Weight_0 = Log_Reg_Trainer(ham_spam_bag_of_word, Lambda, Eta)
    Accuracy_Logistic, Precision, Recall, F1_Score = Logistic_Reg_Predictor(Weight, Weight_0, ham_spam_bernoulli_test, ham_spam_bernoulli)
    print("Logistic Accuracy")
    print(Accuracy_Logistic)
    print("Precision")
    print(Precision)
    print("Recall")
    print(Recall)
    print("F1_Score")
    print(F1_Score)

main()