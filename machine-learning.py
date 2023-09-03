import time
import math
import numpy as np
import random as rd
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt



def clear(): #Czyszczenie okna terminala
    print("\x1B\x5B2J", end="")
    print("\x1B\x5BH", end="")

def int_input(): #Zabezpieczony input liczb by przy wprowadzeniu złego znaku nie psuło programu
    while True:
        x = input("-> ")
        try: int(x)
        except ValueError:
            print("To nie jest liczba!")
        else:
            y = int(x)
            return y    

# OPERACJE NA PLIKACH

def choose_file(n):
    if n == 0: # Zestaw treningowy
        return "train_data.data"
    elif n == 1: # Zestaw walidacyjny
        return "dev_data.data"
    elif n == 2: # Zestaw testowy
        return "test_data.data"
    elif n == 3: # Pełen zestaw danych | Plik, który do testów może być nadpisywany
        return "full_data.data"
    elif n == 4: # Pełen zestaw danych | Plik oryginalny, który nie będzie nadpisywany przez program
        return "house-votes-84.data"
    else:
        print('"n" powinno mieścić się w przedziale <0;2>')
        return 0

def read_file(file_name): #Wczytywanie danych z pliku
    file = open(file_name, "r")
    data1d = []
    data2d = []
    j=0
    for linia in file:
        linia = linia.strip("\n")
        linia = linia.split(",")
        for i in range(len(linia)):
            data1d.append(linia[i])
        data2d.append(data1d)
        data1d = [] 
        j += 1
    file.close()
    return data2d

def save_file(data, file_name):
    file = open(file_name, "w")
    for i in range(len(data)):
        for j in range(len(data[i])):
            file.write(data[i][j])
            file.write(",")
        file.write("\n")        
    file.close()

# OPERACJE NA ZAIMPORTOWANYCH ZBIORACH DANYCH

def mix_data(data):
    check = []
    mixed_data = []
    for i in range(len(data)):
        check.append(0)    
    
    for i in range(len(data)):
        while(1):
            los = rd.randint(0,len(data)-1)
            if check[los] == 0:
                check[los] = 1
                mixed_data.insert(los, data[i])
                break
    return mixed_data 

def make_train_dev_test(data, type): #Podział zbioru na train/dev/test
    data_lenght = len(data) - 1

    train = 0.6*data_lenght
    dev = 0.2*data_lenght
    test = 0.2*data_lenght

    if train != int(train) or dev != int(dev) or test != int(test):
        train = int(train)
        dev = int(dev)
        test = int(test)


    
    if type == 0: #ZESTAW TESTOWY
        train_data = []
        for i in range(train + 1):
            train_data.append(data[i])
        return train_data
    elif type == 1:
        dev_data = []
        for i in range(train + 1, train + dev + 1):
            dev_data.append(data[i]) 
        return dev_data
    elif type == 2:
        test_data = []
        for i in range(train + dev + 1, train + dev + test + 1):
            test_data.append(data[i]) 
        return test_data
    else:
        print('"Type" musi mieścić się w przedziale <0;2>')

def show_data(data): #Wyświetlenie danych zawartych w konkretnym zbiorze
    for i in range(len(data)):
        print(f"{i}." + data[i][0] + " " + data[i][1] + " " + data[i][2] + " " + data[i][3] + " " + data[i][4] + " " + data[i][5] + " " + data[i][6] + " " + data[i][7] + " " + data[i][8] + " " + data[i][9] + " " + data[i][10] + " " + data[i][11] + " " + data[i][12] + " " + data[i][13] + " " + data[i][14] + " " + data[i][15] + " " + data[i][16])

# ZLICZANIE GŁOSÓW 0/1

def one_answer(data):
    politician_answers = []
    if data[0] == 'democrat':
        politician_answers.append(1) # 0 - republican, 1 - democrat
    elif data[0] == 'republican':
        politician_answers.append(0)
    for question_number in range(1, 17): # Y - 1,0 | N - 0,1 | ? - 0,0
        if data[question_number] == "y":
            politician_answers.append(1)
            politician_answers.append(0)
        elif data[question_number] == "n":
            politician_answers.append(0)
            politician_answers.append(1)
        elif data[question_number] == "?":
            politician_answers.append(0)
            politician_answers.append(0)
    return politician_answers  

def find_answers(data): #ILE RAZY KAŻDY POLITYK ODPOWIEDZIAŁ Y/N/?
    all_answers = []
    for i in range(len(data)):
        all_answers.append(one_answer(data[i]))
    return all_answers

# ALGORYTM K - NAJBLIŻSZYCH SĄSIADÓW

def distance(points, unkown_politician):
    power = 0
    for i in range(1, len(points)):
        power +=math.pow((points[i] - unkown_politician[i]), 2)
    distance = math.sqrt(power)

    return distance

def find_neighbours(unknown_politician, answers, k):
    distances = []
    neighbours_check = []

    for i in range(len(answers)):
        distances.append(distance(answers[i], unknown_politician))
        neighbours_check.append(0) # TWORZENIE TABLICY ZER DO SPRAWDZANIA SĄSIADÓW
    for j in range(k):
        distance_number = 0
        for i in range(1, len(answers)):
            if distances[i] < distances[distance_number] and neighbours_check[i] == 0:
                distance_number = i
        neighbours_check[distance_number] = 1

    # for i in range(len(distances)):
    #     print(f"{i}. {distances[i]} | n_check = {neighbours_check[i]}")
    return neighbours_check

def show_answers(answers):
    print("[Y|N|0-R 1-D]")
    print(answers)

def knn(train_data, dev_data, k):
    all_answers = find_answers(train_data)
    prediction = []
    
    for i in range(len(dev_data)):
        D = 0
        R = 0
        neighbours_check = find_neighbours(one_answer(dev_data[i]), all_answers, k)
        for j in range(len(neighbours_check)):
            if neighbours_check[j] == 1:
                if all_answers[j][0] == 1:
                    D += 1
                elif all_answers[j][0] == 0:
                    R += 1
                else:
                    print("knn: error")
        
        #print(f"D = {D} R = {R}")
        if D>=R:
            prediction.append(1) # KNN - democrat
        else:
            prediction.append(0) # KNN - republican

    return prediction

# PERCEPTRON

def change_zeros(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == 0:
                data[i][j] = -1
    return data

def compare_weights(w_old, w):
    change = 1
    for i in range(len(w)):
        #print(f'[{i}] || w = {w[i]} || w_old = {w_old[i]}')
        if abs(w[i] - w_old[i]) > 0.0001:
            change = 0
    return change

def perceptron_check(answers, w):
    predict_sign = []   
    for j in range(len(answers)):
        prediction = w[0]
        for i in range(1, len(w)):
            prediction += w[i]*answers[j][i]
        predict_sign.append(np.sign(prediction))
    return predict_sign    

def perceptron_teach(train_answers, w, learning_rate):
    loop = 0
    while(1):
        predict_sign = []
        w_old = []
        for i in range(len(w)):
            w_old.append(w[i])
        incorrect = 0
        for j in range(len(train_answers)):
            prediction = w[0] #BIAS
            for i in range(1, len(w)):
                prediction += w[i]*train_answers[j][i]
            predict_sign.append(np.sign(prediction))
            if predict_sign[j] != train_answers[j][0]:
                w[0] += learning_rate * (train_answers[j][0] - predict_sign[j]) #BIAS
                for i in range(1, len(w)):
                    w[i] += learning_rate * (train_answers[j][0] - predict_sign[j]) * train_answers[j][i]
                incorrect = 1
        if incorrect == 1 and loop < 1000 and compare_weights(w_old, w) == 0:
            loop += 1
        else:
            #print(f"Loop = {loop}")
            break
    return w

def perceptron_init(train_data, check_data):
    w = []
    learning_rate = 0.2 # <0,1>
    train_answers = change_zeros(find_answers(train_data)) # train_answers[0] -> partia
    check_answers = change_zeros(find_answers(check_data))
    for i in range(len(train_answers[1])):
        #w.append(rd.random())
        w.append(0)
    w = perceptron_teach(train_answers, w, learning_rate)
    predict_sign = perceptron_check(check_answers, w)

    for i in range(len(predict_sign)):
        if predict_sign[i] == -1:
            predict_sign[i] = 0
    return predict_sign

# SKLEARN

def sk_load_data():
    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"

    url = "house-votes-84.data"

    names = ['Class', 'handicapped-infants', 'water-project-cost-sharing', 
             'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 
             'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
             'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
             'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']

    votes = pd.read_csv(url, names=names)  
    votes.replace({'n': '0,1','y': '1,0', '?': '0,0'},inplace=True)
    new_votes = pd.DataFrame()

    new_votes['Class'] = votes['Class']
    for x in range(1, len(names)):
        new_votes[[names[x] + ' YES', names[x] + ' NO']] = votes[names[x]].str.split(',', expand=True)
    new_votes.replace({ '0' : 0, '1' : 1},inplace=True)

    return new_votes

def sk_prepare_data(votes, choose):
    votes = sk_load_data()
    X = votes.iloc[:,1:]
    # Klasy partie
    y = votes.select_dtypes(include=[object])   
    # Zamienia nazwy klas na 0,1
    y.Class.unique()
    le = preprocessing.LabelEncoder()
    y = y.apply(le.fit_transform)

    if choose == 0:
        return X
    elif choose == 1:
        return y
    else:
        print("Wrong number!")

def sk_confusion_matrix(clf, X, actual, predicted):
    #MACIERZ POMYŁEK
    fig=ConfusionMatrixDisplay.from_estimator(clf, X, actual, display_labels=["Democrat","Republican"])
    fig.figure_.suptitle("Confusion Matrix")
    plt.show()
    Accuracy = metrics.accuracy_score(actual, predicted)
    Precision = metrics.precision_score(actual, predicted)
    Sensitivity_recall = metrics.recall_score(actual, predicted)
    Specificity = metrics.recall_score(actual, predicted, pos_label=0)
    F1_score = metrics.f1_score(actual, predicted)
    print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score}) 

def sk_MultiLayerPerceptron(X, y):
    # Podział danych |TEST = 60 %|OTHER = 40%| i potem podział other na pół by był 60:20:20
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size = 0.40)
    X_test, X_val, y_test, y_val = train_test_split(X_other, y_other, test_size = 0.50)  

    print("Wybierz zbiór")
    print("1 - Zbiór walidacyjny")
    print("2 - Zbiór testowy")
    y = int_input()
    if y == 1:
        #MULTILAYERPERCEPTRON
        clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_train, y_train)
        pred=clf.predict(X_val)
        print(clf.score(X_val, y_val))
    elif y == 2:
        #MULTILAYERPERCEPTRON
        clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_train, y_train)
        pred=clf.predict(X_test)
        print(clf.score(X_test, y_test))

    sk_confusion_matrix(clf, X_test, y_test, pred)

def sk_kNN(X,y, k):
    # Podział danych |TEST = 60 %|OTHER = 40%| i potem podział other na pół by był 60:20:20
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size = 0.40)
    X_test, X_val, y_test, y_val = train_test_split(X_other, y_other, test_size = 0.50)  

    print("Wybierz zbiór")
    print("1 - Zbiór walidacyjny")
    print("2 - Zbiór testowy")
    y = int_input()
    if y == 1:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_val,y_val)
        pred = knn.predict(X_test)
    elif y == 2:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train,y_train)
        pred = knn.predict(X_test)


    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)

    sk_confusion_matrix(knn, X_test, y_test, pred)

def sk_DecisionTree(X, y):
    # Podział danych |TEST = 60 %|OTHER = 40%| i potem podział other na pół by był 60:20:20
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size = 0.40)
    X_test, X_val, y_test, y_val = train_test_split(X_other, y_other, test_size = 0.50)  

    print("Wybierz zbiór")
    print("1 - Zbiór walidacyjny")
    print("2 - Zbiór testowy")
    y = int_input()
    if y == 1:
        classifier = DecisionTreeClassifier(random_state = 13)
        classifier.fit(X_val, y_val)
        pred = classifier.predict(X_test)
    elif y == 2:
        classifier = DecisionTreeClassifier(random_state = 13)
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)

    sk_confusion_matrix(classifier, X_test, y_test, pred)
    fig = plt.figure(figsize = (10, 7))
    #tree.plot_tree(classifier, feature_names = X.columns, class_names = y.Class.unique(), filled = True)
    tree.plot_tree(classifier)
    plt.show()

def sk_SVC(X, y):
    # Podział danych |TEST = 60 %|OTHER = 40%| i potem podział other na pół by był 60:20:20
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size = 0.40)
    X_test, X_val, y_test, y_val = train_test_split(X_other, y_other, test_size = 0.50)  

    print("Wybierz zbiór")
    print("1 - Zbiór walidacyjny")
    print("2 - Zbiór testowy")
    y = int_input()
    if y == 1:
        svc = SVC(kernel = 'linear', random_state = 0)
        svc.fit(X_val, y_val)
        pred = svc.predict(X_test)
    elif y == 2:
        svc = SVC(kernel = 'linear', random_state = 0)
        svc.fit(X_train, y_train)
        pred = svc.predict(X_test)

    sk_confusion_matrix(svc, X_test, y_test, pred)


# MACIERZ POMYŁEK

def show_and_compare(prediction, dev_data):
    print("")
    for i in range(len(prediction)):
        if prediction[i] == 1:
            print(f"PREDICTED: democrat   | CORRECT: {dev_data[i][0]} -> [{i}]")
        elif prediction[i] == 0:
            print(f"PREDICTED: republican | CORRECT: {dev_data[i][0]} -> [{i}]")

def confusion_matrix(prediction, data):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(prediction)):
        if prediction[i] == 1 and data[i][0] == "democrat":
            TP += 1 # prawdziwie dodatnia
        elif prediction[i] == 1 and data[i][0] == "republican":
            FP += 1 # fałszywie dodatnia
        elif prediction[i] == 0 and data[i][0] == "democrat":
            FN += 1 # fałszywie ujemna
        elif prediction[i] == 0 and data[i][0] == "republican":
            TN += 1 # prawdziwie ujemna

    TPR = TP/(TP+FN) # czułość
    TNR = TN/(FP+TN) # swoistość
    precision = TP/(TP+FP) # precyzja
    ACC = (TP+TN)/(TP+FN+FP+TN) # dokładność
    #ERR = (FP+TN)/(TP+FN+FP+TN) # poziom błędu

    print(f"Czułość TPR = {round(TPR, 4)} | Swoistość TNR = {round(TNR, 4)} | Precyzja = {round(precision*100, 0)}% | Dokładność ACC = {round(ACC*100, 0)}%")
    # print(f"Prawdziwie dodatnia TP = {TP} | Fałszywie dodatnia FP = {FP} | Fałszywie ujemna FN = {FN} | Prawdziwie ujemna TN = {TN}")

#MENU 

def menu_view(data, train_data, dev_data, test_data):
    clear()
    print("1 - Wyświetl cały zbiór danych")
    print("2 - Wyświetl zbiór treningowy")   
    print("3 - Wyświetl zbiór walidacyjny")
    print("4 - Wyświetl zbiór testowy")
    y = int_input()
    if y == 1: # WYŚWIETLENIE CAŁEGO ZBIORU
        clear()
        print("CAŁY ZBIÓR DANYCH")
        show_data(data)
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif y == 2: # WYŚWIETLENIE ZBIORU TRENINGOWEGO
        clear()
        print("ZBIÓR TRENINGOWY")
        show_data(train_data)
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif y == 3: # WYŚWIETLENIE ZBIORU WALIDACYJNEGO
        clear()
        print("ZBIÓR WALIDACYJNY")
        show_data(dev_data)
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif y == 4: # WYŚWIETLENIE ZBIORU TESTOWEGO
        clear()
        print("ZBIÓR TESTOWY")
        show_data(test_data)
        input("Nacisnij Enter by kontynuowac")
        clear()
    else:
        clear()
        print("Nie ma takiej opcji!\n")
        input("Nacisnij Enter by kontynuowac")
        clear()

def menu_knn_choose(train_data, choosen_data):
    clear()
    while True:
        print("1 - Wprowadź k")
        print("2 - Zbadaj działanie algorytmu dla 'k' z przedziału")
        x = int_input()
        if x == 1:
            clear()
            print("Wprowadź k")
            k = int_input()
            if k > 0:
                prediction = knn(train_data, choosen_data, k)
                show_and_compare(prediction, choosen_data)
                confusion_matrix(prediction, choosen_data)
            else:
                print('"k" musi być większe od zera!')
            break
        elif x == 2:
            clear()
            a = int_input()
            b = int_input()
            clear()
            if a > 0 and b > 0 and b > a:
                for k in range(a, b+1):
                    prediction = knn(train_data, choosen_data, k)
                    print(f"K = {k}")
                    confusion_matrix(prediction, choosen_data)           
            else:
                print('"k" musi być większe od zera!')     
            break
        else:
            print("Nie ma takiej opcji!\n")         
            break    

#MAIN

data = read_file(choose_file(4))
train_data = read_file(choose_file(0))
dev_data = read_file(choose_file(1))
test_data = read_file(choose_file(2))

votes = sk_load_data()
X = sk_prepare_data(votes, 0)
y = sk_prepare_data(votes, 1)

while True: #Menu
    print("\n||===========================================||\n||=      ALGORYTMY UCZENIA MASZYNOWEGO      =||\n||===========================================||")
    print("    1 - Wyświetl dane")
    print("    2 - Algorytm KNN")
    print("    3 - Perceptron")
    print("    4 - SKLEARN - MultiLayerPerceptron")
    print("    5 - SKLEARN - KNN")
    print("    6 - SKLEARN - DecisionTree")
    print("    7 - SKLEARN - SVC")
    print("    8 - Podział zbioru na train/dev/test")
    print("    9 - Zapisz pliki z danymi")
    print("    10 - Wymieszaj dane\n")
    print("    11 - Wyjdz z programu\n")
    x = int_input()

    if x == 1: # WYŚWIETLENIE DANYCH
        menu_view(data, train_data, dev_data, test_data)
    elif x == 2: # KNN
        clear()
        print("Wybierz zbiór")
        print("1 - Zbiór walidacyjny")
        print("2 - Zbiór testowy")
        y = int_input()
        clear()
        if y == 1:
            menu_knn_choose(train_data, dev_data)
        elif y == 2:
            menu_knn_choose(train_data, test_data)
        else:
             print("Nie ma takiej opcji!\n")       
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif x == 3: # PERCEPTRON
        clear()
        print("Wybierz zbiór")
        print("1 - Zbiór treningowy")
        print("2 - Zbiór walidacyjny")
        y = int_input()
        clear()
        if y == 1:
            prediction = perceptron_init(train_data, train_data)
            show_and_compare(prediction, train_data)
            confusion_matrix(prediction, train_data)
        elif y == 2:
            prediction = perceptron_init(train_data, dev_data)
            show_and_compare(prediction, dev_data)
            confusion_matrix(prediction, dev_data)
        else:
             print("Nie ma takiej opcji!\n")       
        input("Nacisnij Enter by kontynuowac")
        clear()      
    elif x == 4: # SKLEARN - MULTILAYERPERCEPTRON
        clear()
        sk_MultiLayerPerceptron(X, y)
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif x == 5: # SKLEARN - KNN
        clear()
        while True:
            print("1 - Wprowadź k")
            print("2 - Zbadaj działanie algorytmu dla 'k' z przedziału")
            x = int_input()
            if x == 1:
                clear()
                print("Wprowadź k")
                k = int_input()
                if k > 0:
                    sk_kNN(X, y, k)
                else:
                    print('"k" musi być większe od zera!')
                break
            elif x == 2:
                clear()
                a = int_input()
                b = int_input()
                clear()
                if a > 0 and b > 0 and b > a:
                    for k in range(a, b+1):
                        sk_kNN(X, y, k)
                        print(f"K = {k}")        
                else:
                    print('"k" musi być większe od zera!')     
                break
            else:
                print("Nie ma takiej opcji!\n")         
                break    

        input("Nacisnij Enter by kontynuowac")
        clear()
    elif x == 6: # SKLEARN - DECISIONTREE
        clear()
        sk_DecisionTree(X, y)
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif x == 7: # SKLEARN - SUPPORT VECTOR CLASSIFIER
        clear()
        sk_SVC(X, y)
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif x == 8: # Podział zbioru
        clear()
        train_data = make_train_dev_test(data, 0)
        dev_data = make_train_dev_test(data, 1)
        test_data = make_train_dev_test(data, 2)
        print("Podzielono zbiór w proporcjach 60:20:20")
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif x == 9: # Zapis plików
        save_file(train_data, choose_file(0))
        save_file(dev_data, choose_file(1))
        save_file(test_data, choose_file(2))
        save_file(data, choose_file(3))
        print("Dane zostały zapisane")
        input("Nacisnij Enter by kontynuowac")
        clear()       
    elif x == 10: # Mieszanie danych
        data = mix_data(data)
        print("Wymieszano dane")
        input("Nacisnij Enter by kontynuowac")
        clear()
    elif x == 11: #ZAKOŃCZ PROGRAM
        break
    else:
        clear()
        print("Nie ma takiej opcji!\n")
        input("Nacisnij Enter by kontynuowac")
        clear()