import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_validate
# import classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Linear Discriminant Analysis
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.svm import SVC  # Support Vector Machine
# import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_accuracy(y_pred, y_test):
    return accuracy_score(y_test, y_pred)


def get_class_level_accuracy(y_pred, y_test, num_classes):
    class_accuracy = []
        
    for i in range(num_classes):
        accurate_positives = np.sum((y_test == i+1) & (y_pred == i+1)) 
        total_positives = np.sum(y_test == i+1)
        
        class_accuracy.append(accurate_positives/total_positives)
    
    return class_accuracy


def get_precision_recall(y_pred, y_test, average_setting):
    precision = precision_score(y_test, y_pred, average=average_setting)
    recall = recall_score(y_test, y_pred, average=average_setting)

    return precision, recall


def cv_optimize_RF(n_estimators_list, X_train, y_train, scoring=['f1_macro'], k=10, visualize_cv=False): # currently only for f1_score

    cv_score = []
    for n_estimators in n_estimators_list:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=1234)
        cv_scores = cross_validate(rf, X_train, y_train, cv=k, scoring=scoring, n_jobs=-1)
        cv_score.append(cv_scores['test_f1_macro'])
        
    cv_score_mean = np.mean(cv_score, axis=1)
    cv_score_std = np.std(cv_score, axis=1)

    if visualize_cv:
        plt.figure(figsize=(8, 6)) 
        plt.title('Cross-validation of Random Forest Classifier', fontsize=12)
        plt.plot(n_estimators_list, cv_score_mean, marker='o', label='F1 macro')
        plt.fill_between(n_estimators_list, cv_score_mean - cv_score_std, cv_score_mean + cv_score_std, alpha=0.2)

        plt.xlabel('Number of estimators')
        plt.ylabel('F1 macro score')
        plt.legend()
        plt.show()
    
    best_n_estimators = n_estimators_list[np.argmax(cv_score_mean)]
    
    return best_n_estimators
    
    
def cv_optimize_KNN(n_neighbors_list, X_train, y_train, scoring=['f1_macro'], k=10, visualize_cv=False): # currently only for f1_score
    
    cv_score = []
    for n_neighbors in n_neighbors_list:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        cv_scores = cross_validate(knn, X_train, y_train, cv=k, scoring=scoring, n_jobs=-1)
        cv_score.append(cv_scores['test_f1_macro'])
        
    cv_score_mean = np.mean(cv_score, axis=1)
    cv_score_std = np.std(cv_score, axis=1)

    if visualize_cv:
        plt.figure(figsize=(8, 6)) 
        plt.title('Cross-validation of Random Forest Classifier', fontsize=12)
        plt.plot(n_neighbors_list, cv_score_mean, marker='o', label='F1 macro')
        plt.fill_between(n_neighbors_list, cv_score_mean - cv_score_std, cv_score_mean + cv_score_std, alpha=0.2)

        plt.xlabel('Number of estimators')
        plt.ylabel('F1 macro score')
        plt.legend()
        plt.show()
    
    best_n_neighbors = n_neighbors_list[np.argmax(cv_score_mean)]
    
    return best_n_neighbors    


def evaluate_classifier(classifier, X_train, y_train, X_test, y_test, average_setting, num_classes):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    accuracy = get_accuracy(y_pred, y_test)
    accuracy_classes = get_class_level_accuracy(y_pred, y_test, num_classes=num_classes)
    
    # get the f1 score, both class-wise and average
    f1_score_avg = f1_score(y_test, y_pred, average=average_setting)
    f1_score_classes = f1_score(y_test, y_pred, average=None)
    
    return y_pred, accuracy, accuracy_classes, f1_score_avg, f1_score_classes

    
def save_classifier_results_as_table(path_name, df_results, column_format='l|c|c|c|c|c|c|c|c|c|c'):
    df_results.to_latex(path_name, index=False, bold_rows=True, float_format="%.3f", column_format=column_format)
    
    
def generate_classifier_results(X_train, y_train, X_test, y_test, classifiers=['RF', 'KNN', 'LDA', 'LR', 'NB', 'SVM']
                                , average_setting='macro', visualize_cv=False, print_results=False):
    
    num_classes = len(np.unique(y_train))
    
    # cross-validation
    if 'RF' in classifiers:
        n_estimators_list = np.arange(200, 650, 50)
        best_n_estimators = cv_optimize_RF(n_estimators_list, X_train, y_train, visualize_cv=visualize_cv)
        print("Optimized number of estimators:", best_n_estimators, "\n")

    if 'KNN' in classifiers:
        n_neighbors_list = np.arange(1, 51, 1)
        best_n_neighbors = cv_optimize_KNN(n_neighbors_list, X_train, y_train, visualize_cv=visualize_cv)
        print("Optimized number of neighbors:", best_n_neighbors, "\n")

    classifiers_dict = {
        'RF': RandomForestClassifier(n_estimators=best_n_estimators, random_state=1234, n_jobs=-1) if 'RF' in classifiers else None,
        'KNN': KNeighborsClassifier(n_neighbors=best_n_neighbors, n_jobs=-1) if 'KNN' in classifiers else None,
        'LDA': LinearDiscriminantAnalysis() if 'LDA' in classifiers else None,
        'LR': LogisticRegression(random_state=1234, max_iter=1000, n_jobs=-1) if 'LR' in classifiers else None,
        'NB': GaussianNB() if 'NB' in classifiers else None,
        'SVM': SVC(kernel='linear', random_state=1234) if 'SVM' in classifiers else None,
    }
    classifiers_dict = {k: v for k, v in classifiers_dict.items() if v is not None}

    predictions = []
    accuracy_scores = []
    accuracy_scores_classes = []
    f1_scores = []
    f1_scores_classes = []

    for name, classifier in classifiers_dict.items():
        (y_pred, accuracy, accuracy_classes, 
        f1_score_avg, f1_score_classes) = evaluate_classifier(classifier, X_train, y_train, X_test, y_test, 
                                                              average_setting, num_classes=num_classes)
        predictions.append(y_pred)
        accuracy_scores.append(accuracy)
        accuracy_scores_classes.append(accuracy_classes)
        f1_scores.append(f1_score_avg)
        f1_scores_classes.append(f1_score_classes)
        if print_results:
            print(f"Classifier: {name}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Accuracy classes: {accuracy_classes}")
            print(f"F1 score average: {f1_score_avg:.3f}")
            print(f"F1 score classes: {f1_score_classes}")
            print()
        
    f1_scores_classes = np.array(f1_scores_classes)
    accuracy_scores_classes = np.array(accuracy_scores_classes)
    
    results = {"accuracy_scores": accuracy_scores, "accuracy_scores_classes": accuracy_scores_classes,
               "f1_scores": f1_scores, "f1_scores_classes": f1_scores_classes}
    
    return classifiers_dict, predictions, results


def summarize_classifier_results_multiple_runs(classifier_dict, results_list, return_df=False, visualize=False, path_name=None):
    num_classes = np.shape(results_list[0]['accuracy_scores_classes'])[1]
    for i in range(len(results_list)):
        if i == 0:
            accuracy_scores = np.expand_dims(np.array(results_list[i]['accuracy_scores']), 0)
            accuracy_class_scores = np.expand_dims(np.array(results_list[i]['accuracy_scores_classes']), 0)
            f1_scores = np.expand_dims(np.array(results_list[i]['f1_scores']), 0)
            f1_class_scores = np.expand_dims(np.array(results_list[i]['f1_scores_classes']), 0)
        else:
            accuracy_scores = np.vstack((accuracy_scores, np.expand_dims(np.array(results_list[i]['accuracy_scores']), 0)))
            accuracy_class_scores = np.vstack((accuracy_class_scores, np.expand_dims(np.array(results_list[i]['accuracy_scores_classes']), 0)))
            f1_scores = np.vstack((f1_scores, np.expand_dims(np.array(results_list[i]['f1_scores']), 0)))
            f1_class_scores = np.vstack((f1_class_scores, np.expand_dims(np.array(results_list[i]['f1_scores_classes']), 0)))

    accuracy_scores_mean = np.mean(accuracy_scores, axis=0)
    accuracy_scores_std = np.std(accuracy_scores, axis=0)
    accuracy_class_scores_mean = np.mean(accuracy_class_scores, axis=0)
    accuracy_class_scores_std = np.std(accuracy_class_scores, axis=0)
    f1_scores_mean = np.mean(f1_scores, axis=0)
    f1_scores_std = np.std(f1_scores, axis=0)
    f1_class_scores_mean = np.mean(f1_class_scores, axis=0)
    f1_class_scores_std = np.std(f1_class_scores, axis=0)

    results_means = {'accuracy_scores': accuracy_scores_mean, 'accuracy_scores_classes': accuracy_class_scores_mean,
                    'f1_scores': f1_scores_mean, 'f1_scores_classes': f1_class_scores_mean}
    results_std_devs = {'accuracy_scores': accuracy_scores_std, 'accuracy_scores_classes': accuracy_class_scores_std,
                    'f1_scores': f1_scores_std, 'f1_scores_classes': f1_class_scores_std}	

    data = {'Classifier': list(classifier_dict.keys()),
                                'Accuracy': results_std_devs['accuracy_scores'],
                                'Acc. C1': results_std_devs['accuracy_scores_classes'][:, 0],
                                'Acc. C2': results_std_devs['accuracy_scores_classes'][:, 1],
                                'Acc. C3': results_std_devs['accuracy_scores_classes'][:, 2],
                                'Acc. C4': results_std_devs['accuracy_scores_classes'][:, 3] if num_classes > 3 else None,
                                'Acc. C5': results_std_devs['accuracy_scores_classes'][:, 4] if num_classes > 4 else None,
                                'F1 score': results_std_devs['f1_scores'],
                                'F1 C1': results_std_devs['f1_scores_classes'][:, 0],
                                'F1 C2': results_std_devs['f1_scores_classes'][:, 1],
                                'F1 C3': results_std_devs['f1_scores_classes'][:, 2],
                                'F1 C4': results_std_devs['f1_scores_classes'][:, 3] if num_classes > 3 else None,
                                'F1 C5': results_std_devs['f1_scores_classes'][:, 4] if num_classes > 4 else None}
    df_std_devs = pd.DataFrame({k: v for k, v in data.items() if v is not None})
    
    accuracy_scores = results_means['accuracy_scores']
    accuracy_classes = results_means['accuracy_scores_classes']
    f1_scores = results_means['f1_scores']
    f1_scores_classes = results_means['f1_scores_classes']
    
    data = {'Classifier': list(classifier_dict.keys()), 
                          'Accuracy': accuracy_scores, 
                          'Acc. C1': accuracy_classes[:,0],
                          'Acc. C2': accuracy_classes[:,1],
                          'Acc. C3': accuracy_classes[:,2],
                          'Acc. C4': accuracy_classes[:,3] if num_classes > 3 else None,
                          'Acc. C5': accuracy_classes[:,4] if num_classes > 4 else None, 
                          'F1 score': f1_scores,
                          'F1 C1': f1_scores_classes[:,0],
                          'F1 C2': f1_scores_classes[:,1],
                          'F1 C3': f1_scores_classes[:,2],
                          'F1 C4': f1_scores_classes[:,3] if num_classes > 3 else None,
                          'F1 C5': f1_scores_classes[:,4] if num_classes > 4 else None}
    df_scores = pd.DataFrame({k: v for k, v in data.items() if v is not None})

    value_vars_acc = ['Accuracy', 'Acc. C1', 'Acc. C2', 'Acc. C3', 
                      'Acc. C4' if num_classes > 3 else None,
                      'Acc. C5' if num_classes > 4 else None]
    value_vars_acc = [x for x in value_vars_acc if x is not None]
    df_scores_acc = pd.melt(df_scores, id_vars=['Classifier'], value_vars=value_vars_acc, var_name='Metric', value_name='Score')
    value_vars_f1 = ['F1 score', 'F1 C1', 'F1 C2', 'F1 C3', 
                     'F1 C4' if num_classes > 3 else None,
                     'F1 C5' if num_classes > 4 else None]
    value_vars_f1 = [x for x in value_vars_f1 if x is not None]
    df_scores_f1 = pd.melt(df_scores, id_vars=['Classifier'], value_vars=value_vars_f1, var_name='Metric', value_name='Score')

    if visualize:
        plt.figure(figsize=(18, 8))

        plt.subplot(1, 2, 1)
        colors_classes = ["#0079FF", "#12E6D0", "#7BE600", "#00A336"]
        colors_metrics = ["#E69F00", "#D55E00"]
        palette_acc = sns.color_palette([colors_metrics[0]]+colors_classes)
        sns.barplot(x='Classifier', y='Score', hue='Metric', data=df_scores_acc,
                    palette=palette_acc)
        plt.title('Accuracy scores')
        plt.yticks(np.arange(0,11)/10)
        plt.legend(loc='lower center')

        plt.subplot(1, 2, 2)
        palette_partial = sns.color_palette([colors_metrics[1]]+colors_classes)
        plt.title('F1 scores')
        sns.barplot(x='Classifier', y='Score', hue='Metric', data=df_scores_f1, 
                    palette=palette_partial)
        plt.legend(loc='lower center')
        plt.yticks(np.arange(0,11)/10)
        
        if path_name is not None:
            plt.savefig(path_name, dpi=300, bbox_inches='tight')
        
        plt.show()
    
        return df_scores, df_std_devs if return_df else None