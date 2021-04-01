from sklearn.datasets import load_digits
dataset = load_digits()
X,Y = dataset.data,dataset.target

from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size = 0.2 ,random_state = 0)

from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# X_train_scaled = StandardScaler().fit_transform(X_train)
# x_test_scaled = StandardScaler().fit_transform(x_test)
clf = SVC(gamma = 0.001,C = 1)
clf.fit(X_train,Y_train)
# print('{:.2f}'.format(clf.score(x_test,y_test)))

#Hyperparameter tuning
# from sklearn.model_selection import GridSearchCV
# grid_values = {'gamma':[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
#     ,'C':[1,10,20,30,40,50,60,70,80,90,100]}
# g_clf = GridSearchCV(clf,param_grid = grid_values)
# g_clf.fit(X_train,Y_train)
# print(g_clf.best_params_)
# print(g_clf.best_score_)

from sklearn.metrics import confusion_matrix
import pandas as pd
y_clf_pred = clf.predict(x_test)
df1 = pd.DataFrame(confusion_matrix(y_true = y_test,y_pred = y_clf_pred))

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data = df1,annot = True)
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.title('plot of digit dataset by using svm clf with accuracy :{:.2f}'.format(clf.score(x_test,y_test)))
plt.show()

#User I/P
plt.matshow(dataset.images[int(input('Enter the digit you want to categorize : '))])
plt.show()
print('Thanks for being here..........')
