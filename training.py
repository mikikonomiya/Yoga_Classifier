import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

dataset= pd.read_csv("dataset_images.csv")

#here i am seperating the features and the label , which is the yoga pose name
X= dataset.drop("Label", axis=1)
y= dataset["Label"]

X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2 )

model= KNeighborsClassifier(n_neighbors=3)

model.fit(X_train,y_train)

prediction_pose=model.predict(X_test)

accuracy=accuracy_score(y_test,prediction_pose)
print(accuracy)



print(cross_val_score(model, X,y))
joblib.dump(model, "mymodel.pkl")


def conf_matrix(y_test, prediction_pose, class_names=None, figsize=(8,6)):
    matrix = confusion_matrix(y_test, prediction_pose)
    
    matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    matrix_percent = (matrix_normalized * 100).round().astype(int)

    pastel_cmap = LinearSegmentedColormap.from_list( "pastel", ["#AEDFF7", "#FAD1E6"])

    plt.figure(figsize=figsize)
    ax = sns.heatmap(matrix_normalized, annot=matrix_percent, fmt="d", cmap=pastel_cmap, square=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='white', cbar=True)
    ax.set_xticklabels(ax.get_xticklabels(), fontname="Times New Roman")
    ax.set_yticklabels(ax.get_yticklabels(), fontname="Times New Roman")
    
    plt.title("Confusion Matrix (%)", fontsize = 17, fontname="Times New Roman")
    plt.xlabel("Predicted Pose", fontsize = 12, fontname="Times New Roman")
    plt.ylabel("True Pose", fontsize = 12, fontname="Times New Roman")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

class_names = ["Butterfly", "Dancer (L)", "Dancer (R)", "Downward dog", "Goddess", "Half Moon (L)", "Half Moon (R)", "Tree (L)", "Tree (R)", "Triangle", "Warrior (L)", "Warrior (R)"]
conf_matrix(y_test,prediction_pose, class_names)
