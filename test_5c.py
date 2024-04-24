import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import cv2

p1 = pd.read_csv('Model_1_pred.csv',header=None)
p1 = np.asarray(p1)
p1 = np.delete(p1,0,1)
p1 = np.delete(p1,0,0)
pred1 = np.argmax(p1, axis=1)

p2 = pd.read_csv('Model_2_pred.csv',header=None)
p2 = np.asarray(p2)
p2 = np.delete(p2,0,1)
p2 = np.delete(p2,0,0)
pred2 = np.argmax(p2, axis=1)

p3 = pd.read_csv('Model_3_pred.csv',header=None)
p3 = np.asarray(p3)
p3 = np.delete(p3,0,1)
p3 = np.delete(p3,0,0)
pred3 = np.argmax(p3, axis=1)

p4 = pd.read_csv('Model_4_pred.csv',header=None)
p4 = np.asarray(p4)
p4 = np.delete(p4,0,1)
p4 = np.delete(p4,0,0)
pred4 = np.argmax(p4, axis=1)

p5 = pd.read_csv('Model_5_pred.csv',header=None)
p5 = np.asarray(p5)
p5 = np.delete(p5,0,1)
p5 = np.delete(p5,0,0)
pred5 = np.argmax(p5, axis=1)

labels = pd.read_csv('GT_labels.csv',header=None)
labels = np.asarray(labels)
labels = np.delete(labels,0,1)
labels = np.delete(labels,0,0)

# Assume we have individual model predictions stored in a list called 'pred'
pred = [tuple(pred1), tuple(pred2), tuple(pred3), tuple(pred4), tuple(pred5)]

# Perform ensemble averaging
ensemble_predictions = np.mean([p1, p2, p3, p4, p5], axis=0)
# Convert predictions to class labels
ensemble_labels = np.argmax(ensemble_predictions, axis=1)
print(classification_report(labels, ensemble_labels))

# Calculate the confusion matrix
confusion = confusion_matrix(labels.ravel(), ensemble_labels.ravel())
print(confusion)

