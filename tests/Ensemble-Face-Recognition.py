import pandas as pd
import numpy as np
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

#--------------------------
#Data set

# Ref: https://github.com/serengil/deepface/tree/master/tests/dataset
idendities = {
    "Angelina": ["img1.jpg", "img2.jpg", "img4.jpg", "img5.jpg", "img6.jpg", "img7.jpg", "img10.jpg", "img11.jpg"],
    "Scarlett": ["img8.jpg", "img9.jpg", "img47.jpg", "img48.jpg", "img49.jpg", "img50.jpg", "img51.jpg"],
    "Jennifer": ["img3.jpg", "img12.jpg", "img53.jpg", "img54.jpg", "img55.jpg", "img56.jpg"],
    "Mark": ["img13.jpg", "img14.jpg", "img15.jpg", "img57.jpg", "img58.jpg"],
    "Jack": ["img16.jpg", "img17.jpg", "img59.jpg", "img61.jpg", "img62.jpg"],
    "Elon": ["img18.jpg", "img19.jpg", "img67.jpg"],
    "Jeff": ["img20.jpg", "img21.jpg"],
    "Marissa": ["img22.jpg", "img23.jpg"],
    "Sundar": ["img24.jpg", "img25.jpg"],
    "Katy": ["img26.jpg", "img27.jpg", "img28.jpg", "img42.jpg", "img43.jpg", "img44.jpg", "img45.jpg", "img46.jpg"],
    "Matt": ["img29.jpg", "img30.jpg", "img31.jpg", "img32.jpg", "img33.jpg"],
    "Leonardo": ["img34.jpg", "img35.jpg", "img36.jpg", "img37.jpg"],
    "George": ["img38.jpg", "img39.jpg", "img40.jpg", "img41.jpg"]
}
#--------------------------
#Positives

positives = []

for key, values in idendities.items():

    #print(key)
    for i in range(0, len(values)-1):
        for j in range(i+1, len(values)):
            #print(values[i], " and ", values[j])
            positive = []
            positive.append(values[i])
            positive.append(values[j])
            positives.append(positive)

positives = pd.DataFrame(positives, columns = ["file_x", "file_y"])
positives["decision"] = "Yes"

positives = positives.sample(12)

print(positives.shape)
#--------------------------
#Negatives

samples_list = list(idendities.values())

negatives = []

for i in range(0, len(idendities) - 1):
    for j in range(i+1, len(idendities)):
        #print(samples_list[i], " vs ",samples_list[j])
        cross_product = itertools.product(samples_list[i], samples_list[j])
        cross_product = list(cross_product)
        #print(cross_product)

        for cross_sample in cross_product:
            #print(cross_sample[0], " vs ", cross_sample[1])
            negative = []
            negative.append(cross_sample[0])
            negative.append(cross_sample[1])
            negatives.append(negative)

negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
negatives["decision"] = "No"

negatives = negatives.sample(positives.shape[0])

print(negatives.shape)
#--------------------------
#Merge positive and negative ones

df = pd.concat([positives, negatives]).reset_index(drop = True)

print(df.decision.value_counts())

df.file_x = "dataset/"+df.file_x
df.file_y = "dataset/"+df.file_y

print(df.head())

#--------------------------
#DeepFace

from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace

pretrained_models = {}

pretrained_models["VGG-Face"] = VGGFace.loadModel()
print("VGG-Face loaded")
pretrained_models["Facenet"] = Facenet.loadModel()
print("Facenet loaded")
pretrained_models["OpenFace"] = OpenFace.loadModel()
print("OpenFace loaded")
pretrained_models["DeepFace"] = FbDeepFace.loadModel()
print("FbDeepFace loaded")

instances = df[["file_x", "file_y"]].values.tolist()

models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
metrics = ['cosine', 'euclidean_l2']

if True:
    for model in models:
        for metric in metrics:

            resp_obj = DeepFace.verify(instances
                                       , model_name = model
                                       , model = pretrained_models[model]
                                       , distance_metric = metric
                                       , enforce_detection = False)

            distances = []

            for i in range(0, len(instances)):
                distance = round(resp_obj["pair_%s" % (i+1)]["distance"], 4)
                distances.append(distance)

            df['%s_%s' % (model, metric)] = distances

    df.to_csv("face-recognition-pivot.csv", index = False)
else:
    df = pd.read_csv("face-recognition-pivot.csv")

df_raw = df.copy()

#--------------------------
#Distribution

fig = plt.figure(figsize=(15, 15))

figure_idx = 1
for model in models:
    for metric in metrics:

        feature = '%s_%s' % (model, metric)

        ax1 = fig.add_subplot(len(models) * len(metrics), len(metrics), figure_idx)

        df[df.decision == "Yes"][feature].plot(kind='kde', title = feature, label = 'Yes', legend = True)
        df[df.decision == "No"][feature].plot(kind='kde', title = feature, label = 'No', legend = True)

        figure_idx = figure_idx + 1

plt.show()
#--------------------------
#Pre-processing for modelling

columns = []
for model in models:
    for metric in metrics:
        feature = '%s_%s' % (model, metric)
        columns.append(feature)

columns.append("decision")

df = df[columns]

df.loc[df[df.decision == 'Yes'].index, 'decision'] = 1
df.loc[df[df.decision == 'No'].index, 'decision'] = 0

print(df.head())
#--------------------------
#Train test split

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.30, random_state=17)

target_name = "decision"

y_train = df_train[target_name].values
x_train = df_train.drop(columns=[target_name]).values

y_test = df_test[target_name].values
x_test = df_test.drop(columns=[target_name]).values

#--------------------------
#LightGBM

import lightgbm as lgb

features = df.drop(columns=[target_name]).columns.tolist()
lgb_train = lgb.Dataset(x_train, y_train, feature_name = features)
lgb_test = lgb.Dataset(x_test, y_test, feature_name = features)

params = {
    'task': 'train'
    , 'boosting_type': 'gbdt'
    , 'objective': 'multiclass'
    , 'num_class': 2
    , 'metric': 'multi_logloss'
}

gbm = lgb.train(params, lgb_train, num_boost_round=250, early_stopping_rounds = 15 , valid_sets=lgb_test)

gbm.save_model("face-recognition-ensemble-model.txt")

#--------------------------
#Evaluation

predictions = gbm.predict(x_test)

prediction_classes = []
for prediction in predictions:
    prediction_class = np.argmax(prediction)
    prediction_classes.append(prediction_class)

cm = confusion_matrix(y_test, prediction_classes)
print(cm)

tn, fp, fn, tp = cm.ravel()

recall = tp / (tp + fn)
precision = tp / (tp + fp)
accuracy = (tp + tn)/(tn + fp +  fn + tp)
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision: ", 100*precision,"%")
print("Recall: ", 100*recall,"%")
print("F1 score ",100*f1, "%")
print("Accuracy: ", 100*accuracy,"%")
#--------------------------
#Interpretability

ax = lgb.plot_importance(gbm, max_num_features=20)
plt.show()

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

plt.rcParams["figure.figsize"] = [20, 20]

for i in range(0, gbm.num_trees()):
    ax = lgb.plot_tree(gbm, tree_index = i)
    plt.show()

    if i == 2:
        break
#--------------------------
#ROC Curve

y_pred_proba = predictions[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(7,3))
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

#--------------------------
