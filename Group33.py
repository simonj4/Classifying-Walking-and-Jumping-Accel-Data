from tkinter import Tk, Label, Button, filedialog
from tkinter.font import Font
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score,
                             roc_auc_score, roc_curve, RocCurveDisplay, f1_score, precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def reduce_noise_and_normalize(input_csv, output_csv):
    window_size = 5
    data = pd.read_csv(input_csv)
    timestamps = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    filtered_data = data.rolling(window=window_size, min_periods=1).mean()
    sc = preprocessing.StandardScaler()
    normalized_data = sc.fit_transform(filtered_data)
    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

    final_df = pd.concat([timestamps, normalized_df], axis=1)
    final_df = final_df.fillna(final_df.mean())
    final_df.to_csv(output_csv, index=False)


reduce_noise_and_normalize("ErikWalk.csv", "ErikWalk2.csv")
reduce_noise_and_normalize("ErikJump.csv", "ErikJump2.csv")
reduce_noise_and_normalize("SimonJump.csv", "SimonJump2.csv")
reduce_noise_and_normalize("SimonWalk.csv", "SimonWalk2.csv")
reduce_noise_and_normalize("EmreWalk.csv", "EmreWalk2.csv")
reduce_noise_and_normalize("EmreJump.csv", "EmreJump2.csv")


def labelCSV(inPath, outPath, label):  # turn raw data into labled data : saved as csv
    # assigns labels to all rows of a cvs file
    # give function the file path (String) of a cvs file, the path (String) of where you want the output, and the label (String) to be assigned to all rows

    df = pd.read_csv(inPath)  # read
    df['label'] = label  # label
    df.to_csv(outPath, index=False)


# Emre
labelCSV('EmreWalk2.csv', 'EmreLabeledWalk.csv', '0')
labelCSV('EmreJump2.csv', 'EmreLabeledJump.csv', '1')

# Erik
labelCSV('ErikWalk2.csv', 'ErikLabeledWalk.csv', '0')
labelCSV('ErikJump2.csv', 'ErikLabeledJump.csv', '1')

# Simon
labelCSV('SimonWalk2.csv', 'SimonLabeledWalk.csv', '0')
labelCSV('SimonJump2.csv', 'SimonLabeledJump.csv', '1')


def shuffleData(inPath, outPath):  # shuffle labled data and save as csv

    df = pd.read_csv(inPath)

    df['window'] = df['Time (s)'] // 5  # get number of 5 second intervals in data

    windowsArray = df['window'].unique()  # array of different number of values in window column^

    np.random.shuffle(windowsArray)  # shuffle windows

    windowMapping = dict(
        zip(windowsArray, range(len(windowsArray))))  # map shuffled array to new index and store in dictionary

    df['window'] = df['window'].map(windowMapping)  # replace unmapped with mapped index

    # sort by index and time, axis 0 to sort along rows in ascending order, inplace to change current df rather than returning a copy
    df.sort_values(['window', 'Time (s)'], axis=0, ascending=[True, True], inplace=True)

    df = df.iloc[:, :-1]  # remove window column

    df.to_csv(outPath, index=False)


def concat(data1, data2, data3, data4, data5, data6):  # concatenates 3 dataframes to one

    df1 = pd.read_csv(data1)
    df2 = pd.read_csv(data2)
    df3 = pd.read_csv(data3)
    df4 = pd.read_csv(data4)
    df5 = pd.read_csv(data5)
    df6 = pd.read_csv(data6)

    bigdf = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)  # concatenate rows

    bigdf.to_csv('Big.csv', index=False)
    bigdf.dropna()


concat('EmreLabeledWalk.csv', 'ErikLabeledWalk.csv', 'SimonLabeledWalk.csv', 'SimonLabeledJump.csv',
       'ErikLabeledJump.csv', 'EmreLabeledJump.csv')
shuffleData('Big.csv', 'BigShuffled.csv')

from sklearn.model_selection import train_test_split


def splitCSV(inPath, outPathTrain, outPathTest, randomSeed):  # split data set, 90% to Train, 10 to Test
    df = pd.read_csv(inPath)
    # split input
    Traindf, Testdf = train_test_split(df, test_size=0.1, random_state=randomSeed, shuffle=True)
    Traindf.to_csv(outPathTrain, index=False)
    Testdf.to_csv(outPathTest, index=False)


splitCSV('BigShuffled.csv', 'Train.csv', 'Test.csv', 4)

labledWalking1 = 'EmreLabeledWalk.csv'
labledWalking2 = 'ErikLabeledWalk.csv'
labledWalking3 = 'SimonLabeledWalk.csv'
labledJumping1 = 'EmreLabeledJump.csv'
labledJumping2 = 'ErikLabeledJump.csv'
labledJumping3 = 'SimonLabeledJump.csv'

w1df = pd.read_csv(labledWalking1)
w2df = pd.read_csv(labledWalking2)
w3df = pd.read_csv(labledWalking3)
j1df = pd.read_csv(labledJumping1)
j2df = pd.read_csv(labledJumping2)
j3df = pd.read_csv(labledJumping3)

Train = 'Train.csv'
Traindf = pd.read_csv(Train)

Test = 'Test.csv'
Testdf = pd.read_csv(Test)

with h5py.File('walkOrJump.h5', 'w') as hdf:
    G1 = hdf.create_group('/emre')
    G1.create_dataset('walking', data=w1df)
    G1.create_dataset('jumping', data=j1df)

    G2 = hdf.create_group('/erik')
    G2.create_dataset('walking', data=w2df)
    G2.create_dataset('jumping', data=j2df)

    G3 = hdf.create_group('/simon')
    G3.create_dataset('walking', data=w3df)
    G3.create_dataset('jumping', data=j3df)

    G10 = hdf.create_group('/data/train')
    G10.create_dataset('Train', data=Traindf)

    G11 = hdf.create_group('/data/test')
    G11.create_dataset('Test', data=Testdf)


# Visualization

# Acceleration Vs Time graph
def plot_accVstime(csv_file_path, activity, person):
    file = pd.read_csv(csv_file_path)
    plt.figure(figsize=(10, 6))
    plt.plot(file.iloc[:, 0], file.iloc[:, 1], label='X-axis')
    plt.plot(file.iloc[:, 0], file.iloc[:, 2], label='Y-axis')
    plt.plot(file.iloc[:, 0], file.iloc[:, 3], label='Z-axis')
    plt.title(f'{person} - {activity} Acceleration vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.show()


# Acceleration Distribution Histograms
def plot_histograms(csv_file_path, activity, person):
    data = pd.read_csv(csv_file_path)

    # Creating a figure with 3 subplots, one for each axis
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Acceleration Distribution for {person} - {activity}')

    ax[0].hist(data.iloc[:, 1], color='r')
    ax[0].set_title('X-axis')

    ax[1].hist(data.iloc[:, 2], color='g')
    ax[1].set_title('Y-axis')

    ax[2].hist(data.iloc[:, 3], color='b')
    ax[2].set_title('Z-axis')

    for ax in ax:
        ax.set_xlabel('Acceleration (m/s^2)')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()



plot_accVstime('ErikWalk.csv', 'Walking', 'Erik')
plot_accVstime('ErikJump.csv', 'Jumping', 'Erik')

plot_accVstime('EmreWalk.csv', 'Walking', 'Emre')
plot_accVstime('EmreJump.csv', 'Jumping', 'Emre')

plot_accVstime('SimonWalk.csv', 'Walking', 'Simon')
plot_accVstime('SimonJump.csv', 'Jumping', 'Simon')

plot_histograms('ErikWalk.csv', 'Walking', 'Erik')
plot_histograms('ErikJump.csv', 'Jumping', 'Erik')

plot_histograms('EmreWalk.csv', 'Walking', 'Emre')
plot_histograms('EmreJump.csv', 'Jumping', 'Emre')

plot_histograms('SimonWalk.csv', 'Walking', 'Simon')
plot_histograms('SimonJump.csv', 'Jumping', 'Simon')

plt.figure(figsize=(10, 6))
plt.scatter("17:31:46.740", 0, label='Simon Walking')
plt.scatter("16:38:23.053", 0, label='Simon Jumping')

plt.scatter("14:37:03.106", 0, label='Erik Walking')
plt.scatter("14:41:04.312", 0, label='Erik Jumping')

plt.scatter("14:59:47.717", 0, label='Emre Walking')
plt.scatter("15:05:21.023", 0, label='Emre Jumping')

plt.xlabel('Time Recorded')
plt.title('Data Collection Start Times On 2024-03-26 ')
plt.legend()
plt.tight_layout()

participants = ['Erik', 'Simon', 'Emre']
conditions = ['Right Hand', 'Left Pocket', 'Back Right Pocket']
phones = ['iPhone 13', 'iPhone 13 Pro', 'iPhone 14']
activities = ['Walking', 'Jumping']

labels = [f'{activity} - {condition}' for activity in activities for condition in conditions]
fig, ax = plt.subplots()
ax.pie([1, 1, 1, 1, 1, 1], labels=labels, autopct='%1.1f%%')
ax.axis('equal')
plt.title('Distribution of Data Collection Across Conditions and Activities for each Individual for 6 Minuets Total')

fig, ax = plt.subplots()
ax.pie([1, 1, 1], labels=phones, autopct='%1.1f%%')
ax.axis('equal')
plt.title('Data Collection Distribution Among Different Phones using phyphox Application')








reduce_noise_and_normalize('ErikWalk.csv', 'ErikWalkOptimized.csv')
reduce_noise_and_normalize('ErikJump.csv', 'ErikJumpOptimized.csv')


plot_accVstime('ErikWalk.csv', 'Walking Data', 'Erik')
plot_accVstime('ErikWalkOptimized.csv', 'Optimized Walking Data', 'Erik')
plt.show()



def extract_features_and_normalize(input_csv, output_csv):
    window_size = 5
    data = pd.read_csv(input_csv)
    labels = data.iloc[:, -2]

    features = pd.DataFrame()

    for i in range(1, data.shape[1]-2):
        column_features = pd.DataFrame()
        column_features[f'mean.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).mean()
        column_features[f'std.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).std()
        column_features[f'max.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).max()
        column_features[f'min.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).min()
        column_features[f'kurtosis.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).kurt()
        column_features[f'skew.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).skew()
        column_features[f'range.{i}'] = column_features[f'max.{i}'] - column_features[f'min.{i}']
        column_features[f'variance.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).var()
        column_features[f'median.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).median()
        column_features[f'Sum.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).sum()
        column_features[f'Standard Error.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).sem()
        column_features[f'Exponential Moving Average.{i}'] = data.iloc[:,i].ewm(span=window_size).mean()
        features = pd.concat([features, column_features], axis=1)

    normalized_features = (features - features.mean()) / features.std()
    final_data = pd.concat([normalized_features, labels.reset_index(drop=True)], axis=1)
    final_data = final_data.fillna(method='bfill')
    final_data.to_csv(output_csv, index=False)


extract_features_and_normalize('Train.csv', 'TrainFeatures.csv')
extract_features_and_normalize('Test.csv', 'TestFeatures.csv')

trainingdf = pd.read_csv('TrainFeatures.csv')
testdf = pd.read_csv('TestFeatures.csv')

trainingData = trainingdf.iloc[:, :-1]
testData = testdf.iloc[:, :-1]
trainingLabel = trainingdf.iloc[:, -1]
testLabel = testdf.iloc[:, -1]

logisticReg = LogisticRegression(max_iter=10000)  # make pipeline & classifier
classifier = make_pipeline(StandardScaler(), logisticReg)

classifier.fit(trainingData, trainingLabel)  # train classifier

prediction = classifier.predict(testData)  # get prediction and probabilities
classifierProb = classifier.predict_proba(testData)
print('Prediction is: ', prediction)
print('classifier Probability is: ', classifierProb)

accuracy = accuracy_score(testLabel, prediction)  # get accuracy
print('accuracy: ', accuracy)

recall = recall_score(testLabel, prediction)  # get recall
print('recall: ', recall)

F1 = f1_score(testLabel, prediction)
print("F1 : ", F1)

Precision = precision_score(testLabel, prediction)
print("Precision: ", Precision)

cm = confusion_matrix(testLabel, prediction)  # show confusion matrix
cmDisplay = ConfusionMatrixDisplay(cm).plot()
plt.show()

fpr, tpr, _ = roc_curve(testLabel, classifierProb[:, 1], pos_label=classifier.classes_[1])  # show roc curve
rocDisplay = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

auc = roc_auc_score(testLabel, classifierProb[:, 1])  # get auc
print('AUC: ', auc)

# Save the trained model to a .pkl file
joblib.dump(classifier, 'Final 292 assignment/model.plk')

model = joblib.load('Final 292 assignment/model.plk')


def reduce_noise_and_normalize(indf): # from other code slightly adjusted to take and return df
    window_size = 5
    data = indf
    timestamps = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    filtered_data = data.rolling(window=window_size, min_periods=1).mean()
    sc = preprocessing.StandardScaler()
    normalized_data = sc.fit_transform(filtered_data)
    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

    final_df = pd.concat([timestamps, normalized_df], axis=1)
    final_df = final_df.dropna()
    return final_df


def extract_features_and_normalize(indf): # from other code slightly adjusted to take and return df
    window_size = 5
    data = indf
    features = pd.DataFrame()

    for i in range(1, data.shape[1]):
        column_features = pd.DataFrame()
        column_features[f'mean.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).mean()
        column_features[f'std.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).std()
        column_features[f'max.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).max()
        column_features[f'min.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).min()
        column_features[f'kurtosis.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).kurt()
        column_features[f'skew.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).skew()
        column_features[f'range.{i}'] = column_features[f'max.{i}'] - column_features[f'min.{i}']
        column_features[f'variance.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).var()
        column_features[f'median.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).median()
        column_features[f'Sum.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).sum()
        column_features[f'Standard Error.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).sem()
        column_features[f'Exponential Moving Average.{i}'] = data.iloc[:, i].ewm(span=window_size).mean()
        features = pd.concat([features, column_features], axis=1)

    normalized_features = (features - features.mean()) / features.std()
    final_data = normalized_features.dropna()

    return final_data


def open_csv_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        pickedFile.config(text=f"Selected file: {filepath}")
        pickedFile.pack
        middle = pickedFile.winfo_x()
        pickedFile.place(x=middle, y=300)
    else:
        pickedFile.config(text="No file Selected")
        pickedFile.pack
        middle = pickedFile.winfo_x()
        pickedFile.place(x=middle, y=300)

    df = pd.read_csv(filepath)

    df = reduce_noise_and_normalize(df)
    og = df
    df = extract_features_and_normalize(df)

    # Use the model to predict the labels
    df = model.predict(df)
    predictions_df = pd.DataFrame(df, columns=['Predictions'])

    df = pd.concat([og, predictions_df], axis=1)

    # create stem plot of accel vs time
    t = df.iloc[:, 0]
    a = df.iloc[:, 4]
    mask = df.iloc[:, 5] == 1

    plt.figure(figsize=(10, 6))
    plt.stem(t[mask], a[mask], 'r', markerfmt='', label='Jumping')
    plt.stem(t[~mask], a[~mask], 'b', markerfmt='', label='Walking')

    plt.xlabel('Time')
    plt.ylabel('Absolute Acceleration')
    plt.title('ABS Acceleration vs Time')
    plt.legend()
    plt.show()

    df.to_csv("labeledOutput.csv")
    outFile.config(text="The output file is saved as labeledOutput.csv in the root directory")
    outFile.pack()
    outFile.place(x=200, y=350)


box = Tk()
box.geometry("800x600")  # create window

helv36 = Font(family="Helvetica", size=36, weight="bold")
title = Label(box, text="Walking or Jumping Predictor", font=helv36)  # big title
title.place(x=70, y=60)

pickedFile = Label(box, text="")  # empty boxes to be used later
pickedFile.place(x=100, y=100)
outFile = Label(box, text="")
outFile.place(x=10, y=10)

b = Button(box, text="Open CSV File", command=open_csv_file)  # button to start everything
b.place(x=350, y=250)

box.mainloop()