import pandas as pd
import matplotlib.pyplot as plt

# Reading the CSV file
df = pd.read_csv('DSL-StrongPasswordData.csv')

# Counting the values of the "subject" column
subject_counts = df["subject"].value_counts()

# Filtering columns with names containing "H.", "UD.", "subject", or "sessionIndex"
df = df.filter(regex='H\.|DD\.|UD\.|subject|sessionIndex')
column_names = df.columns

# keep 3third to end columns
df[(df["subject"] == "s057") & (df["sessionIndex"] == 1)][df.columns[2:]].iloc[0]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8), sharex=True)
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for subject, axlist in enumerate(axes):
    for i, ax in enumerate(axlist):
        filtered_df = df[(df["subject"] == f"s00{subject+3}") & (df["sessionIndex"] == i+1)][df.columns[3:]]
        for attemp in range(0, len(filtered_df)):
            ax.plot(filtered_df.iloc[attemp], color=colors[subject] , alpha=0.1)
        ax.set_ylim(0, 1.7)
        if i == 0:
            ax.set_ylabel(f"Subject {subject+1}")
        if subject == 2:
            ax.set_xlabel(f"Session {i+1}")

#plt.xticks(range(0, len(df.columns[3:])), df.columns[3:], rotation=90)
#remove xticks
plt.xticks([])

plt.ylim(0, 1.7)
plt.savefig('boxplot.png')
plt.close()

#make knn algorithm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_curve
import numpy as np

def decorrelate_normalize(matrix):
    # Calculate the covariance matrix
    covariance_matrix = np.cov(matrix, rowvar=False)
    
    # Calculate the square root of the inverse of the covariance matrix
    sqrt_inv_covariance_matrix = np.linalg.inv(np.linalg.cholesky(covariance_matrix))
    
    # Decorrelate the matrix by multiplying it with the square root of the inverse of the covariance matrix
    decorrelated_matrix = np.dot(matrix, sqrt_inv_covariance_matrix)
    
    # Normalize the decorrelated matrix by subtracting the mean and dividing by the standard deviation
    normalized_matrix = (decorrelated_matrix - np.mean(decorrelated_matrix, axis=0)) / np.std(decorrelated_matrix, axis=0)
    
    return normalized_matrix


def apply_transformation(features):
    features = decorrelate_normalize(features)
    transformed_features = features
    # Calculate the square root of the covariance matrix
    cov_matrix = np.cov(features.T)
    chol_matrix = np.linalg.cholesky(cov_matrix)
    # # Invert the matrix
    inv_matrix = np.linalg.inv(chol_matrix)
    # # Apply the linear transformation
    transformed_features = np.dot(features, inv_matrix)

    return transformed_features

#Apllying transformation
# df[df.columns[2:]] = apply_transformation(df[df.columns[2:]])
random_state = np.random.randint(0, 10000)
# random_state = 12354
np.random.seed(random_state)
mean_far = []
mean_frr = []
mean_err = []
for subject in df["subject"].unique():
    # nn = NearestNeighbors(n_neighbors=1, metric='manhattan')
    df_s057 = df[df["subject"] == subject]
    #split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(df_s057[df.columns[2:]], df_s057["subject"], test_size=0.5, random_state=random_state)
    
    #get mean of each column of x_train and reshape (-1, 1)
    mean = X_train.mean(axis=0)

    #create test data with x_test and y y_test
    test_data = pd.concat([X_test, y_test], axis=1)

    other_users = df[df["subject"] != subject]["subject"].unique()
    #pick 50 records from each user
    other_users_data = pd.concat([df[df["subject"] == user].sample(5) for user in other_users])
    other_users_data.drop(columns=["sessionIndex"], inplace=True)
    #concat test data with other users data
    test_data = pd.concat([test_data, other_users_data], axis=0)
    test_data['subject'].value_counts()
    #get EER
    #get distance from each record to mean
    y_pred = []
    for i in range(0, len(test_data)):
        distance = np.abs(test_data.iloc[i, :-1] - mean).sum()
        y_pred.append(distance)


    result = pd.DataFrame(y_pred, columns=["distance"])
    result["subject"] = test_data["subject"].values
    result["subject"] = result["subject"].apply(lambda x: subject if x == subject else f"not {subject}")


    #get EER
    umbral = 0.2
    errs = []
    fars = []
    frrs = []
    err = 10000

    x_list = np.arange(0, 1, 0.01)

    result["distance"] = result["distance"].apply(lambda x: x / result["distance"].max()) 
    for umbral in x_list:
        result["predicted"] = result["distance"].apply(lambda x: subject if x < umbral else f"not {subject}")

        #calculate accuracy
        false_acceptance = result[(result["subject"] == f"not {subject}") & (result["predicted"] == subject)].shape[0]
        false_rejection = result[(result["subject"] == subject) & (result["predicted"] == f"not {subject}")].shape[0]
        false_acceptance_rate = false_acceptance / result[result["subject"] == f"not {subject}"].shape[0]
        false_rejection_rate = false_rejection / result[result["subject"] == subject].shape[0]
        current_err = (false_acceptance_rate + false_rejection_rate) / 2
        # errs.append(current_err)
        fars.append(false_acceptance_rate)
        frrs.append(false_rejection_rate)
        err = min(err, current_err)
    
    mean_far.append(fars)
    mean_frr.append(frrs)

mean_far = np.array(mean_far)
mean_frr = np.array(mean_frr)
mean_far = mean_far.mean(axis=0)
mean_frr = mean_frr.mean(axis=0)

err= np.argmin(np.abs(mean_far - mean_frr))

plt.plot(x_list, mean_far, label='FAR')
plt.plot(x_list, mean_frr, label='FRR')
plt.xlabel('Umbral')
plt.ylabel('Error rate')
plt.title('Manhattan distance')

print(err)
plt.plot(x_list[err], mean_far[err], 'ro', label='EER')

#add text to plot
plt.text(x_list[err]+0.02, mean_far[err], f'EER: {mean_far[err]}')

plt.legend()
plt.savefig('eer.png')

# plt.show()
