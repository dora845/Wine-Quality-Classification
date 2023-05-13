# %%
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("Wine_Quality_Data.csv")

df.head()

# %%
# check for missing values

info = pd.DataFrame(df.isnull().sum(), columns=["isNull"])
info.insert(1, "isNa", df.isna().sum(), True)
info.insert(2, "duplicate", df.duplicated().sum(), True)
info.insert(3, "unique", df.nunique(), True)
info

# %%
df.info()

# %% [markdown]
# Preprocessing

# %%
df1 = df.copy()

# plot for each quality

plot = sns.catplot(x='quality', data=df1, kind="count")

# %%
df["quality"].value_counts()
df['color'].unique()


# %%

label_encoder = preprocessing.LabelEncoder()

df['color'] = label_encoder.fit_transform(df['color'])

df['color'].unique()

# %%


def diagnostic_plots(df, variable, target):
    # The function takes a dataframe (df) and
    # the variable of interest as arguments.

    # Define figure size.
    plt.figure(figsize=(20, 4))

    # histogram
    plt.subplot(1, 4, 1)
    sns.histplot(df[variable], kde=True, color='r')
    plt.title('Histogram')

    # scatterplot
    plt.subplot(1, 4, 2)
    plt.scatter(df[variable], df[target], color='g')
    plt.title('Scatterplot')

    # boxplot
    plt.subplot(1, 4, 3)
    sns.boxplot(y=df[variable], color='b')
    plt.title('Boxplot')

    # barplot
    plt.subplot(1, 4, 4)
    sns.barplot(x=target, y=variable, data=df)
    plt.title('Barplot')

    plt.show()


# %%
for features in df:
    diagnostic_plots(df, features, 'quality')

# %%

plt.figure(figsize=(10, 8))
# sb.heatmap(df.corr(),
#             annot=True,
#             linewidths=.5,
#             center=0,
#             cbar=False,
#             cmap="YlGnBu")
sb.heatmap(df.corr(),
           annot=True,)
plt.show()

# %%
df.drop_duplicates()

# %%
df.drop('total_sulfur_dioxide', axis=1, inplace=True)

# %%

plt.figure(figsize=(10, 8))
# sb.heatmap(df.corr(),
#             annot=True,
#             linewidths=.5,
#             center=0,
#             cbar=False,
#             cmap="YlGnBu")
sb.heatmap(df.corr(),
           annot=True,)
plt.show()

# %%
# Z = df.drop('color',axis=1)
# array = Z.values
X = df.drop('quality', axis=1)
Y = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
Y.value_counts(normalize=True)

# %%

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
Y_train.head()

# %% [markdown]
# Scaling Data

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# scaler.fit(X)

# %% [markdown]
# Training Data with RandomForest

# %%
# train a decision tree classifier

dt = RandomForestClassifier(n_estimators=100, random_state=42)

dt.fit(X_train_scaled, Y_train)

# %%
y_pred = dt.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, y_pred)
print('Accuracy:', accuracy)
print("f1 score :", f1_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

# %% [markdown]
# Training data with XGBoost

# %%
model1 = xgb.XGBClassifier(random_state=42)
model1.fit(X_train_scaled, Y_train)
y_pred1 = model1.predict(X_test_scaled)
print(classification_report(Y_test, y_pred1))
print("f1 score :", f1_score(Y_test, y_pred1))

# %% [markdown]
# Training data with decisionTree
#

# %%
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

print(classification_report(Y_test, y_pred2))

# %% [markdown]
# Select Features

# %%
# select the top 5 features
selector = SelectKBest(score_func=chi2, k=5)


# %%
X_train_selected = selector.fit_transform(X_train, Y_train)
X_test_selected = selector.transform(X_test)

# %%
# print the top 5 features using SelectkBest
selected_features = X.columns[selector.get_support()]

print("Selected features: ", selected_features)

# %%
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, Y)

# %%
selected_features_indices = selector.get_support(indices=True)
selected_features_names = X.columns[selected_features_indices]
print(selected_features_names)

# %%
dfModified = pd.DataFrame(df1["volatile_acidity"],
                          columns=["volatile_acidity"])
# info.insert(1,"isNa",df.isna().sum(),True)
dfModified.insert(1, "residual_sugar", df1["residual_sugar"], True)
dfModified.head()

# %%
dfModified = pd.DataFrame()
i = 0
for features in selected_features_names:
    dfModified.insert(i, features, df1[features], True)
    i = i+1

dfModified.head()

# %% [markdown]
# train the new dataset

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    dfModified, Y, test_size=0.2, random_state=42)

# %%
dt.fit(X_train, Y_train)

# %%
y_pred = dt.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print('Accuracy:', accuracy)
print("f1 score :", f1_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

# %% [markdown]
# Optimize class weight
