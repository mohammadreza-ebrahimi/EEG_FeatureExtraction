#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import matplotlib.pyplot as plt
import mne               # package for EEG and MEG data analysis
import seaborn as sns    # for visualization 
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import antropy as an
import pandas as pd
import imblearn


import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")     # hides INFO + WARNING


# In[ ]:


# base_dir = "D:/M.Sc_complex-systems_1402/Msc.Thesis/SearchingDataset/derivatives/"
# base_dir = "E:/data_AD/SearchingDataset/SearchingDataset/derivatives/"
base_dir = 'dir_to_EEG_dataset'

n_samples = 88
raw_data_list = []
ica_list = []

for i in range(1, n_samples + 1):
    sub_dir = os.path.join(base_dir, f"sub-{i:03d}", "eeg")

    # Find .set file
    set_file = next((os.path.join(sub_dir, f) 
                     for f in os.listdir(sub_dir) if f.endswith(".set")), None)

    if set_file:
        raw = mne.io.read_raw_eeglab(set_file, preload=True)

        raw.filter(0.5, 45.)
        raw.notch_filter([50, 100, 150, 200])
        raw.set_eeg_reference(ref_channels='average', projection=False)

        ica = mne.preprocessing.ICA(n_components=19, random_state=97)
        ica.fit(raw, verbose=False)

        raw_data_list.append(raw)
        ica_list.append(ica)


# Combine all participants dat into a single dataset variable (optional)
# if len(raw_data_list) > 1:
#     raw_combined = mne.concatenate_raws(raw_data_list)
# else:
#     raw_combined = raw_data_list[0]

#print(type(raw_combined))


# In[ ]:


raw_data_list[0].plot()
plt.show()


# In[ ]:


ica_list[0].plot_components()
ica_list[0].plot_sources(raw_data_list[0])


# In[ ]:


clean_raw_list = []

for i, (raw, ica) in enumerate(zip(raw_data_list, ica_list)):
    print(f"Processing subject {i+1}...")

    # 1) Detect muscle artifacts
    bad_muscle, scores = ica.find_bads_muscle(raw) 

    # 2) Assign bad ICs
    ica.exclude = bad_muscle

    # 3) Apply ICA to remove bad ICs
    clean_raw = ica.apply(raw.copy())

    # 4) Store cleaned EEG
    clean_raw_list.append(clean_raw)

    print(f"Subject {i+1} bad ICs: {bad_muscle}")


# In[ ]:


# comparision the first sample; before bad muscle and after bad muscle filtering
raw_data_list[0].plot(title='Before Artifact filtering')
clean_raw_list[0].plot(title='After Artifact filtering')
plt.show()


# In[ ]:


# sliding window

def sliding_window(data, fs, window_length=1.0, overlap=0.5):
    window_size = int(fs * window_length)

    # Compute the step in *samples*, guaranteed > 0
    step = int(window_size * (1 - overlap))
    #if step <= 0:
    #    step = 1  #avoid range() error

    n_channels, n_samples = data.shape

    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(data[:, start:end])

    return np.stack(windows, axis=0)


# In[ ]:


patients_data = []
for patient in clean_raw_list:
    data = patient.get_data()
    data = data.astype('float32')
    patients_data.append(data)


patients_data[0].shape


# In[ ]:


all_windows_5p = []

for data in patients_data:
    windows = sliding_window(data, fs=500, window_length=1, overlap=0.5)

    all_windows_5p.append(windows)

X_5 = np.concatenate(all_windows_5p, axis=0)
np.save('data_50_overlap.npy', X_5)
#X.shape

# save list
np.save("all_windows_5p_list.npy", np.array(all_windows_5p, dtype=object), allow_pickle=True)


# In[ ]:


X_5.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Loading Data

# In[2]:


X_5 = np.load('data_50_overlap.npy')
X5_window_list = np.load('all_windows_5p_list.npy', allow_pickle=True)

# shap
print(
    f'Shape of concatenated data:{X_5.shape}\nShape of one patient\'s data after sliding window (N_windows, channels, n_smaple): {X5_window_list[0].shape}'
)


# In[3]:


participants_file = "E:/data_AD/SearchingDataset/SearchingDataset/participants.tsv"

participants_data = pd.read_csv(participants_file, sep='\t')
print("Participants data loaded successfully!")

# first 20 
# participants_subset = participants_data.head(20)
# print("Info of the first 20 participants:\n")
# print(participants_subset)
participants_data.head()


# In[4]:


window_counts = [w.shape[0] for w in X5_window_list]
len(window_counts)


# In[5]:


labels = []
for i, counts in enumerate(window_counts):
    if i < 36:
        labels += ['A'] * counts
    elif i < 36 + 29:
        labels += ['C'] * counts
    else:
        labels += ['F'] * counts

labels = np.array(labels)


# In[6]:


labels.shape


# In[7]:


# X_5[0,0,:].shape


# # With PSD analysis

# In[8]:


from mne.time_frequency import psd_array_welch
import antropy as an
import numpy as np

def extract_psd_svd_entropy(window, sfreq, bands, order=10):
    """
    window: shape (n_channels, n_times)
    returns: feature vector of shape (n_channels * n_bands,)
    """

    psd, freqs = psd_array_welch(
        window,
        sfreq=sfreq,
        fmin=min(low for low, _ in bands.values()),
        fmax=max(high for _, high in bands.values()),
        n_fft=window.shape[1],
        n_overlap=window.shape[1] // 2,
        window="hann",
        average="mean"
    )

    feature_vector = []

    for low, high in bands.values():
        mask = (freqs >= low) & (freqs <= high)

        for ch in range(psd.shape[0]):
            band_psd = psd[ch, mask]          # PSD spectrum in band
            ent = an.svd_entropy(
                band_psd,
                order=min(order, band_psd.size - 1),
                normalize=True
            )
            feature_vector.append(ent)

    return np.array(feature_vector)


# In[9]:


bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 45)
}

sfreq = 500

X_features = []

for w in range(X_5.shape[0]):
    feats = extract_psd_svd_entropy(
        X_5[w],
        sfreq=sfreq,
        bands=bands,
        order=10
    )
    X_features.append(feats)

X_features = np.array(X_features)


# In[10]:


X_features.shape


# In[ ]:





# # without Power Spectral Density (PSD) analysis
# - SVD (Singular Value Decomposition) Entropy calculation for each window

# In[10]:


# band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
# n_channels = 19

# bands_dict = {
#     band: entropy_features[:, i*n_channels:(i+1)*n_channels]
#     for i, band in enumerate(band_names)
# }


# In[12]:


subject_ids = np.concatenate([
    np.full(X5_window_list[i].shape[0], i)
    for i in range(len(X5_window_list))
])
assert subject_ids.shape[0] == sum(x.shape[0] for x in X5_window_list)


# In[13]:


subject_ids


# In[28]:


bands_name = ['delta', 'theta', 'alpha', 'beta', 'gamma']
cols = []
for name in bands_name:
    cols.append([f'{name}_ch{i+1}' for i in range(19)])

columns = [
    x
    for xs in cols
    for x in xs
]


# In[41]:


# ch_names = [
#     'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
#     'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'
# ]
df = pd.DataFrame(
    # entropy_features,
    data=X_features,
    columns=columns
)

df['subject_id'] = subject_ids
df['label'] = labels




# In[42]:


df


# In[43]:


df[['subject_id', 'label']].drop_duplicates().head(100)


# In[44]:


# check if dataset splitted subject-wise (per patient)
2783 - 1198 == X5_window_list[1].shape[0]


# In[45]:


assert df.groupby('subject_id')['label'].nunique().max() == 1


# In[46]:


windows_per_subject = df.groupby('subject_id').size()
windows_per_subject.sort_values(ascending=False).head(14000)
windows_per_subject.describe()


# In[47]:


# labels distribution

subject_labels = (
    df[['subject_id', 'label']]
    .drop_duplicates()
    .value_counts('label')
)

print(subject_labels)


# In[212]:


df[['subject_id', 'label']].drop_duplicates()['label'].value_counts()


# In[48]:


# Now we get the mean of mean of windows of each participant for examp. num_subject ID 0 = 1198 
# that we mean to have 88 averaged windows for 88 particpants 

df_subject = (
    df
    .groupby('subject_id')
    .mean(numeric_only=True)
)

df_subject['label'] = (
    df[['subject_id', 'label']]
    .drop_duplicates()
    .set_index('subject_id')['label']
)

df_subject


# In[140]:


# ===================================
# RUN THIS CELL FOR comparing two groups like: AD and C, or FDT and C, Or FDT and AD
# ===============================

# df.columns = df.columns.astype(str)

df_filtered = df_subject[df_subject['label'].isin(['A', 'C'])]



# Now df_filtered only contains rows with labels 'C' and 'A'
print(df_filtered['label'].value_counts())


# In[141]:


# ===================================
# RUN THIS CELL FOR comparing 2 groups
# ==================================

X = df_filtered.drop('label', axis=1)  # features
y = df_filtered['label']               # target
y.shape


# In[142]:


df.shape, df_subject.shape
# df is dataframe of all windows (140336 * 21), but df_subject is mean of windows for each subjectss (88, 20)


# In[ ]:





# In[145]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)


# In[146]:


print('Train dataset')
print(y_train.value_counts())
print('Test dataset')
print(y_test.value_counts())


# In[134]:


# smote = imblearn.over_sampling.SMOTE(random_state=42)

# X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


# In[135]:


#print(y_train_sm.value_counts())


# In[136]:


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train_sm = scaler.fit_transform(X_train_sm)
# X_test = scaler.transform(X_test)


# In[ ]:





# In[147]:


from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine (SVM)': SVC(random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    #'AdaBoost': AdaBoostClassifier(random_state=42),
    #'XGBoost': XGBClassifier(random_state=42)
}



for model_name, model in models.items():

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ( model_name, model)
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    # cv = RepeatedStratifiedKFold(
    #     n_splits=5,
    #     n_repeats=10,
    #     random_state=42
    # )


    # gkf = GroupKFold(n_splits=5)

    # cv_scores = cross_val_score(
    #     pipe,
    #     X,
    #     y,
    #     cv=gkf,
    #     groups=df_filtered['subject_id'],
    #     scoring='accuracy'
    # )

    cv_scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy'
    )
    print(model_name)
    #print(cv_scores)
    print("CV Accuracy:%.3f" %(cv_scores.mean()), "+/-%.4f" %(cv_scores.std()))
    print('='*80)


# In[148]:


from sklearn.metrics import accuracy_score, classification_report


for model_name, model in models.items():

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ( model_name, model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(model_name)
    print("Final Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('='*80)

    #cm = confusion_matrix(y_test, y_pred)
    #print(cm)


# In[ ]:




