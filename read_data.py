import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder


eICU_time_features = ['pH', 'Temperature (C)', 'Respiratory Rate', 'O2 Saturation', 'MAP (mmHg)',
                      'Heart Rate', 'glucose', 'GCS Total', 'Motor', 'Eyes', 'Verbal', 'FiO2',
                      'Invasive BP Diastolic', 'Invasive BP Systolic']
eICU_nontime_features = [ 'gender', 'age', 'ethnicity', 'admissionheight', 'admissionweight']
eICU_label_name = ['hospitaldischargestatus']

mimic_time_features = ['pH', 'Temperature', 'Respiratory rate', 'Oxygen saturation', 'Mean blood pressure',
                        'Heart Rate', 'Glucose', 'Systolic blood pressure', 'Diastolic blood pressure'] #'Fraction inspired oxygen', 'glascow coma scale total' are removed but you can add them if necessary

mimic_nontime_features = ['gender', 'age', 'admission_type', 'first_careunit']
mimic_label_name = ['mort_hosp'] # also use 'readmission_30'


def lower(input):
    return input.lower()
eICU_time_features = list(map(lower, eICU_time_features))
mimic_time_features = list(map(lower, mimic_time_features))

def generate_stats_name_from_feature_name(input):

    """
    Combines the names of the statistics applied to the features with the given input feature name.

    Parameters:
    -----------
    input: str
        The input feature name.

    Returns:
    --------
    list
        A list of statistics-based feature names derived from the input feature name.
    """
    
    returned_names = []
    for percentage in ['0_100', '0_10', '90_100', '0_25', '75_100', '0_50', '50_100']:
      for statistic in ['mean', 'var', 'min', 'max', 'skew', 'count']:
            name = input + '_' + statistic + '_' + percentage
            returned_names.append(name)
    return returned_names

def generate_colomns_name(features, time_features_name, nontime_features_name, label_name):
    
    """
    Generate the list of column names for a dataset based on desired time-dependent and time-independent features.

    Parameters:
    -----------
    features: pd.DataFrame
        The dataset containing the features.
    time_features_name: list
        List of names of time-based features.
    nontime_features_name : list
        List of names of time-independent features.
    label_name: str
        Name of the label column.

    Returns:
    --------
    list
        A list of columns to be used for a dataset.
    """
    
    colomns_name = []
    for colomn in time_features_name:
        colomns_name += generate_stats_name_from_feature_name(colomn)

    colomns_name += nontime_features_name
    colomns_name += label_name
    for name in colomns_name:
        if name not in features.columns:
            print(name)

    return colomns_name
    

def get_eICU_selected_features(eICU_features):

    """
    Select and preprocess features from the eICU dataset.

    Parameters:
    -----------
    eICU_features: pd.DataFrame
        The eICU dataset containing features and labels.

    Returns:
    --------
    tuple
        A tuple containing two DataFrames:
        - eICU_features_selected: The selected and preprocessed feature dataset.
        - eICU_label: The corresponding labels.
    """
    
    colomns_name = generate_colomns_name(eICU_features, eICU_time_features, eICU_nontime_features, eICU_label_name)
    eICU_features_selected = eICU_features[colomns_name]
    print('eICU selected features shape before dropna:', eICU_features_selected.shape)
    
    eICU_features_selected = eICU_features_selected.dropna()
    print('eICU selected features shape after dropna:', eICU_features_selected.shape)
    
    eICU_features_selected = eICU_features_selected[eICU_features_selected['age'] >= 18]
    eICU_features_selected['gender'].replace(['male', 'female'], [0, 1], inplace=True)
    
    dummies = pd.get_dummies(eICU_features_selected.ethnicity)
    eICU_features_selected = pd.concat([eICU_features_selected, dummies], axis='columns')
    eICU_features_selected = eICU_features_selected.drop(['ethnicity'], axis='columns')
    
    eICU_label = eICU_features_selected[eICU_label_name]
    eICU_features_selected = eICU_features_selected.drop(eICU_label_name, axis='columns')
    print('eICU final features shape:', eICU_features_selected.shape, 'eICU final labels shape:', eICU_label.shape)
    
    return eICU_features_selected, eICU_label
 
 
def get_mimic_selected_features(mimic_features):

    """
    Select and preprocess features from the MIMIC dataset.

    Parameters:
    -----------
    mimic_features: pd.DataFrame
        The MIMIC dataset containing features and labels.

    Returns:
    --------
    tuple
        A tuple containing two DataFrames:
        - mimic_features_selected: The selected and preprocessed feature dataset.
        - mimic_label: The corresponding labels.
    """
    
    for i in range(len(mimic_time_features)):
        mimic_time_features[i] = mimic_time_features[i].lower()

    colomns_name = generate_colomns_name(mimic_features, mimic_time_features, mimic_nontime_features, mimic_label_name)
    mimic_features_selected = mimic_features[colomns_name]
    print('mimic selected features shape before dropna:', mimic_features_selected.shape)
    
    mimic_features_selected = mimic_features_selected.dropna()
    print('mimic selected features shape after dropna:', mimic_features_selected.shape)
    
    mimic_features_selected = mimic_features_selected[mimic_features_selected['age'] >= 18]
    mimic_features_selected['gender'].replace(['M', 'F'], [0, 1], inplace=True)
    
    dummies = pd.get_dummies(mimic_features_selected.admission_type)
    dummies2 = pd.get_dummies(mimic_features_selected.first_careunit)
    mimic_features_selected = pd.concat([mimic_features_selected, dummies, dummies2], axis='columns')
    mimic_features_selected = mimic_features_selected.drop(['admission_type', 'first_careunit'], axis='columns')
    
    mimic_label = mimic_features_selected[mimic_label_name]
    mimic_features_selected = mimic_features_selected.drop(mimic_label_name, axis='columns')
    
    print('mimic final features shape:', mimic_features_selected.shape, 'mimic final labels shape:', mimic_label.shape)
    
    return mimic_features_selected, mimic_label
    
    
    
def read_data(files_path='', data_name='eicu'):

    """
    Load feature data from the source files.

    Parameters:
    -----------
    files_path: str, optional
        The path to the directory where the CSV files are located.
    data_name: str, optional
        The name of the dataset to load. Default is 'eICU'.

    Returns:
    --------
    tuple
        A tuple containing two DataFrames:
        - data: The selected and preprocessed feature dataset.
        - label: The corresponding labels.
    """
    
    if data_name == 'eicu':
        print('Loading eICU_features ...')
        data = pd.read_csv(files_path + 'eICU_preprocessed.csv').drop(columns=['Unnamed: 0'])
        data.columns = data.columns.str.lower()
        data, label = get_eICU_selected_features(data)
    
    elif data_name == 'mimic':
        print('Loading mimic_features ...')
        data = pd.read_csv(files_path + 'MIMIC_iv_preprocessed.csv').drop(columns=['Unnamed: 0']) 
        data.columns = data.columns.str.lower()
        data, label = get_mimic_selected_features(data)
    
    elif data_name == 'sepsis':
        print('Loading sepsis_features ...')
        data = pd.read_csv(files_path + 'Sepsis_preprocessed.csv').drop(columns=['Unnamed: 0']) 
        
        sampled_fraction = 0.1
        num_rows_to_sample = int(len(data) * sampled_fraction)
        random_index = data.sample(n=num_rows_to_sample, random_state=42).index
        data = data.loc[random_index]
    
        label = data['SepsisLabel']
        data = data.drop('SepsisLabel', axis='columns')
        print('sepsis features shape:', data.shape, 'sepsis labels shape:', label.shape)
    
    elif data_name == 'diabetics':
        print('Loading diabetics_features ...')
        data = arff.loadarff(files_path + 'messidor_features.arff')
        data = pd.DataFrame(data[0])
        label = data['Class'].apply(lambda x: int(x.decode('utf-8')))
        data = data.drop('Class', axis='columns')
        print('diabetics features shape:', data.shape, 'diabetics labels shape:', label.shape)
    
    elif data_name == 'drybean':
        print('Loading DryBeans_features ...')
        data = arff.loadarff(files_path + 'DryBeanDataset/Dry_Bean_Dataset.arff')
        data = pd.DataFrame(data[0])
        label_encoder = LabelEncoder()
        data['Class'] = label_encoder.fit_transform(data['Class'])
        label = data['Class']
        data = data.drop('Class', axis='columns')
        print('DryBeans features shape:', data.shape, 'DryBeans labels shape:', label.shape)
    
    elif data_name == 'wine':
        data_white = pd.read_csv(files_path + 'winequality-white.csv', delimiter=';')
        data_red = pd.read_csv(files_path + 'winequality-red.csv', delimiter=';')
        data = pd.concat([data_white, data_red])
        label = pd.DataFrame(np.concatenate([np.zeros(len(data_white), dtype=np.int64), np.ones(len(data_red), dtype=np.int64)]))
        print('Wine features shape:', data.shape, 'Wine labels shape:', label.shape)
    
    return data, label