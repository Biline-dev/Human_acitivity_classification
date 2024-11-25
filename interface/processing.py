
import pandas as pd
import numpy as np
import pickle


def load_data(file_objects):
    data_path="/home/biline/Desktop/Master2 - Paris8/architexture_model_complex/project/code/data/"
    try:
        df_emg = pd.read_csv(data_path+str(file_objects[0]), index_col=0)
        df_imu = pd.read_csv(data_path+str(file_objects[1]), index_col=0)
        df_ips = pd.read_csv(data_path+str(file_objects[2]), index_col=0)
        df_mocap = pd.read_csv(data_path+str(file_objects[3]), index_col=0)
    except Exception as e:
        raise ValueError(f"Error reading input files: {e}")
    
    # Extract labels (category) and remove person_id if present
    label_df = df_emg[['category']]

    def clean_df(df):
        if 'person_id' in df.columns:
            return df.drop(columns=['person_id'])
        return df

    df_emg = clean_df(df_emg)
    df_imu = clean_df(df_imu)
    df_ips = clean_df(df_ips)
    df_mocap = clean_df(df_mocap)

    # Convert to numpy arrays and prepare inputs for the model
    X0 = df_emg.iloc[:, :-1].values
    X1 = df_imu.iloc[:, :-1].values
    X2 = df_ips.iloc[:, :-1].values
    X3 = df_mocap.iloc[:, :-1].values
    y = label_df['category'].values

    return [X0, X1, X2, X3], y
    


# merge
def merge_modalities(X_list):
    """
    The features of all modes are combined into an overall feature matrix.
    """
    return np.hstack(X_list)


def load_data_multi_modal(file_objects):
    """
    Load and preprocess data from a list of file-like objects.

    Args:
        file_objects (list): List of file-like objects corresponding to CSV files (EMG, IMU, IPS, Mocap).

    Returns:
        y_pred (np.ndarray): Model predictions.
    """
    # Read each file into a DataFrame
    data_path="../data/"
    try:
        df_emg = pd.read_csv(data_path+str(file_objects[0]), index_col=0)
        df_imu = pd.read_csv(data_path+str(file_objects[1]), index_col=0)
        df_ips = pd.read_csv(data_path+str(file_objects[2]), index_col=0)
        df_mocap = pd.read_csv(data_path+str(file_objects[3]), index_col=0)
    except Exception as e:
        raise ValueError(f"Error reading input files: {e}")

    # Extract labels (category) and remove person_id if present
    label_df = df_emg[['category']]

    def clean_df(df):
        if 'person_id' in df.columns:
            return df.drop(columns=['person_id'])
        return df

    df_emg = clean_df(df_emg)
    df_imu = clean_df(df_imu)
    df_ips = clean_df(df_ips)
    df_mocap = clean_df(df_mocap)

    # Convert to numpy arrays and prepare inputs for the model
    X0 = df_emg.iloc[:, :-1].values
    X1 = df_imu.iloc[:, :-1].values
    X2 = df_ips.iloc[:, :-1].values
    X3 = df_mocap.iloc[:, :-1].values
    y = label_df['category'].values

    # Reshape the data for the model (samples, time_steps, features)
    X_test = [x.reshape(x.shape[0], 1, x.shape[1]) for x in [X0, X1, X2, X3]]

    # Load the model based on the type (Unified or Multimodal)
    model_path = '../models/multi_model.pkl'  # Replace with appropriate model paths
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # Make predictions
    y_pred = loaded_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    dict_classes = {
        0:"back",
        1:"forward",
        2:"halfsquat",
        3:"still",
    }
    y_pred_classes = [dict_classes[i] for i in y_pred_classes]
    return y_pred_classes

def load_data_unified_modal(file_objects):
    
    X_list, y = load_data(file_objects)
    X = merge_modalities(X_list)
    # Load the model based on the type (Unified or Multimodal)
    model_path = '../models/model_unified.pkl'  # Replace with appropriate model paths
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # Make predictions
    y_pred = loaded_model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    dict_classes = {
        0:"back",
        1:"forward",
        2:"halfsquat",
        3:"still",
    }
    y_pred_classes = [dict_classes[i] for i in y_pred_classes]
    return y_pred_classes