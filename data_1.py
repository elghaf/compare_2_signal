import os

def load_signature_data(directory):
    signature_data = []
    
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Read the file and extract the lines
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # Remove the first line (total number of lines)
        lines = lines[1:]
        
        # Parse and process each line of signature data
        signature = []
        for line in lines:
            line = line.strip().split(' ')
            signature.append([float(value) for value in line])
        
        signature_data.append(signature)
    
    return signature_data

# Specify the directory where the signature data is stored
data_directory = 'Task1'

# Load the signature data
signature_data = load_signature_data(data_directory)

import numpy as np

def extract_features(signature_data):
    features = []
    
    for signature in signature_data:
        signature_features = []
        
        # Extract statistical features from each dimension
        for dimension in range(len(signature[0])):
            dimension_data = [data_point[dimension] for data_point in signature]
            mean = np.mean(dimension_data)
            std = np.std(dimension_data)
            signature_features.extend([mean, std])
        
        # Additional features
        x_coords = [data_point[0] for data_point in signature]
        y_coords = [data_point[1] for data_point in signature]
        timestamps = [data_point[2] for data_point in signature]
        pressures = [data_point[6] for data_point in signature]
        
        # Curvature features
        curvatures = calculate_curvature(x_coords, y_coords)
        mean_curvature = np.mean(curvatures)
        std_curvature = np.std(curvatures)
        signature_features.extend([mean_curvature, std_curvature])
        
        # Speed features
        speeds = calculate_speed(x_coords, y_coords, timestamps)
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        signature_features.extend([mean_speed, std_speed])
        
        # Pressure features
        mean_pressure = np.mean(pressures)
        max_pressure = np.max(pressures)
        signature_features.extend([mean_pressure, max_pressure])
        
        # Stroke count
        stroke_count = calculate_stroke_count(signature)
        signature_features.append(stroke_count)
        
        features.append(signature_features)
    
    return features


def calculate_curvature(x_coords, y_coords):
    # Calculate the curvature of the signature trajectory
    dx = np.gradient(x_coords)
    ddx = np.gradient(dx)
    dy = np.gradient(y_coords)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)
    return curvature.tolist()

def calculate_speed(x_coords, y_coords, timestamps):
    # Calculate the speed of the signature movement
    deltas = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    time_diffs = np.diff(timestamps)
    speeds = deltas / time_diffs
    return speeds.tolist()

def calculate_stroke_count(signature):
    # Calculate the stroke count based on pen-up and pen-down events
    button_status = [data_point[3] for data_point in signature]
    stroke_count = sum(button_status)
    return stroke_count

# Extract features from the loaded signature data
extracted_features = extract_features(signature_data)



#-----------------


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Prepare the data for training
X = extracted_features
y = [1] * len(X)  # Assign a label of 1 to all signatures (can be modified for different classes)

# Convert X to a NumPy array
X = np.array(extracted_features)

# Replace infinity values with NaN
X[np.isinf(X)] = np.nan

# Perform data cleaning and preprocessing
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Check for NaN or infinite values after imputation
if np.isnan(X).any() or np.isinf(X).any():
    # Handle remaining NaN or infinite values, e.g., by setting them to 0 or a large negative value
    X = np.nan_to_num(X, nan=0, posinf=np.finfo(X.dtype).max, neginf=np.finfo(X.dtype).min)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy}")



