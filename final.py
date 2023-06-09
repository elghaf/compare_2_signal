import os

def load_user_data(data_path):
    user_data = {}
    folder_path = "Task1"
    da =  os.listdir(folder_path)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        user_id, signature_id = map(int, file_name.split(".")[0][1:].split("S"))
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            signature_data = [line.strip().split() for line in lines[1:]]  # Skip the first line
            
            if user_id not in user_data:
                user_data[user_id] = []
                
            user_data[user_id].append((signature_id, signature_data))

    return user_data


def extract_features(signature_data):
    features = []
    for entry in signature_data:
        x_coord = float(entry[0])
        y_coord = float(entry[1])
        timestamp = int(entry[2])
        button_status = int(entry[3])
        azimuth = float(entry[4])
        altitude = float(entry[5])
        pressure = float(entry[6])
        
        # Perform additional feature extraction or transformations as needed
        # For example, you can calculate velocity, acceleration, or other derived features
        
        feature_vector = [x_coord, y_coord, timestamp, button_status, azimuth, altitude, pressure]
        features.append(feature_vector)
    
    return features


def prepare_dataset(user_data):
    dataset = []
    for user_id, signatures in user_data.items():
        for signature_id, signature_data in signatures:
            features = extract_features(signature_data)
            label = user_id  # Use the user ID as the label
            
            dataset.append((features, label))
    
    return dataset


data_path = "Task1"  # Provide the path to the parent folder containing "Task1"
user_data = load_user_data(data_path)
dataset = prepare_dataset(user_data)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Split dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    np.array([sample[0] for sample in dataset]),  # Features
    np.array([sample[1] for sample in dataset]),  # Labels
    test_size=0.2,  # Specify the desired ratio of testing set
    random_state=42  # Set a random seed for reproducibility
)

print("Training set size:", len(train_data))
print("Testing set size:", len(test_data))

# Create an SVM classifier
svm = SVC()

# Train the classifier
svm.fit(train_data, train_labels)

# Make predictions on the testing data
predictions = svm.predict(test_data)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)
