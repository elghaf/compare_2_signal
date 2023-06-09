import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_signature_data(file_path):
    with open(file_path, 'r') as file:
        num_lines = int(file.readline().strip())
        signature_data = np.empty((num_lines, 7))
        for i, line in enumerate(file.readlines()):
            line_data = line.strip().split(' ')
            signature_data[i] = [int(x) for x in line_data]
    return signature_data.astype(np.float64)


def extract_features(signature):
    # Extracting features from signature data
    x_coordinates = signature[:, 0]
    y_coordinates = signature[:, 1]
    timestamps = signature[:, 2]
    button_status = signature[:, 3]
    azimuth = signature[:, 4]
    altitude = signature[:, 5]
    pressure = signature[:, 6]

    # Calculate time duration
    time_duration = timestamps[-1] - timestamps[0]

    # Calculate velocity
    distance = np.sqrt(np.sum(np.diff(x_coordinates) ** 2 + np.diff(y_coordinates) ** 2))
    velocity = distance / time_duration

    # Count direction changes
    direction_changes = np.sum(np.abs(np.diff(np.arctan2(np.diff(y_coordinates), np.diff(x_coordinates)))) > np.pi / 2)

    # Calculate pressure variation
    pressure_range = np.max(pressure) - np.min(pressure)
    pressure_std = np.std(pressure)

    # Count stroke count
    stroke_count = np.sum(np.diff(button_status) > 0)

    # Calculate aspect ratio
    aspect_ratio = (np.max(x_coordinates) - np.min(x_coordinates)) / (np.max(y_coordinates) - np.min(y_coordinates))

    # Extract start and end point positions
    start_point = [x_coordinates[0], y_coordinates[0]]
    end_point = [x_coordinates[-1], y_coordinates[-1]]

    # Return the extracted features as a dictionary or an array
    features = {
        'time_duration': time_duration,
        'velocity': velocity,
        'direction_changes': direction_changes,
        'pressure_range': pressure_range,
        'pressure_std': pressure_std,
        'stroke_count': stroke_count,
        'aspect_ratio': aspect_ratio,
        'start_point': start_point,
        'end_point': end_point
    }
    return features



def calculate_similarity(features1, features2, threshold):
    # Compare the features and calculate a similarity score
    similarity_score = 0.0

    # Compare each feature and calculate the similarity score
    # Here's an example of calculating similarity based on Euclidean distance
    for i in range(len(features1)):
        if isinstance(features1[i], list):
            # Handle lists (e.g., start_point and end_point)
            similarity_score += np.linalg.norm(np.array(features1[i]) - np.array(features2[i]))
        else:
            # Handle numerical features
            similarity_score += (features1[i] - features2[i]) ** 2

    similarity_score = np.sqrt(similarity_score)

    # Assign the similarity label based on the similarity threshold
    if similarity_score <= threshold:
        similarity_label = 1  # Similar
    else:
        similarity_label = 0  # Dissimilar

    return similarity_label





# Example usage
signature1 = load_signature_data("Task1//U1S1.TXT")
signature2 = load_signature_data("Task1//U1S2.TXT")

features1 = extract_features(signature1)
features2 = extract_features(signature2)

threshold = 5.0

similarity_label = calculate_similarity(features1, features2, threshold)
# Assume you have a list of labeled signature pairs and their similarity labels
labeled_pairs = [
    (features1, features2, similarity_label1),
    (features3, features4, similarity_label2),
    ...
]

# Prepare the data for training
X = []
y = []
for features1, features2, similarity_label in labeled_pairs:
    # Append the absolute difference of features to X
    feature_diff = np.abs(features1 - features2)
    X.append(feature_diff)
    # Append the similarity label to y
    y.append(similarity_label)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model (k-nearest neighbors classifier in this example)
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
