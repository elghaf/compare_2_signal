import numpy as np


#load the data from the Task folder :
def load_signature_data(file_path):
    with open(file_path, 'r') as file:
        num_lines = int(file.readline().strip())
        signature_data = np.empty((num_lines, 7))
        for i, line in enumerate(file.readlines()):
            line_data = line.strip().split(' ')
            signature_data[i] = [int(x) for x in line_data]
    return signature_data.astype(np.float64)













def calculate_similarity(features1, features2):
    # Initialize the similarity score and total weight
    similarity_score = 0
    total_weight = 0

    # Define the weights for each feature
    weights = {
        'time_duration': 0.5,
        'velocity': 0.3,
        'direction_changes': 0.2,
        'pressure_range': 0.4,
        'pressure_std': 0.6,
        'stroke_count': 0.8,
        'aspect_ratio': 0.2,
        # Add weights for other features here
    }

    # Iterate over the features and calculate the weighted similarity for each feature
    for feature_name in features1:
        if feature_name in weights:
            feature_value1 = features1[feature_name]
            feature_value2 = features2[feature_name]

            # Calculate the similarity for the current feature by taking the absolute difference
            feature_similarity = abs(feature_value1 - feature_value2)

            # Multiply the similarity by the weight for the current feature
            feature_weight = weights[feature_name]
            similarity_score += feature_weight * feature_similarity
            total_weight += feature_weight

    # Calculate the average similarity across all features
    if total_weight > 0:
        similarity_score /= total_weight

    # Calculate the similarity percentage
    similarity_percentage = similarity_score * 100

    return similarity_percentage




#---------- Extract features

def extract_features(signature):
    # Extracting features from signature data that we load
    x_coordinates = signature[:, 0]
    y_coordinates = signature[:, 1]
    timestamps = signature[:, 2]
    button_status = signature[:, 3]
    azimuth = signature[:, 4]
    altitude = signature[:, 5]
    pressure = signature[:, 6]

    # Calculate time duration of our signature :
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

    # Calculate pen inclination
    pen_inclination = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

    # Calculate curvature
    curvature = np.diff(azimuth)

    # Calculate acceleration
    acceleration = np.diff(altitude)

    # Additional features can be added here

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
        'end_point': end_point,
        'pen_inclination': pen_inclination,
        'curvature': curvature,
        'acceleration': acceleration
    }
    return features


# -------------- Compare the signature



def compare_signatures(signature_path1, signature_path2) -> bool:
    # Load and extract features from the first signature
    signature1 = load_signature_data(signature_path1)
    features1 = extract_features(signature1)

    # Load and extract features from the second signature
    signature2 = load_signature_data(signature_path2)
    features2 = extract_features(signature2)

    # Perform similarity analysis using machine learning model or similarity measure
    similarity = calculate_similarity(features1, features2)

    # Determine if the signatures are similar or not
    if similarity <= 1000:
        return True
    else:
        return False




# Print resultat

# path of our data to load
signature_path1 = "Task1//U6S1.TXT"
signature_path2 = "Task1//U6S5.TXT"

# Set the similarity threshold (adjust according to your needs)
threshold = 0.5

# comparaison the signature input
is_similar = compare_signatures(signature_path1, signature_path2)


if is_similar:
    print("The signatures are similar.")
else:
    print("The signatures are not similar.")