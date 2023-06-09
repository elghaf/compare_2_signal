import numpy as np
from sklearn.preprocessing import StandardScaler

def load_signature_data(file_path):
    with open(file_path, 'r') as file:
        num_lines = int(file.readline().strip())
        signature_data = np.empty((num_lines, 7))
        for i, line in enumerate(file.readlines()):
            line_data = line.strip().split(' ')
            signature_data[i] = [int(x) for x in line_data]
    return signature_data.astype(np.float64)

def extract_features(signature):
    time_stamps = signature[:, 2]
    pressure = signature[:, 6]
    x_coordinates = signature[:, 0]
    y_coordinates = signature[:, 1]
    pen_inclination = np.arctan2(y_coordinates[-1] - y_coordinates[0], x_coordinates[-1] - x_coordinates[0])
    dx = np.gradient(x_coordinates)
    dy = np.gradient(y_coordinates)
    dt = np.gradient(time_stamps)
    speed = np.sqrt(dx**2 + dy**2) / dt
    acceleration = np.gradient(speed) / dt
    curvature = np.gradient(np.arctan2(dy, dx)) / dt
    max_pressure = np.max(pressure)
    min_pressure = np.min(pressure)
    mean_pressure = np.mean(pressure)
    duration = time_stamps[-1] - time_stamps[0]
    average_speed = np.mean(speed)
    jerk = np.gradient(acceleration) / dt
    average_jerk = np.mean(jerk)
    jerk_change = np.gradient(jerk) / dt
    acceleration_change = np.gradient(acceleration) / dt
    speed_change = np.gradient(speed) / dt
    acceleration_ratio = acceleration / speed
    speed_ratio = speed / duration
    features = np.array([speed[-1], acceleration[-1], curvature[-1], max_pressure, min_pressure, mean_pressure, average_speed, pen_inclination, average_jerk, jerk_change[-1], acceleration_change[-1], speed_change[-1], acceleration_ratio[-1], speed_ratio])
    
    # Reshape the features to match the expected shape of StandardScaler
    features = features.reshape(-1, 1)
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features).flatten()
    return features

def compare_signatures(signature1, signature2):
    features1 = extract_features(signature1)
    features2 = extract_features(signature2)
    distance = np.linalg.norm(features1 - features2)
    return distance



signature1 = load_signature_data("Task1//U1S1.TXT")
signature2 = load_signature_data("Task1//U1S2.TXT")


distance = compare_signatures(signature1, signature2)
print("Distance between the signatures:", distance)
