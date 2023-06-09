import os

# Function to read a signature file
def read_signature_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        total_lines = int(lines[0])
        signature_data = []
        for line in lines[1:]:
            data = line.strip().split(',')
            signature_data.append({
                'x': float(data[0]),
                'y': float(data[1]),
                'timestamp': int(data[2]),
                'button_status': int(data[3]),
                'azimuth': float(data[4]),
                'altitude': float(data[5]),
                'pressure': float(data[6])
            })
        return total_lines, signature_data

# Function to process a signature folder
def process_signature_folder(folder_path):
    signature_files = [file for file in os.listdir(folder_path) if file.startswith('U') and file.endswith('.txt')]
    for file in signature_files:
        file_path = os.path.join(folder_path, file)
        total_lines, signature_data = read_signature_file(file_path)
        
        # Perform further processing or analysis on the signature data
        # Example: Calculate the average pressure
        total_pressure = sum(point['pressure'] for point in signature_data)
        average_pressure = total_pressure / total_lines
        print(f"Average pressure for {file}: {average_pressure}")
        
        # Example: Classify signature based on button status
        button_statuses = [point['button_status'] for point in signature_data]
        if all(status == 0 for status in button_statuses):
            print(f"Signature {file} is classified as pen-up")
        elif all(status == 1 for status in button_statuses):
            print(f"Signature {file} is classified as pen-down")
        else:
            print(f"Signature {file} has mixed pen-up and pen-down")
        
        # Example: Check if the signature is a single stroke or multiple strokes
        stroke_count = 0
        prev_button_status = button_statuses[0]
        for status in button_statuses:
            if status != prev_button_status:
                stroke_count += 1
                prev_button_status = status
        if stroke_count == 0:
            print(f"Signature {file} is a single stroke")
        else:
            print(f"Signature {file} has {stroke_count+1} strokes")
        
        # Example: Calculate the total time taken for the signature
        start_time = signature_data[0]['timestamp']
        end_time = signature_data[-1]['timestamp']
        total_time = end_time - start_time
        print(f"Total time for {file}: {total_time} milliseconds")
        
        # Example: Calculate the average speed of the signature
        total_distance = 0
        prev_point = signature_data[0]
        for point in signature_data[1:]:
            distance = ((point['x'] - prev_point['x'])**2 + (point['y'] - prev_point['y'])**2)**0.5
            total_distance += distance
            prev_point = point
        average_speed = total_distance / total_time
        print(f"Average speed for {file}: {average_speed} units/ms")
        
        # Example: Check if the signature is clockwise or counterclockwise
        azimuth_values = [point['azimuth'] for point in signature_data]
        avg_azimuth = sum(azimuth_values) / total_lines
        if avg_azimuth > 0:
            print(f"Signature {file} is clockwise")
        else:
            print(f"Signature {file} is counterclockwise")
        
        # Add your own code here to perform more analysis or classification on the signature data

# Main function
def main():
    folder_path = 'Task1'  # Replace with the actual folder path
    process_signature_folder(folder_path)

if __name__ == '__main__':
    main()
