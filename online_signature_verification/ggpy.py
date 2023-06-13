import plotly.graph_objects as go

def generate_gauge(value):
    # Define the range for the gauge
    min_value = 300
    max_value = 850

    # Calculate the position of the indicator based on the value
    indicator_position = (value - min_value) / (max_value - min_value)

    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [min_value, max_value]},
            'bar': {'color': 'blue'},
            'bgcolor': 'lightgray',
            'steps': [
                {'range': [min_value, max_value], 'color': 'gray'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    # Set the layout properties
    fig.update_layout(
        title = {'text': f"Gauge Chart ({value})", 'x': 0.5},
        width = 400,
        height = 300,
        font = {'family': "Arial", 'size': 18}
    )

    # Save the chart as an image
    fig.write_image("gauge_image.png")

# Generate the gauge image for a specific value
generate_gauge(832)
