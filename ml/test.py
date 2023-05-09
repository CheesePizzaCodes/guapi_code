import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load your high-dimensional dataframe
# df = pd.read_csv('your_data.csv')

# For demonstration purposes, let's create a sample dataframe
import numpy as np
np.random.seed(42)
data = np.random.rand(8000, 50)
df = pd.DataFrame(data, columns=[f'attr_{i}' for i in range(1, 51)])

fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# Add a trace for each combination of axis selections
for x_col in df.columns:
    for y_col in df.columns:
        for z_col in df.columns:
            scatter = go.Scatter3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                mode='markers',
                marker=dict(
                    color=df['attr_4'],
                    colorscale='Viridis',
                    showscale=True
                ),
                visible=False,  # set all traces to invisible by default
                name=f"{x_col}-{y_col}-{z_col}"
            )
            fig.add_trace(scatter)

# Make the first trace visible
fig.data[0].visible = True


# Generate buttons for the dropdown menus
def generate_buttons(df, axis_name):
    buttons = []
    for col in df.columns:
        button = dict(
            args=[{"visible": [False] * len(fig.data)}],
            label=col,
            method="restyle"
        )
        buttons.append(button)

    return buttons





# Define the generate_info_dict function
def generate_info_dict(df, axis_name, x_position, menu_name, active_index):
    buttons = generate_buttons(df, axis_name)
    for i, button in enumerate(buttons):
        button['args'][0]['visible'][i] = True

    return dict(
        buttons=buttons,
        direction='down',
        showactive=True,
        x=x_position,
        xanchor='left',
        y=1.1,
        yanchor='top',
        pad={"r": 10, "t": 10},
        bgcolor='rgba(200,200,200,0.7)',
        type='dropdown',
        font=dict(size=11, color='#000'),
        bordercolor='#000',
        name=menu_name,
        active=active_index
    )

# Define the layout with the dropdown menus
fig.update_layout(
    scene=dict(
        xaxis_title="attr_1",
        yaxis_title="attr_2",
        zaxis_title="attr_3"
    ),
    updatemenus=[
        generate_info_dict(df, "x", 0.1, "X-Axis", 0),
        generate_info_dict(df, "y", 0.4, "Y-Axis", 1),
        generate_info_dict(df, "z", 0.7, "Z-Axis", 2),
        generate_info_dict(df, "marker.color", 1.0, "Color", 3)
    ]
)

fig.show()