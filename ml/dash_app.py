from threading import Timer
import webbrowser

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

import file_io
from file_io import read_formatted_data

# Assuming you have a dataframe 'df' with 50 attributes and a column label for coloring
SIZE = 3
label = 'Hull Type'

# Create a DataFrame with 50 attributes
# df = pd.DataFrame(np.random.rand(8000, 50), columns=[f'attr_{i}' for i in range(50)])

df = file_io.read_formatted_data('final')  # TODO make this a global constant

# Add a label column for coloring. Let's assume there are 10 different labels.

# Create a Dash app
app = dash.Dash(__name__)

# Create the initial 3D scatter plot using Plotly Express
initial_fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2], color=label, ).update_traces(
    marker=dict(size=SIZE))

app.layout = html.Div([
    html.Div([
        html.Label('Select X Axis'),
        dcc.Dropdown(
            id='x-axis',
            options=[{'label': col, 'value': col} for col in df.columns if col != label],
            value=df.columns[0]
        ),
        html.Label('Select Y Axis'),
        dcc.Dropdown(
            id='y-axis',
            options=[{'label': col, 'value': col} for col in df.columns if col != label],
            value=df.columns[1]
        ),
        html.Label('Select Z Axis'),
        dcc.Dropdown(
            id='z-axis',
            options=[{'label': col, 'value': col} for col in df.columns if col != label],
            value=df.columns[2]
        ),
    ], style={'width': '25%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='scatter-3d', figure=initial_fig)
    ], style={'width': '75%', 'display': 'inline-block'}),
])


@app.callback(
    Output('scatter-3d', 'figure'),
    [Input('x-axis', 'value'), Input('y-axis', 'value'), Input('z-axis', 'value')]
)
def update_scatter(x_col, y_col, z_col):
    updated_fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=label).update_traces(marker=dict(size=SIZE))
    return updated_fig


def open_browser():
    webbrowser.open('http://127.0.0.1:8050/')


if __name__ == '__main__':
    app.run_server(debug=True)
    Timer(1, open_browser).start()
