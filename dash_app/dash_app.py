
from enum import Enum
from threading import Timer
import webbrowser

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np


# Assuming you have a dataframe 'df' with 50 attributes and a column label for coloring
MARKER_SIZE = 3

label = 'Hull Type'

class Label(Enum):
    HULL_TYPE = "Hull Type"

# Create a DataFrame with 50 attributes


df = pd.read_json('./dash_app/data/final.json')  # TODO make this a global constant
df_string_view = df.select_dtypes(include='object')
df_number_view = df.select_dtypes(include=[np.float64, ])
# Add a label column for coloring. Let's assume there are 10 different labels.

# Create a Dash app
app = dash.Dash(__name__)

# Create the initial 3D scatter plot using Plotly Express
initial_fig = px.scatter_3d(df,
                            x=df_number_view.columns[0],
                            y=df_number_view.columns[1],
                            z=df_number_view.columns[2],
                            color=df_string_view.columns[0], ).update_traces(marker=dict(size=MARKER_SIZE))

app.layout = html.Div([
    ### Graph div
    html.Div(
        [dcc.Graph(id='scatter-3d', figure=initial_fig), ],
        style={'width': '85%', 'display': 'inline-block', }),

    html.Div([
        html.Label('Select X Axis'),
        dcc.Dropdown(
            id='x-axis',
            options=[{'label': col, 'value': col} for col in df_number_view.columns],
            value=df_number_view.columns[0]),
        html.Label('Select Y Axis'),
        dcc.Dropdown(
            id='y-axis',
            options=[{'label': col, 'value': col} for col in df_number_view.columns],
            value=df_number_view.columns[1]),
        html.Label('Select Z Axis'),
        dcc.Dropdown(
            id='z-axis',
            options=[{'label': col, 'value': col} for col in df_number_view.columns],
            value=df_number_view.columns[2]),
        html.Label('Select Color'),
        dcc.Dropdown(
            id='color',
            options=[{'label': col, 'value': col} for col in df_string_view.columns],
            value=df_string_view.columns[0]),
    ], style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top'}),

],
    # style={'height': '100vh', 'width': '100%'}
)


@app.callback(
    Output('scatter-3d', 'figure'),
    [Input('x-axis', 'value'), Input('y-axis', 'value'), Input('z-axis', 'value'), Input('color', 'value')])
def update_scatter(x_col, y_col, z_col, color_col):
    updated_fig = px.scatter_3d(df,
                                x=x_col, y=y_col, z=z_col,
                                color=color_col).update_traces(marker=dict(size=MARKER_SIZE))
    return updated_fig





if __name__ == '__main__':
    host='0.0.0.0'

    app.run_server(debug=True)
                   # , host='0.0.0.0')

