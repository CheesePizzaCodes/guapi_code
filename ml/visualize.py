from typing import List, Tuple

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

import file_io
import formatting

def scatter_2d_p(df: pd.DataFrame, classes: str = None) -> go.Figure:
    labeled_traces = generate_labeled_traces_2d(df, classes)
    fig = go.Figure(data=labeled_traces)
    fig = modify_layout_2d(fig, df)  # makes layout interactive
    return fig
def scatter_3d_p(df: pd.DataFrame, classes: str = None) -> go.Figure:
    labeled_traces = generate_labeled_traces_3d(df, classes)
    fig = go.Figure(data=labeled_traces)
    fig = modify_layout_3d(fig, df)  # makes layout interactive
    return fig

def generate_hsl_colors(n_colors, saturation=0.7, lightness=0.6):
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        hsl_color = f'hsl({hue * 360:.2f}, {saturation * 100:.2f}%, {lightness * 100:.2f}%)'
        colors.append(hsl_color)
    return colors

def generate_labeled_traces_2d(df, classes):

    num, _ = extract_col_names_by_dtype(df)
    x, y = num[0:2]

    unique_labels = df[classes].unique()
    color_scale = generate_hsl_colors(len(unique_labels))
    label_to_color = {label: color for (label, color) in zip(unique_labels, color_scale)}
    # colors = df[classes].apply(lambda x: label_to_color[x])
    traces = []
    for label, color in label_to_color.items():
        label_df = df[df[classes] == label]  # portion of the data of the correct label
        trace = go.Scatter(
            x=label_df[x],
            y=label_df[y],
            mode='markers',
            marker=dict(
                color=color,
                size=5,
            ),
            text=[f'{x}: {x}<br>{y}: {y}<br>{classes}: {label}'
                  for x, y in zip(label_df[x], label_df[y])],
            hoverinfo='text',
            name=label,
        )
        traces.append(trace)
    return tuple(traces)
def generate_labeled_traces_3d(df, classes):

    num, _ = extract_col_names_by_dtype(df)
    x, y, z = num[0:3]

    unique_labels = df[classes].unique()
    color_scale = generate_hsl_colors(len(unique_labels))
    label_to_color = {label: color for (label, color) in zip(unique_labels, color_scale)}
    # colors = df[classes].apply(lambda x: label_to_color[x])
    traces = []
    for label, color in label_to_color.items():
        label_df = df[df[classes] == label]  # portion of the data of the correct label
        trace = go.Scatter3d(
            x=label_df[x],
            y=label_df[y],
            z=label_df[z],
            mode='markers',
            marker=dict(
                color=color,
                size=5,
            ),
            text=[f'{x}: {_x}<br>{y}: {_y}<br>{z}: {_z}<br>{classes}: {label}'
                  for _x, _y, _z in zip(label_df[x], label_df[y], label_df[z])],
            hoverinfo='text',
            name=label,
        )
        traces.append(trace)
    return tuple(traces)

def modify_layout_2d(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """
    Modifies the layout so that it is interactive
    :param fig:
    :param df:
    :return: interactive figure
    """
    fig.update_traces(marker_size=2)
    fig.update_layout(
        scene=dict(
            xaxis_title='attr_1',
            yaxis_title='attr_2',
        ),

        # Add buttons
        updatemenus=[
            generate_dropdown_info(axis_id=0, df=df, x_position=0.1),
            generate_dropdown_info(axis_id=1, df=df, x_position=0.4),
            # generate_dropdown_info(axis_id=2, df=df, x_position=0.7),
            # generate_dropdown_info(axis_id=3, df=df, x_position=1.0)
        ]
    )

    return fig

def modify_layout_3d(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """
    Modifies the layout so that it is interactive
    :param fig:
    :param df:
    :return: interactive figure
    """
    fig.update_traces(marker_size=2)
    fig.update_layout(
        scene=dict(
            xaxis_title='attr_1',
            yaxis_title='attr_2',
            zaxis_title='attr_3'
        ),

        # Add buttons
        updatemenus=[
            generate_dropdown_info(axis_id=0, df=df, x_position=0.1),
            generate_dropdown_info(axis_id=1, df=df, x_position=0.4),
            generate_dropdown_info(axis_id=2, df=df, x_position=0.7),
            # generate_dropdown_info(axis_id=3, df=df, x_position=1.0)
        ]
    )

    return fig


def generate_dropdown_info(axis_id: int, df: pd.DataFrame, x_position) -> dict:
    """
    Creates info to pass as argument to the restyle method
    :param x_position:
    :param axis_id: Axis of the plot to be controlled by this dropdown
    :param df: Data
    :return:
    """
    opts = ['x', 'y', 'z', 'marker.color']
    axis_name = opts[axis_id]
    # menu_name = f'{axis_name.upper()}-Axis'
    menu_name = f'scene.{axis_name}axis.title'
    nums, cats = extract_col_names_by_dtype(df)  # here is a bug

    if axis_id in (0, 1, 2):
        buttons = create_buttons(axis_name, df, nums)
    else:
        buttons = create_buttons(axis_name, df, cats)

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
        active=axis_id
    )


def create_buttons(axis_title_name, df, columns):
    buttons = []
    for col in columns:
        # Values
        # update_dict = {axis_title_name: [df[col].values] * data.shape[1]}
        update_dict = {axis_title_name: [df[col].values]}
        # Layout
        update_layout_dict = {axis_title_name: col} if axis_title_name else {}  # New
        # update_layout_dict = {axis_title_name: col}  # Old
        buttons.append(dict(
            # args=[update_dict, {'trace_indices': [0]}, update_layout_dict],
            # args=[update_dict, {"visible": [True] * df.shape[1]}, update_layout_dict],
            args=[update_dict, {}, update_layout_dict],
            label=col,
            method='update',
        ))
    return buttons


def extract_col_names_by_dtype(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numerical_col_names = [col_name for col_name in df.columns if is_numeric_dtype(df[col_name])]
    categorical_col_names = [col_name for col_name in df.columns if is_string_dtype(df[col_name])]
    numerical_col_names.append('HP')
    categorical_col_names.remove('HP')
    categorical_col_names.remove('url')
    return numerical_col_names, categorical_col_names


if __name__ == '__main__':
    data = file_io.read_formatted_data('final')
    fig2 = scatter_3d_p(data, 'Construction Type')
    fig2.show()
    inp = input('desired file name:')
    fig2.write_html(f'./out_data/html/{inp}.html')
    ...
