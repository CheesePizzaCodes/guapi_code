from typing import List, Tuple

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

import file_io
import formatting


def scatter_3d_p(df: pd.DataFrame, classes: str = None):
    col_names = data.columns[0:3]
    fig = px.scatter_3d(data_frame=df,
                        x=col_names[0], y=col_names[1], z=col_names[2],
                        color=classes,
                        )
    fig.update_traces(marker_size=2)
    fig = modify_layout(fig, df)  # makes layout interactive
    return fig


def modify_layout(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """
    Modifies the layout so that it is interactive
    :param fig:
    :param df:
    :return: interactive figure
    """
    fig.update_layout(
        scene=dict(
            xaxis_title='attr_1',
            yaxis_title='attr_2',
            zaxis_title='attr_3'
        ),

        # Add buttons
        updatemenus=[generate_dropdown_info(axis_id=0, df=df, x_position=0.1),
                     generate_dropdown_info(axis_id=1, df=df, x_position=0.4),
                     generate_dropdown_info(axis_id=2, df=df, x_position=0.7),
                     # generate_info_dict(opt=3, df=df, x_position=1.0)
                     ]
    )

    return fig


def generate_dropdown_info(axis_id: int, df: pd.DataFrame, x_position) -> dict:
    """
    Creates info to pass as argument to the restyle method
    :param x_position:
    :param axis_id:
    :param df:
    :return:
    """
    opts = ['x', 'y', 'z', 'marker.color']
    axis_name = opts[axis_id]
    menu_name = f'{axis_name.upper()}-Axis'
    # menu_name = f'scene.{axis_name}axis.title'
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
        update_dict = {axis_title_name: [df[col].values]}
        # Layout
        update_layout_dict = {axis_title_name: col} if axis_title_name else {}  # New
        # update_layout_dict = {axis_title_name: col}  # Old
        buttons.append(dict(
            # args=[update_dict, {'trace_indices': [0]}, update_layout_dict],
            args=[update_dict, {}, update_layout_dict],
            label=col,
            method='restyle'
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
    data = file_io.load_formatted_data('final')
    fig2 = scatter_3d_p(data, 'Rigging Type')
    fig2.show()
    # fig2.write_html('./out_data/html/visualization_rigging.html')
    ...

