from typing import List, Tuple

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from sklearn.preprocessing import LabelEncoder

import file_io
import formatting

le = LabelEncoder()


#
# def scatter_3d(df: pd.DataFrame, col_names: List[str], classes: str = None):
#     """
#     Obsolete.
#     :param df:
#     :param col_names:
#     :param classes:
#     :return:
#     """
#     f: plt.Figure = plt.figure()
#     a: plt.Axes = f.add_subplot(projection='3d')
#     label_names = df[classes].values
#     label_names_unique = np.unique(label_names)
#     le.fit(list(set(label_names)))
#     encoding = le.transform(label_names)
#     df_copy = df.copy()
#     for label in label_names_unique:
#         df = df.loc[df[classes] == label]  # select correct class
#         df = df[col_names]
#         # data_slice = data_slice[col_names]
#         a.scatter(*df.values.T.astype('float'), label=label)
#         df = df_copy
#
#     a.set_xlabel(cols[0])
#     a.set_ylabel(cols[1])
#     a.set_zlabel(cols[2])
#     box = a.get_position()
#     a.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#     # Put a legend to the right of the current axis
#     a.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
#     return f, a


def scatter_3d_p(df: pd.DataFrame, col_names: List[str], classes: str = None):
    fig = px.scatter_3d(data_frame=df,
                        x=col_names[0], y=col_names[1], z=col_names[2],
                        color=classes,
                        )
    fig.update_traces(marker_size=2)
    fig = modify_layout(fig, df)
    return fig


def modify_layout(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    fig.update_layout(
        scene=dict(
            xaxis_title='attr_1',
            yaxis_title='attr_2',
            zaxis_title='attr_3'
        ),
        updatemenus=[generate_dropdown_info(active_index=0, df=df, x_position=0.1),
                     generate_dropdown_info(active_index=1, df=df, x_position=0.4),
                     generate_dropdown_info(active_index=2, df=df, x_position=0.7),
                     # generate_info_dict(opt=3, df=df, x_position=1.0)
                     ]
    )

    return fig


def generate_dropdown_info(active_index: int, df: pd.DataFrame, x_position) -> dict:
    """
    Creates info to pass as argument to the restyle method
    :param x_position:
    :param active_index:
    :param df:
    :return:
    """
    opts = ['x', 'y', 'z', 'marker.color']
    axis_name = opts[active_index]
    menu_name = f'{axis_name.upper()}-Axis'
    menu_name = f'scene.{axis_name}axis.title'
    nums, cats = extract_col_names_by_dtype(df)  # here is a bug

    if active_index in (0, 1, 2):
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
        active=active_index
    )
    #
    # return dict(
    #     buttons=list([
    #         dict(
    #             args=[axis_name, df[num]],
    #             label=num,
    #             method='update'
    #         ) for num in nums
    #     ]),
    #     direction='down',
    #     showactive=True,
    #     x=0,
    #     xanchor='left',
    #     y=x_pos,
    #     yanchor='top',
    #     pad={"r": 10, "t": 10},
    #     bgcolor='rgba(200,200,200,0.7)',
    #     type='dropdown',
    #     active=opt,
    #     font=dict(size=11, color='#000'),
    #     bordercolor='#000',
    #     name=f'{axis_name.upper()}-Axis'
    # )


def create_buttons(axis_title_name, df, columns):
    buttons = []
    for col in columns:
        # Values
        update_dict = {axis_title_name: [df[col].values]}
        # Layout
        update_layout_dict = {axis_title_name: col} if axis_title_name else {}  # New
        # update_layout_dict = {axis_title_name: col}  # Old
        buttons.append(dict(
            args=[update_dict, {'trace_indices': [0]}, update_layout_dict],
            label=col,
            method='update'
        ))
    return buttons


def extract_col_names_by_dtype(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numerical_col_names = [col_name for col_name in df.columns if is_numeric_dtype(df[col_name])]
    categorical_col_names = [col_name for col_name in df.columns if is_string_dtype(df[col_name])]
    categorical_col_names.remove('HP')
    categorical_col_names.remove('url')
    return numerical_col_names, categorical_col_names


if __name__ == '__main__':
    # data = file_io.load_scrape_data('final_4')
    data = file_io.load_formatted_data('final')
    # cols = ['LOA [m]', 'S.A. / Displ.', 'Beam [m]']
    cols = ['Displacement [kg]', 'Hull Type', 'Beam [m]']

    # fig2 = scatter_3d_p(data, cols, 'Hull Type')
    fig2 = scatter_3d_p(data, cols, 'Hull Type')
    fig2.show()
    ...
