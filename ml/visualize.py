from typing import List

# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
# import numpy as np
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
    fig = px.scatter_3d(df,
                        x=col_names[0], y=col_names[1], z=col_names[2],
                        color=classes,
                        )
    fig.update_traces(marker_size=2)
    return fig



if __name__ == '__main__':
    data = file_io.load_data('./out_data/finalisimo.json')
    data = formatting.format_data(data)
    cols = ['LOA [m]', 'S.A./Disp.', 'Beam [m]']
    fig2 = scatter_3d_p(data, cols, 'Construction')
    fig2.show()
    ...
