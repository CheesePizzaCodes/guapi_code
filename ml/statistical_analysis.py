from typing import List

import plotly
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

import file_io
import formatting


def main():
    ...


def grouped_summary(df, categ_column, num_column):
    """

    :param df:
    :return:
    """
    return df.groupby(categ_column)[num_column].agg(['count', 'mean', 'median', 'std', 'min', 'max'])


def filter_column_by_values(df: pd.DataFrame, col_name: str, value: str, reverse: bool = False) -> pd.DataFrame:
    data[col_name] = data[col_name].fillna('')

    if reverse:
        return df[~df[col_name].str.contains(value)]
    return df[df[col_name].str.contains(value)]


if __name__ == '__main__':
    main()
    col = "Displacement [kg]"
    data = file_io.read_formatted_data('final')

    cat_data = filter_column_by_values(data, 'Hull Type', 'Cata')

    other_data = filter_column_by_values(data, 'Hull Type', 'Cata', reverse=True)

    cat_desc = cat_data['Displacement [kg]'].describe()
    monoh_desc = other_data['Displacement [kg]'].describe()




    hist1 = go.Histogram(x=cat_data[col], opacity=0.75, name='catamaran')
    hist2 = go.Histogram(x=other_data[col], opacity=0.75, name='monohull')
    dataa = [hist1, hist2]
    layout = go.Layout(title='Superimposed Histogram', barmode='overlay', xaxis=dict(title='Desplazamiento [kg]'),
                       yaxis=dict(title='Frecuencia'))

    fig = go.Figure(data=dataa[::-1], layout=layout)
    fig.show()
    ...
