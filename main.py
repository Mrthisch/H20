# Python modules
# import math  # Mathematical standard computations
import numpy as np  # Mathematical array computations & data vectorization
import pandas as pd  # Dataframe creation, CSV format computations
import plotly  # Graphical, data visualizations
import plotly.express
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import dash  # Dash webb app designer
from dash import html, dcc, dash_table, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
# from datetime import datetime
import inspect
import base64
#import platform
import os
#import sys
# import pkg_resources
# import socket
# import errno
# import psutil
# import logging
# import cdsapi
# import cbsodata
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Define file paths in code as absolute reference to source code file

Pname = 'P10'
Pname_CODE = '3148'
mod_v = 'v0.1'

def import_data():
    df_labels = pd.read_csv('labels.csv')
    df_vars = df_labels['label']
    variables = df_vars.tolist()

    df_vars_context = df_labels[df_labels['origin'] == 'Geographic context'].reset_index()
    df_vars_context_values = df_vars_context.loc[:, 'Aa en Hunze':'Weststellingwerf']
    variables_context = df_vars_context['label'].tolist()

    df_data = pd.read_csv('data.csv')

    # df_params = pd.read_csv('parameters.csv', header=None)
    # df_params = df_params.replace('#DIV/0!', 0).T
    # df_params.columns = df_params.iloc[0]
    # df_params = df_params[1:]
    # df_params = df_params.reset_index(drop=True)

    # municipalities = df_params.columns.values.tolist()
    # municipalities = municipalities[1:]

    df_weights = pd.read_csv('weights.csv')
    df_weights = df_weights.replace('#DIV/0!', 0)
    df_weights = df_weights.reset_index(drop=True)

    params = df_weights['Parameter'].tolist()

    ## Results
    Results_T1 = pd.DataFrame()
    for i in df_labels.loc[:, 'Aa en Hunze':'Weststellingwerf']:
        result = df_labels['weights_1'].astype(float) * df_labels[i].astype(float)
        result.name = i  # Set the name of the series as the column name
        #Results_T1 = Results_T1.append(result)
        Results_T1 = pd.concat([Results_T1, result], axis=1)

    Results_T1 = Results_T1.T
    Results_T1.columns = variables
    Score_T1 = Results_T1.sum()

    Results_T2 = pd.DataFrame()
    for i in df_labels.loc[:, 'Aa en Hunze':'Weststellingwerf']:
        result = df_labels['weights_2'].astype(float) * df_labels[i].astype(float)
        result.name = i  # Set the name of the series as the column name
        Results_T2 = pd.concat([Results_T2, result], axis=1)

    Results_T2 = Results_T2.T
    Results_T2.columns = variables
    Score_T2 = Results_T2.sum()

    logo = 'logo.png'
    def b64_image(logo):
        with open(logo, 'rb') as f:
            image = f.read()
        return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

    globals().update(locals())
import_data()

## Plot functions
def typologies():
    global fig_typ1, fig_typ2
    # Typologies sunburst charts
    T1 = {   'Dimensions': np.array(df_labels['dimension']),
             'Subcategories': np.array(df_labels['category']),
             'Parameters': np.array(df_labels['label']),
             'Weights': np.array(df_labels['weights_1']),
             }

    typ1_data = pd.DataFrame(T1)
    #colorMapSubset = dict(zip(strat_data.RGBColors, strat_data.RGBColors))
    fig_typ1 = plotly.express.sunburst(typ1_data,
                                       path=['Dimensions', 'Subcategories','Parameters'],
                                       values='Weights',
                                       height=850,
                                       )
    fig_typ1.update_traces(textinfo='label+percent parent')
    fig_typ1.update_traces(plotly.graph_objects.Sunburst(hovertemplate='%{value}'),
                            selector=dict(type='sunburst'))
    fig_typ1.update_layout(
        title={
            'text': 'Typology 1: The Ecovillage',
            'y': 0.9,  # Adjust the y position of the title
            'x': 0.5,  # Adjust the x position of the title
            'xanchor': 'center',  # Set the title's x anchor to the center
            'yanchor': 'top'  # Set the title's y anchor to the top
        }
    )

    T2 = {'Dimensions': np.array(df_labels['dimension']),
          'Subcategories': np.array(df_labels['category']),
          'Parameters': np.array(df_labels['label']),
          'Weights': np.array(df_labels['weights_2']),
          }

    typ2_data = pd.DataFrame(T2)
    #colorMapSubset = dict(zip(strat_data.RGBColors, strat_data.RGBColors))
    fig_typ2 = plotly.express.sunburst(typ2_data,
                                       path=['Dimensions', 'Subcategories','Parameters'],
                                       values='Weights',
                                       height=850,
                                       )
    fig_typ2.update_traces(textinfo='label+percent parent')
    fig_typ2.update_traces(plotly.graph_objects.Sunburst(hovertemplate='%{value}'),
                            selector=dict(type='sunburst'))
    fig_typ2.update_layout(
        title={
            'text': 'Typology 2,3,4: The Econeighbourhood',
            'y': 0.9,  # Adjust the y position of the title
            'x': 0.5,  # Adjust the x position of the title
            'xanchor': 'center',  # Set the title's x anchor to the center
            'yanchor': 'top'  # Set the title's y anchor to the top
        }
    )

def model_results():
    df_MAP0 = pd.DataFrame()
    for i in range(len(df_data['municipality'])):
        MAP0_string = ''
        for j in range(len(variables_context)):
            MAP0_string += variables_context[j] + ': ' + str(round(df_vars_context_values.loc[j][i] * 100, 1)) + '% <br>'

        df_MAP0_string = pd.DataFrame([MAP0_string])  # Wrapping MAP0_string in a list
        df_MAP0 = pd.concat([df_MAP0, df_MAP0_string], axis=0)

    MAP_0 = go.Figure(data=plotly.graph_objects.Scattergeo(
        lon=df_data['longitude'],
        lat=df_data['latitude'],
        hovertemplate='%{text}<extra><p><b>Normalised geographic context score:</b></p><br>'
                      '%{customdata}'
                      '</extra>',
        text=df_data['municipality'],
        marker=dict(
            size=df_vars_context_values.sum(),
            sizeref=0.3,
            opacity=0.9,
            color='green',
        ),
        customdata=df_MAP0,
    )
    )

    MAP_0.update_layout(
        height=750,
        title='P10 Areas',
        title_x=0.5,  # Adjust the x position of the title
        # geo_scope='netherlands',
        geo=dict(
            center=dict(lon=5.2913, lat=52.1326),
            projection_scale=70,
            showland=True,
            landcolor='rgb(212, 212, 212)',
            countrycolor='rgb(255, 255, 255)'
        ),
        plot_bgcolor="rgb(212, 212, 212)",

    )
    MAP_0.update_geos(
        projection_type="mercator",
        lataxis_showgrid=True, lonaxis_showgrid=True,
        resolution=50,  # Increase the resolution value for a more detailed map
        showcountries=True,
        countrycolor="rgb(255, 255, 255)",
        showcoastlines=True,
        coastlinecolor="rgb(128, 128, 128)",
    )

    # df_MAP1 = pd.DataFrame()
    # for i in range(len(df_data['municipality'])):
    #     MAP1_string = ''
    #     for j in range(len(variables)):
    #         MAP1_string += variables[j] + ': ' + str(round(Results_T1.loc[j][i], 1)) + '<br>'
    #
    #     df_MAP1_string = pd.DataFrame([MAP1_string])  # Wrapping MAP0_string in a list
    #     df_MAP1 = pd.concat([df_MAP1, df_MAP1_string], axis=0)
    #     print(df_MAP1_string)

    MAP_1 = go.Figure(data=plotly.graph_objects.Scattergeo(
        lon=df_data['longitude'],  # Longitude of Amsterdam
        lat=df_data['latitude'],  # Latitude of Amsterdam
        hovertemplate="%{text}<extra><p><b>Factor results:</b></p><br>"
                      #'%{customdata}'
                      '</extra>',
        text=df_data['municipality'],
        marker=dict(
            size=Score_T1,  # Marker size based on land price data
            sizemode='area',  # Use area scaling for marker size
            sizeref=0.3,  # Adjust the size scaling factor as needed
            color=Score_T1,  # Marker color based on land price data
            colorscale='Jet',  # Choose a colorscale for the marker colors
            reversescale=True,  # Reverse the colorscale if desired
            opacity=0.9,
            colorbar=dict(
                title='Viability',
                tickvals=[Score_T1.min(), Score_T1.max()],  # Set the low and high values as tick values
                ticktext=['Low', 'High'],  # Set the labels for the low and high values
                lenmode = 'fraction',
                len = 0.7,  # Adjust the length of the colorbar
                x = 0.9,  # Adjust the x position of the colorbar
                y = 0.5  # Adjust the y position of the colorbar
            )
        ),
        #customdata = df_MAP1,
    ))

    MAP_1.update_layout(
        height=750,
        title='Results P10 vs. Typology 1: Ecovillages',
        title_x=0.5,  # Adjust the x position of the title
        #geo_scope='netherlands',
        geo=dict(
            center=dict(lon=5.2913, lat=52.1326),
            projection_scale=70,
            showland=True,
            landcolor='rgb(212, 212, 212)',
            countrycolor='rgb(255, 255, 255)'
        ),
        plot_bgcolor="rgb(212, 212, 212)",

    )
    MAP_1.update_geos(
        projection_type="mercator",
        lataxis_showgrid=True, lonaxis_showgrid=True,
        resolution=50,  # Increase the resolution value for a more detailed map
        showcountries=True,
        countrycolor="rgb(255, 255, 255)",
        showcoastlines=True,
        coastlinecolor="rgb(128, 128, 128)",
        )

    fig_1 = go.Figure(go.Sankey(
        arrangement='freeform',
        valueformat=".2f",
        valuesuffix="",
        node=dict(
            #pad=10,
            #thickness=5,
            label=df_labels['label'],
            #x=[0.2, 0.1, 0.5, 0.7, 0.3, 0.5],
            #y=[0.7, 0.5, 0.2, 0.4, 0.2, 0.3],
        ),
        link=dict(
            #arrowlen=15,
            source=df_labels['source'],
            target=df_labels['target'],
            value=df_labels['Aa en Hunze'],
            #color=df_labels['color'].astype(str)
        ),
    ))

    fig_1.update_layout(
        title='Aa en Hunze vs. Custom typology 1',
        title_x=0.5,
        width = 1800,
        height=1000,
        margin=dict(l=0),
    )

    MAP_2 = go.Figure(data=plotly.graph_objects.Scattergeo(
        lon=df_data['longitude'],  # Longitude of Amsterdam
        lat=df_data['latitude'],  # Latitude of Amsterdam
        hovertemplate="%{text}<extra>"
                      "</extra>", #<br><a href=%{marker.link}>Click here for more info</a>",
        text=df_data['municipality'],
        marker=dict(
            size=Score_T2,  # Marker size based on land price data
            sizemode='area',  # Use area scaling for marker size
            sizeref=0.3,  # Adjust the size scaling factor as needed
            color=Score_T2,  # Marker color based on land price data
            colorscale='Jet',  # Choose a colorscale for the marker colors
            reversescale=True,  # Reverse the colorscale if desired
            opacity=0.9,
            colorbar=dict(
                title='Viability',
                tickvals=[Score_T2.min(), Score_T2.max()],  # Set the low and high values as tick values
                ticktext=['Low', 'High'],  # Set the labels for the low and high values
                lenmode = 'fraction',
                len = 0.7,  # Adjust the length of the colorbar
                x = 0.9,  # Adjust the x position of the colorbar
                y = 0.5  # Adjust the y position of the colorbar
            )
        )
    ))

    MAP_2.update_layout(
        height=750,
        title='Results P10 vs. Typology 2,3,4: Econeighbourhoods',
        title_x=0.5,  # Adjust the x position of the title
        #geo_scope='netherlands',
        geo=dict(
            center=dict(lon=5.2913, lat=52.1326),
            projection_scale=70,
            showland=True,
            landcolor='rgb(212, 212, 212)',
            countrycolor='rgb(255, 255, 255)'
        ),
        plot_bgcolor="rgb(212, 212, 212)",

    )
    MAP_2.update_geos(
        projection_type="mercator",
        lataxis_showgrid=True, lonaxis_showgrid=True,
        resolution=50,  # Increase the resolution value for a more detailed map
        showcountries=True,
        countrycolor="rgb(255, 255, 255)",
        showcoastlines=True,
        coastlinecolor="rgb(128, 128, 128)",
        )

    globals().update(locals())

# APP content
content = html.Div(id="page-content",
                   style={"margin-left": "18rem", "margin-right": "2rem",
                          "padding": "2rem 1rem",
                          "font_family": 'Serif'
                          }
                   )

# Sidebar navigation
sidebar = html.Div(
    [

        html.A(
            html.Img(
                src=b64_image(logo),
                style={
                    'width': '100%',
                    'marginTop': 0,
                    'marginLeft': 0
                }
            ),
            href="/",
        ),
        html.H2(dbc.NavLink(Pname, id='sidebar-title', href="/",
                            style={"color": "Black",
                                   'fontSize': 26,
                                   'text-align': 'center',
                                   }
                            ),
                {'fontWeight': "bold",
                 }),
        html.Hr(),
        html.P(""),
        html.P("Input", className="lead"),

        dbc.Nav(
            [
                dbc.NavLink("Locations", href="/" + Pname_CODE + "/project_info",
                            id="P_i", n_clicks=0,
                            class_name="page-link",
                            style={"color": "dodgerblue"}),
                dbc.NavLink("Typologies", href="/" + Pname_CODE + "/typologies",
                            class_name="page-link",
                            style={"color": "dodgerblue"}),
                html.P(""),
                html.P("Calculations", className="lead"),
                dbc.NavLink("Model", href="/" + Pname_CODE + "/model",
                            class_name="page-link",
                            style={"color": "dodgerblue"}),
                dbc.NavLink("Results", href="/" + Pname_CODE + "/results",
                            class_name="page-link",
                            style={"color": "dodgerblue"}),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.P(inspect.currentframe().f_code.co_filename,
               style={"color": "lightgrey",
                      'fontSize': 6,
                      'fontWeight': "italic"
                      }
               ),
        html.P(mod_v,
               style={"color": "lightgrey",
                      'fontSize': 6,
                      'fontWeight': "italic"
                      }
               ),
    ],
    style={"position": "fixed",
           "top": 0, "left": 0, "bottom": 0,
           "width": "10rem",
           "padding": "2rem 1rem",
           "background-color": 'whitesmoke',
           "font_family": 'Verdana',
           #"overflow": "scroll",
           }
)

# LOAD APP
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='H2ousing@work', )
server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])



@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div(children=
        [
            html.H2(children='ACT Team #3148',
                    style={'textAlign': 'center',
                           'display': 'inline-block',
                           'marginLeft': 55,
                           'marginTop': 0,
                           'fontSize': 30,
                           }),
        ])
    elif pathname == "/" + Pname_CODE + "/typologies":
        typologies()
        return html.Div(children=
        [
            html.H2(children='Typologies',
                    style={'textAlign': 'top',
                           'display': 'inline-block',
                           'marginLeft': 55,
                           'marginTop': 0,
                           'fontSize': 30,
                           }),
            html.Br(),
            dcc.Graph(figure=fig_typ1,
                      style={'textAlign': 'center',
                             'display': 'inline-block'}),
            dcc.Graph(figure=fig_typ2,
                      style={'textAlign': 'center',
                             'display': 'inline-block'}),
            dash_table.DataTable(
                id='datatable-input',
                columns=[
                    {"name": i, "id": i} for i in df_labels.loc[:,['index', 'origin', 'category', 'dimension', 'label', 'weights_1', 'weights_2']]
                ],
                editable=True,
                page_current=0,
                page_size=100,
                page_action='custom',
                sort_action='custom',
                sort_mode='single',
                sort_by=[],
                style_cell={'textAlign': 'left',
                            'fontSize': 12,
                            },
                style_cell_conditional=[{
                    'if': {'column_id': 'VALUE'},
                    'textAlign': 'right'
                }],
                export_format='none',
                export_headers='names',
                merge_duplicate_headers=False,
                css=[
                    {"selector": ".column-header--delete svg", "rule": 'display: "none"'},
                    {"selector": ".column-header--delete::before", "rule": 'content: "X"'}
                ]
            ),
            html.Button("Download CSV", id="btn-csv",
                        style={'display': 'inline-block',
                               'marginLeft': 0,
                               }),
            html.Button("Save changes", id="btn-save", n_clicks=0,
                        style={'display': 'inline-block',
                               'marginLeft': 10,
                               }),
            html.Div(children=[
                dcc.Download(id="download-dataframe-csv"),
                dcc.Input(value='', id='filter-input', placeholder='Filter', debounce=True),
                dcc.Store(id='data-store')
            ],
                style=dict(display='flex', justifyContent='center'),
            ),
            html.A(html.Button("Refresh Data", id="save-input", n_clicks=0,
                               style={'display': 'none'}), href='/'),
        ])
    elif pathname == "/" + Pname_CODE + "/project_info":
        model_results()
        return html.Div(children=
        [
            dcc.Graph(figure=MAP_0,
                      style={
                          'marginTop': 0,
                      }),
        ])
    elif pathname == "/" + Pname_CODE + "/model":
        model_results()
        return html.Div(children=
        [
            html.H2(children='Model information',
                    style={'textAlign': 'center',
                           'fontSize': 25,
                           'display': 'inline-block',
                           'marginLeft': 55,
                           'marginTop': 0,
                            }),
            # dcc.Graph(figure=fig_1,
            #           style={
            #               'marginTop': 0,
            #           }),
        ])
    elif pathname == "/" + Pname_CODE + "/results":
        model_results()
        return html.Div(children=
        [
            html.H2(children='Results',
                    style={'textAlign': 'center',
                           'fontSize': 25,
                           'display': 'inline-block',
                           'marginLeft': 55,
                           'marginTop': 0,
                           }),
            dcc.Graph(figure=MAP_1,
                      style={
                          'marginTop': 0,
                      }),
            dcc.Graph(figure=MAP_2,
                      style={
                          'marginTop': 0,
                      }),
        ])

    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# Change link color when clicked
@app.callback(Output('P_i', 'style'), [Input('P_i', 'n_clicks')])
def change_button_style(n_clicks):
    if n_clicks > 0:
        return {"color": "dodgerblue"}
    else:
        return {"color": "dodgerblue"}

## Datatable callbacks
# Datatable df_SCOPE filter
@app.callback(
    Output('datatable-input', 'data'),
    [Input('datatable-input', 'page_current'),
     Input('datatable-input', 'page_size'),
     Input('datatable-input', 'sort_by'),
     Input('filter-input', 'value')],
)
def search(page_current, page_size, sort_by, filter_string):
    # Filter
    dff_labels = df_labels[df_labels.apply(lambda row: row.str.contains(filter_string, regex=False).any(), axis=1)]
    # Sort if necessary
    if len(sort_by):
        dff_labels = dff_labels.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc',
            inplace=False
        )
    return dff_labels.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records')

# Datatable CSV export
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df_labels.to_csv, "labels.csv")

# Datatable CSV save
@app.callback(
    Output('datatable-input', 'columns'),
    Input('btn-save', 'n_clicks'),
    State('datatable-input', 'data'),
    prevent_initial_call=True,
)
def save(n_clicks, data):
    import_data()
    typologies()
    df_weights.to_csv("weights.csv")
    print('reload..')
    return [{"name": i, "id": i} for i in df_weights.columns]

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080, use_reloader=False)
    # Runs @ local host e.g. "http://127.0.0.1:8080/" OR server subhost "http://10.10.10.200:8080/"

#os.close()