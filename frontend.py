import json
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from datetime import datetime
from ast import literal_eval
import operator
from frontend_helper import *

import dash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

colors = {
    'plot_background': '#ffffff',
    'plot_text': '#000000',
    'bar_fill_color':'#00CC96',
    'button_text': '#00CC96',
    'button_border': '#000000',
    'title_text_color': '#323232',
}

loc = 'data/'

#load model and insta data
insta_df = pd.read_csv(loc + 'model_output_insta.csv')

#load model with no insta data
ts_no_insta_df = pd.read_csv(loc + 'model_output_noinsta.csv')
ts_no_insta_df = ts_no_insta_df.set_index('Order Date')[insta_df['Order Date'].min():].reset_index()
ts_no_insta_df['Retail Price'] = 220

#load insta data for NLP and text
insta_posts_df = pd.read_csv(loc + 'instagram_post.csv')
insta_posts_df = insta_posts_df[['time','bigrams_final', 'caption adjectives']]
insta_posts_df = insta_posts_df.rename(columns = {'time' : 'Order Date'})
insta_posts_df['Order Date'] = pd.to_datetime(insta_posts_df['Order Date'])
insta_posts_df['Order Date'] = insta_posts_df['Order Date'].values.astype('datetime64[D]')
insta_posts_df['bigrams_final'] = insta_posts_df['bigrams_final'].apply(literal_eval)
insta_posts_df['caption adjectives'] = insta_posts_df['caption adjectives'].apply(literal_eval)

#load sneaker stockx data
stockx_df = pd.read_excel(loc + 'StockX-Data-Contest-2019.xlsx', sheet_name = 'Raw Data')
stockx_sneaker_df = stockx_df[stockx_df['Sneaker Name'] == 'Adidas-Yeezy-Boost-350-V2-Zebra'].reset_index()
stockx_sneaker_df = stockx_sneaker_df[['Order Date', 'Shoe Size', 'Sale Price', 'Retail Price']]
stockx_sneaker_df = stockx_sneaker_df.sort_values('Order Date')

#load sneaker data
with open(loc + 'yeezyzebraprice.json', encoding="utf8") as json_file:
    shoe_dict = json.load(json_file)
new_shoe_data = pd.io.json.json_normalize(shoe_dict, record_path="ProductActivity")[["chainId",
                                                                                       "amount",
                                                                                       "createdAt",
                                                                                       "shoeSize",
                                                                                       "localAmount"]]
new_shoe_data = new_shoe_data.rename(columns = {'createdAt' : 'Order Date',
                                                'localAmount' : 'Sale Price',
                                                'shoeSize' : 'Shoe Size'})
cols = ['chainId', 'amount']
new_shoe_data.drop(cols, axis=1, inplace=True)
new_shoe_data['Order Date'] = pd.to_datetime(new_shoe_data['Order Date'])
new_shoe_data['Order Date'] = new_shoe_data['Order Date'].values.astype('datetime64[D]')
new_shoe_data['Shoe Size'] = new_shoe_data['Shoe Size'].values.astype('float')
new_shoe_data['Retail Price'] = 220
new_shoe_data = new_shoe_data.sort_values('Order Date')
sneaker_df = new_shoe_data

zebra_releases = ['2017-02-15', '2017-06-24', '2018-11-09', '2019-08-02', '2019-12-21', '2020-06-26']
sneaker_df['Release Date'] = np.where(sneaker_df['Order Date'].isin(zebra_releases), 1, 0)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div(children=[
    html.Div(
        [
            html.Div(
                className='two columns'
            ),
            html.Div(
                [
                    html.H2(
                        children='Sneaker Reselling Analysis',
                        style={
                            'textAlign': 'center',
                            'color': colors['title_text_color']
                        }
                    ),
                    html.Div(children='Analyzing trends between sneaker resell prices and Instagram post sentiment',
                             style={
                                 'textAlign': 'center',
                                 'color': colors['title_text_color']
                             }
                            ),
                ],
                style={'textAlign': 'center'},
                className='eight columns',
            ),
            dcc.Location(id='url',
                         refresh=True),
            html.Div(
                [
                    dcc.Link(
                        html.Button('Data Exploration',
                                    id = 'data_button',
                                    style = {'border-color': colors['button_border'],
                                             'vertical-align': 'middle',
                                             'textAlign': 'center',
                                             'marginBottom': '10px'}),
                        href='/data_exploration'
                    ),
                    html.Br(),
                    dcc.Link(
                        html.Button('Model Results',
                                    id = 'model_button',
                                    style = {'border-color':colors['button_border'],
                                             'vertical-align': 'middle',
                                             'textAlign': 'center'}),
                        href='/model_results'
                    ),
                ],
                style={'textAlign': 'center'},
                className='one columns'
            ),
        ],
        id="header",
        className='row',
        style={'height': '50px', 'vertical-align': 'top'}
    ),
    html.Br(),
    html.Br(),

    html.Div(id='page-content'),
])

#update page content
@app.callback([Output('page-content', 'children'),
               Output('data_button', 'style'),
               Output('model_button', 'style')],
              [Input('url', 'pathname')])
def update_page(pathname):
    if pathname == '/model_results':
        data_button_style = {'border-color':'black',
                             'vertical-align': 'middle',
                             'textAlign': 'center',
                             'marginBottom': '10px'}
        model_button_style = {'border-color':'black',
                              'vertical-align': 'middle',
                              'textAlign': 'center',
                              'color': colors['button_text']}
        return html.Div([
            dcc.Tabs(id="model_tabs", value='tab_no_insta', children=[
                dcc.Tab(label='Simple Time Series', value='tab_no_insta'),
                dcc.Tab(label='Time Series with Instagram Sentiment Analysis', value='tab_with_insta'),
            ]),
            html.Div(id='model-tabs-content'),
        ]), data_button_style, model_button_style
    else:
        data_button_style = {'border-color':'black',
                             'vertical-align': 'middle',
                             'textAlign': 'center',
                             'marginBottom': '10px',
                             'color': colors['button_text']}
        model_button_style = {'border-color':'black',
                              'vertical-align': 'middle',
                              'textAlign': 'center',}
        return html.Div([
            dcc.Tabs(id="data_tabs", value='tab_sneaker', children=[
                dcc.Tab(label='Sneaker Price Exploration', value='tab_sneaker'),
                dcc.Tab(label='Instagram Post Exploration', value='tab_insta'),
            ]),
            html.Div(id='data-tabs-content'),
        ]), data_button_style, model_button_style

#update data exploration page
@app.callback(Output('data-tabs-content', 'children'),
              [Input('data_tabs', 'value')])
def update_data_tab(value):
    if value == 'tab_sneaker':
        return html.Div([
            html.Div(
                dcc.DatePickerRange(
                    id='sneaker_df_range',
                    min_date_allowed=sneaker_df['Order Date'].min(),
                    max_date_allowed=sneaker_df['Order Date'].max(),
                    start_date=sneaker_df['Order Date'].min(),
                    end_date=sneaker_df['Order Date'].max(),
                    minimum_nights=33,
                    with_portal=True,
                    clearable=True
                ),
                style={"background-color": "white", 'textAlign': 'center'},
                className="pretty_container",
            ),

            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            len(sneaker_df.index),
                                            style={'color':'black'},
                                            id="transaction_text",
                                        ),
                                        html.P("Total Transactions"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            "$ " + str(round(sneaker_df['Sale Price'].mean(), 2)),
                                            style={'color':'black'},
                                            id="average_sale_price",
                                        ),
                                        html.P("Average Resell Price"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            percent_difference(sneaker_df, 'Sale Price', 'Retail Price'),
                                            style={'color':'black'},
                                            id="percent_difference_price",
                                        ),
                                        html.P("% Gain"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            volatility(sneaker_df),
                                            style={'color':'black'},
                                            id="price_volatility",
                                        ),
                                        html.P("% Price Volatility"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            '$ ' + str(max_min_any_range(sneaker_df, sneaker_df['Order Date'].max(), -30)[0])
                                            + '/'
                                            + str(max_min_any_range(sneaker_df, sneaker_df['Order Date'].max(), -30)[1]),
                                            style={'color':'black'},
                                            id="thirty_day_range",
                                        ),
                                        html.P("30 Day High/Low"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                        ],
                        justify="center",
                    ),
                ],
            ),
            html.Div(
                dcc.Graph(
                    id='main_sneaker_graph',
                    figure=main_sneaker_fig_generator(sneaker_df),
                ),
                style={"background-color": "white"},
                className="pretty_container",
            ),
            html.Div(
                dcc.Graph(
                    id='transactions_bar_sneaker_graph',
                    figure=transactions_bar_sneaker_fig_generator(sneaker_df),
                ),
                style={"background-color": "white"},
                className="pretty_container",
            ),
        ])
    elif value == 'tab_insta':
        return html.Div([
            html.Div(
                dcc.DatePickerRange(
                    id='insta_df_range',
                    min_date_allowed=insta_df['Order Date'].min(),
                    max_date_allowed=insta_df['Order Date'].max(),
                    start_date=insta_df['Order Date'].min(),
                    end_date=insta_df['Order Date'].max(),
                    minimum_nights=33,
                    with_portal=True,
                    clearable=True
                ),
                style={"background-color": "white", 'textAlign': 'center'},
                className="pretty_container",
            ),

            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            round(insta_df['posts_num'].mean(), 2),
                                            style={'color':'black'},
                                            id="insta_posts_mean",
                                        ),
                                        html.P("Number Of Posts / Day"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            round(insta_df['likes_num'].mean(), 2),
                                            style={'color':'black'},
                                            id="insta_likes_mean",
                                        ),
                                        html.P("Number Of Likes / Day"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            round(insta_df['comments_num'].sum() / insta_df['posts_num'].sum(), 2),
                                            style={'color':'black'},
                                            id="insta_comments_per_post_mean",
                                        ),
                                        html.P("Number Of Comments / Post"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            round(insta_df['posts_compound'].sum() / insta_df['posts_num'].sum(), 2),
                                            style={'color':'black'},
                                            id="insta_sentiment_mean",
                                        ),
                                        html.P("Average Sentiment / Post"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                        ],
                        justify="center",
                    ),
                ],
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id='insta_dropdown',
                        options=[
                            {'label': 'Number of Likes', 'value': 'likes_num'},
                            {'label': 'Number of Posts', 'value': 'posts_num'},
                            {'label': 'Post Sentiment', 'value': 'posts_compound'},
                            {'label': 'Comment Sentiment', 'value': 'comments_compound_real'}
                        ],
                        value='likes_num'
                    ),
                    dcc.Graph(
                        id='main_insta_graph',
                        figure=main_insta_fig_generator(insta_df, 'likes_num')
                    ),
                ],
                style={"background-color": "white"},
                className="pretty_container",
            ),
            html.Div(
                dcc.Graph(
                    id='bigrams_bar_insta_graph',
                    figure=bigrams_bar_insta_fig_generator(insta_posts_df, 20),
                ),
                style={"background-color": "white"},
                className="pretty_container",
            ),
        ])


#update model results page
@app.callback(Output('model-tabs-content', 'children'),
              [Input('model_tabs', 'value')])
def update_model_tab(value):
    if value == 'tab_no_insta':
        return html.Div([
            html.Div(
                dcc.DatePickerRange(
                    id='model_no_insta_range',
                    min_date_allowed=ts_no_insta_df['Order Date'].min(),
                    max_date_allowed=ts_no_insta_df['Order Date'].max(),
                    start_date=ts_no_insta_df['Order Date'].min(),
                    end_date=ts_no_insta_df['Order Date'].max(),
                    minimum_nights=33,
                    with_portal=True,
                    clearable=True
                ),
                style={"background-color": "white", 'textAlign': 'center'},
                className="pretty_container",
            ),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            calculate_rmse(ts_no_insta_df),
                                            style={'color':'black'},
                                            id="predicted_no_insta_rmse",
                                        ),
                                        html.P("Predicted vs Actual RMSE"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            "$ " + str(round(ts_no_insta_df['Predicted Price'].mean(), 2)),
                                            style={'color':'black'},
                                            id="average_predicted_no_insta_price",
                                        ),
                                        html.P("Average Predicted Price"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            percent_difference(ts_no_insta_df, 'Predicted Price', 'Retail Price'),
                                            style={'color':'black'},
                                            id="percent_difference_no_insta_predicted",
                                        ),
                                        html.P("Predicted % Gain"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                        ],
                        justify="center",
                    ),
                ],
            ),
            html.Div(
                dcc.Graph(
                    id='main_model_no_insta_graph',
                    figure=main_model_fig_generator(ts_no_insta_df),
                ),
                style={"background-color": "white"},
                className="pretty_container",
            ),
        ])
    elif value == 'tab_with_insta':
        return html.Div([
            html.Div(
                dcc.DatePickerRange(
                    id='model_insta_range',
                    min_date_allowed=insta_df['Order Date'].min(),
                    max_date_allowed=insta_df['Order Date'].max(),
                    start_date=insta_df['Order Date'].min(),
                    end_date=insta_df['Order Date'].max(),
                    minimum_nights=33,
                    with_portal=True,
                    clearable=True
                ),
                style={"background-color": "white", 'textAlign': 'center'},
                className="pretty_container",
            ),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            calculate_rmse(insta_df),
                                            style={'color':'black'},
                                            id="predicted_no_insta_rmse",
                                        ),
                                        html.P("Predicted vs Actual RMSE"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            "$ " + str(round(insta_df['Predicted Price'].mean(), 2)),
                                            style={'color':'black'},
                                            id="average_predicted_no_insta_price",
                                        ),
                                        html.P("Average Predicted Price"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6(
                                            percent_difference(insta_df, 'Predicted Price', 'Retail Price'),
                                            style={'color':'black'},
                                            id="percent_difference_no_insta_predicted",
                                        ),
                                        html.P("Predicted % Gain"),
                                    ],
                                ),
                                style={"background-color": "white", 'textAlign': 'center'},
                                className="pretty_container",
                                width=2,
                            ),
                        ],
                        justify="center",
                    ),
                ],
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id='model_checkbox',
                        options=[
                            {'label': 'Number of Posts', 'value': 'posts_num'},
                            {'label': 'Number of Likes', 'value': 'likes_num'},
                            {'label': 'Post Sentiment', 'value': 'posts_compound'},
                            {'label': 'Comment Sentiment', 'value': 'comments_compound_real'},
                            {'label': 'Number of Ads', 'value': 'num_ads'},
                            {'label': 'Number of Ad Likes', 'value': 'ads_likes_num'},
                        ],
                        value='likes_num',
                    ),
                    dcc.Graph(
                        id='main_model_insta_graph',
                        figure=main_model_fig_generator(insta_df),
                    ),
                ],
                style={"background-color": "white"},
                className="pretty_container",
            ),
        ])

#callbacks for sneaker tab
@app.callback([Output('sneaker_df_range', 'start_date'),
               Output('sneaker_df_range', 'end_date')],
              [Input('main_sneaker_graph', 'relayoutData')])
def update_sneaker_daterangepicker(relayoutData):
    if relayoutData is None:
        return sneaker_df['Order Date'].min(), sneaker_df['Order Date'].max()
    elif 'xaxis.range[0]' in relayoutData:
        return relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
    else:
        return sneaker_df['Order Date'].min(), sneaker_df['Order Date'].max()

@app.callback(Output('main_sneaker_graph', 'figure'),
              [Input('sneaker_df_range', 'start_date'),
               Input('sneaker_df_range', 'end_date')])
def update_main_sneaker_graph(start_date, end_date):
    new_sneaker_df = filter_by_dates(sneaker_df, start_date, end_date)
    return main_sneaker_fig_generator(new_sneaker_df)

@app.callback(Output('transactions_sneaker_graph', 'figure'),
              [Input('sneaker_df_range', 'start_date'),
               Input('sneaker_df_range', 'end_date')])
def update_transactions_sneaker_graph(start_date, end_date):
    new_sneaker_df = filter_by_dates(sneaker_df, start_date, end_date)
    return transactions_sneaker_fig_generator(new_sneaker_df)

@app.callback(Output('transactions_bar_sneaker_graph', 'figure'),
              [Input('sneaker_df_range', 'start_date'),
               Input('sneaker_df_range', 'end_date')])
def update_transactions_bar_sneaker_graph(start_date, end_date):
    new_sneaker_df = filter_by_dates(sneaker_df, start_date, end_date)
    return transactions_bar_sneaker_fig_generator(new_sneaker_df)

@app.callback(Output('transaction_text', 'children'),
              [Input('sneaker_df_range', 'start_date'),
               Input('sneaker_df_range', 'end_date')])
def update_transaction_text(start_date, end_date):
    new_sneaker_df = filter_by_dates(sneaker_df, start_date, end_date)
    return len(new_sneaker_df)

@app.callback(Output('average_sale_price', 'children'),
              [Input('sneaker_df_range', 'start_date'),
               Input('sneaker_df_range', 'end_date')])
def update_average_sale_price(start_date, end_date):
    new_sneaker_df = filter_by_dates(sneaker_df, start_date, end_date)
    return "$ " + str(round(new_sneaker_df['Sale Price'].mean(), 2))

@app.callback(Output('percent_difference_price', 'children'),
              [Input('sneaker_df_range', 'start_date'),
               Input('sneaker_df_range', 'end_date')])
def update_percent_difference_price(start_date, end_date):
    new_sneaker_df = filter_by_dates(sneaker_df, start_date, end_date)
    return percent_difference(new_sneaker_df, 'Sale Price', 'Retail Price')

@app.callback(Output('price_volatility', 'children'),
              [Input('sneaker_df_range', 'start_date'),
               Input('sneaker_df_range', 'end_date')])
def update_price_volatility(start_date, end_date):
    new_sneaker_df = filter_by_dates(sneaker_df, start_date, end_date)
    return volatility(new_sneaker_df)

@app.callback(Output('thirty_day_range', 'children'),
              [Input('sneaker_df_range', 'start_date'),
               Input('sneaker_df_range', 'end_date')])
def thirty_day_range(start_date, end_date):
    new_sneaker_df = filter_by_dates(sneaker_df, start_date, end_date)
    return  ('$ ' + str(max_min_any_range(new_sneaker_df, new_sneaker_df['Order Date'].max(), -30)[0]) + '/'
             + str(max_min_any_range(new_sneaker_df, new_sneaker_df['Order Date'].max(), -30)[1]))

#callbacks for insta tab
@app.callback([Output('insta_df_range', 'start_date'),
               Output('insta_df_range', 'end_date')],
              [Input('main_insta_graph', 'relayoutData')])
def update_insta_daterangepicker(relayoutData):
    if relayoutData is None:
        return insta_df['Order Date'].min(), insta_df['Order Date'].max()
    elif 'xaxis.range[0]' in relayoutData:
        return relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
    else:
        return insta_df['Order Date'].min(), insta_df['Order Date'].max()

@app.callback(Output('main_insta_graph', 'figure'),
              [Input('insta_df_range', 'start_date'),
               Input('insta_df_range', 'end_date'),
               Input('insta_dropdown', 'value')])
def update_main_insta_graph(start_date, end_date, value):
    new_insta_df = filter_by_dates(insta_df, start_date, end_date)
    return main_insta_fig_generator(new_insta_df, value)

@app.callback(Output('bigrams_bar_insta_graph', 'figure'),
              [Input('insta_df_range', 'start_date'),
               Input('insta_df_range', 'end_date')])
def update_bigrams_insta_graph(start_date, end_date):
    new_insta_posts_df = filter_by_dates(insta_posts_df, start_date, end_date)
    return bigrams_bar_insta_fig_generator(new_insta_posts_df, 20)

@app.callback(Output('insta_posts_mean', 'children'),
              [Input('insta_df_range', 'start_date'),
               Input('insta_df_range', 'end_date')])
def update_insta_posts_mean(start_date, end_date):
    new_insta_df = filter_by_dates(insta_df, start_date, end_date)
    return round(new_insta_df['posts_num'].mean(), 2)

@app.callback(Output('insta_likes_mean', 'children'),
              [Input('insta_df_range', 'start_date'),
               Input('insta_df_range', 'end_date')])
def update_insta_likes_mean(start_date, end_date):
    new_insta_df = filter_by_dates(insta_df, start_date, end_date)
    return round(new_insta_df['likes_num'].mean(), 2)

@app.callback(Output('insta_comments_per_post_mean', 'children'),
              [Input('insta_df_range', 'start_date'),
               Input('insta_df_range', 'end_date')])
def update_insta_comments_per_post_mean(start_date, end_date):
    new_insta_df = filter_by_dates(insta_df, start_date, end_date)
    return round(new_insta_df['comments_num'].sum() / new_insta_df['posts_num'].sum(), 2)

@app.callback(Output('insta_sentiment_mean', 'children'),
              [Input('insta_df_range', 'start_date'),
               Input('insta_df_range', 'end_date')])
def update_insta_sentiment_mean(start_date, end_date):
    new_insta_df = filter_by_dates(insta_df, start_date, end_date)
    return round(new_insta_df['posts_compound'].sum() / new_insta_df['posts_num'].sum(), 2)

#callbacks for model without insta tab
@app.callback([Output('model_no_insta_range', 'start_date'),
               Output('model_no_insta_range', 'end_date')],
              [Input('main_model_no_insta_graph', 'relayoutData')])
def update_main_model_no_insta_daterangepicker(relayoutData):
    if relayoutData is None:
        return ts_no_insta_df['Order Date'].min(), ts_no_insta_df['Order Date'].max()
    elif 'xaxis.range[0]' in relayoutData:
        return relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
    else:
        return ts_no_insta_df['Order Date'].min(), ts_no_insta_df['Order Date'].max()

@app.callback(Output('main_model_no_insta_graph', 'figure'),
              [Input('model_no_insta_range', 'start_date'),
               Input('model_no_insta_range', 'end_date')])
def update_main_model_no_insta_graph(start_date, end_date):
    new_model_df = filter_by_dates(ts_no_insta_df, start_date, end_date)
    return main_model_fig_generator(new_model_df)


#callbacks for model with insta tab
@app.callback([Output('model_insta_range', 'start_date'),
               Output('model_insta_range', 'end_date')],
              [Input('main_model_insta_graph', 'relayoutData')])
def update_main_model_insta_daterangepicker(relayoutData):
    if relayoutData is None:
        return insta_df['Order Date'].min(), insta_df['Order Date'].max()
    elif 'xaxis.range[0]' in relayoutData:
        return relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
    else:
        return insta_df['Order Date'].min(), insta_df['Order Date'].max()

@app.callback(Output('main_model_insta_graph', 'figure'),
              [Input('model_checkbox', 'value'),
               Input('model_insta_range', 'start_date'),
               Input('model_insta_range', 'end_date')])
def update_main_model_insta_graph(value, start_date, end_date):
    new_model_df = filter_by_dates(insta_df, start_date, end_date)
    new_graph = main_model_fig_generator(new_model_df)
    add_insta_plots(new_model_df, value, new_graph)
    return new_graph

if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_ui=False, dev_tools_props_check=False, port = 8050)
