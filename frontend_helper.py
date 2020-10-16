import json
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from datetime import datetime
from ast import literal_eval
import operator

colors = {
    'plot_background': '#ffffff',
    'plot_text': '#000000',
    'bar_fill_color':'#00CC96',
    'button_text': '#00CC96',
    'button_border': '#000000',
    'title_text_color': '#323232',
}

#generate main sneaker graph helper function
def main_sneaker_fig_generator(dataframe):
    main_sneaker_fig = go.Figure(
        data = [
            {'x' : dataframe['Order Date'], 'y' : dataframe['Sale Price'], 'name': 'Sale Price'},
            {'x' : dataframe['Order Date'], 'y' : dataframe['Retail Price'], 'name': 'Retail Price'}
        ],
        layout =  {
            'title': 'Sneaker Price Over Time (Size 9)',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Price ($)'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': colors['plot_background'],
            'font': {'color': colors['plot_text']},
        },
    )
    sneaker_count_df = dataframe.groupby('Order Date').count().reset_index()
    main_sneaker_fig.add_trace(
        go.Bar(x=sneaker_count_df['Order Date'].to_list(),
               y=sneaker_count_df['Sale Price'].to_list(),
               name='Transaction',
               yaxis="y2",
               marker_line_width=0.1,
               marker_color=colors['bar_fill_color']
              ),
    )
    main_sneaker_fig.update_layout(
        barmode='group',
        hovermode='x',
        legend=dict(
                orientation = 'h',
                yanchor='bottom',
                xanchor='center',
                y=1.03,
                x=0.5,
        ),
        xaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            title='Price ($)',
#             anchor="x",
#             overlaying="y",
            side="left",
            showline=True,
            linewidth=1.5,
            linecolor='black',
        ),
        yaxis2=dict(
            title='Transactions',
            anchor="x",
            overlaying="y",
            side="right",
            showline=True,
            linewidth=1.5,
            linecolor='black',
        )
    )

    release_dates = dataframe[dataframe['Release Date'] == 1]['Order Date'].unique().astype('datetime64[D]')
    for date in release_dates:
        main_sneaker_fig = main_sneaker_fig.add_shape(
            dict(
                type="line",
                x0=date,
                y0=0,
                x1=date,
                y1=dataframe['Sale Price'].max()*1.2,
                line=dict(
                    color="Gray",
                    width=1,
                    dash='dash'
                ),
                name='Release Date',
            )
        )
    return main_sneaker_fig

#generate line graph for number of sneaker transactions
def transactions_sneaker_fig_generator(dataframe):
    sneaker_count_df = dataframe.groupby('Order Date').count().reset_index()
    transactions_sneaker_fig = go.Figure(
        data = [go.Bar(
                {'x' : sneaker_count_df['Order Date'].to_list(),
                 'y' : sneaker_count_df['Sale Price'].to_list(),
                 'name' : 'Transaction'}
            ),
        ],
        layout =  {
            'title': 'Number of Sneaker Transactions (Size 9)',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Number of Transactions'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': colors['plot_background'],
            'font': {
                'color': colors['plot_text']
            }
        }
    )
    release_dates = dataframe[dataframe['Release Date'] == 1]['Order Date'].unique().astype('datetime64[D]')
    for date in release_dates:
        transactions_sneaker_fig = transactions_sneaker_fig.add_shape(
            dict(
                type="line",
                x0=date,
                y0=0,
                x1=date,
                y1=sneaker_count_df['Sale Price'].max()*1.2,
                line=dict(
                    color="Gray",
                    width=1,
                    dash='dash'
                ),
                name='Release Date',
            )
        )
    transactions_sneaker_fig.update_layout(
        barmode='group',
        hovermode='x',
        legend=dict(
                orientation = 'h',
                yanchor='bottom',
                xanchor='center',
                y=1.03,
                x=0.5,
        ),
        xaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
    )
    return transactions_sneaker_fig

weekday_dict = {0 : 'Monday', 1 : 'Tuesday', 2 : 'Wednesday', 3 : 'Thursday', 4 : 'Friday', 5 : 'Saturday', 6 : 'Sunday'}
#generate bar graph of sneaker transactions by week
def transactions_bar_sneaker_fig_generator(dataframe):
    dataframe['Day of Week'] = dataframe['Order Date'].dt.weekday
    sneaker_count_df = dataframe.groupby('Day of Week').count().reset_index()
    transactions_bar_sneaker_fig = go.Figure(
        data = [go.Bar(
                {'x' : [weekday_dict[x] for x in sneaker_count_df['Day of Week'].to_list()],
                 'y' : sneaker_count_df['Sale Price'].to_list(),
                 'name' : 'Transaction',
                 'marker_color':colors['bar_fill_color']},
            ),
        ],
        layout =  {
            'title': 'Number of Sneaker Transactions Each Week Day (Size 9)',
            'xaxis': {'title': 'Day of Week'},
            'yaxis': {'title': 'Number of Transactions'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': colors['plot_background'],
            'font': {
                'color': colors['plot_text']
            }
        }
    )
    transactions_bar_sneaker_fig.update_layout(
        legend=dict(
                orientation = 'h',
                yanchor='bottom',
                xanchor='center',
                y=1.03,
                x=0.5,
        ),
        xaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
    )
    return transactions_bar_sneaker_fig

#calculate percent difference helper function
def percent_difference(dataframe, col_1, col_2):
    difference = dataframe[col_1] - dataframe[col_2]
    percent_difference = difference / dataframe[col_2]
    mean_percent_difference = percent_difference.mean()
    return round(mean_percent_difference * 100, 2)

#calculate volatility using mean price per day helper function
def volatility(dataframe):
    new_df = dataframe.groupby('Order Date').mean()
#     new_df = dataframe.set_index('Order Date')
    log_return = np.log(1 + new_df['Sale Price'].pct_change())
    volatility = (log_return.rolling(30).std() * (365**0.5)).iloc[-1]
    return round(volatility * 100, 2)

#calculate max and min of dataframe
def max_min_range(dataframe):
    return dataframe['Sale Price'].max(), dataframe['Sale Price'].min()

#filter dataframe to a set range of dates
def filter_by_dates(dataframe, date_start, date_end):
    new_df = dataframe.set_index('Order Date')[date_start:date_end]
    return new_df.reset_index()

#get new date X days from date
def get_date(date_start, days):
    if isinstance(date_start, str):
        return datetime.strptime(date_start, '%Y-%m-%d').date() + timedelta(days)
    else:
        return date_start + timedelta(days)

# calculate max and min of dataframe for days before/after date
def max_min_any_range(dataframe, date, days):
    return max_min_range(filter_by_dates(dataframe, get_date(date, days), date))

#generate main insta graph helper function
def main_insta_fig_generator(dataframe, insta_data):
    value_dict = {'likes_num' : ['Instagram Likes', 'Instagram Likes Over Time', 'Likes'],
                  'posts_num' : ['Instagram Posts', 'Instagram Posts Over Time', 'Posts'],
                  'posts_compound' : ['Instagram Sentiment', 'Instagram Sentiment Over Time', 'Sentiment Score'],
                  'comments_compound_real' : ['Instagram Comment Sentiment', 'Instagram Comment Sentiment Over Time', 'Sentimen Score']}
    main_insta_fig = go.Figure(
        data =  [
            {'x' : dataframe['Order Date'], 'y' : dataframe[insta_data], 'name': value_dict[insta_data][0]},
        ],
        layout = {
            'title': value_dict[insta_data][1],
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': value_dict[insta_data][2]},
            'plot_bgcolor': 'white',
            'paper_bgcolor': colors['plot_background'],
            'font': {
                'color': colors['plot_text']
            }
        }
    )
    main_insta_fig.update_layout(
        legend=dict(
                orientation = 'h',
                yanchor='bottom',
                xanchor='center',
                y=1.03,
                x=0.5,
        ),
        xaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
    )
    return main_insta_fig

#get top bigrams from insta_posts_df dataframe
def get_top_bigrams(dataframe, top_num):
    bigrams_count = {}
    for i in dataframe.index:
        token_post = dataframe['bigrams_final'][i]
        for bigram in token_post:
            phrase = bigram[0] + ' ' + bigram[1]
            if (phrase[0:1] != '#' and phrase[0:1] != '@' and len(phrase) > 1):
                key = phrase
                if key in bigrams_count:
                    bigrams_count[key] += 1
                else:
                    bigrams_count[key] = 1
    top_bigrams = dict(sorted(bigrams_count.items(), key=operator.itemgetter(1), reverse=True)[:top_num])
    return top_bigrams

#generate bar graph of sneaker transactions by week
def bigrams_bar_insta_fig_generator(dataframe, top_num):
    top_bigrams = get_top_bigrams(dataframe, top_num)
    bigrams_bar_insta_fig = go.Figure(
        data = [go.Bar(
                {'x' : list(top_bigrams.keys()),
                 'y' : list(top_bigrams.values()),
                 'name' : 'Bigrams'}
            ),
        ],
        layout =  {
            'title': 'Top Counted Bigrams',
            'xaxis': {'title': 'Bigram'},
            'yaxis': {'title': 'Number of Bigrams'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': colors['plot_background'],
            'font': {
                'color': colors['plot_text']
            }
        }
    )
    bigrams_bar_insta_fig.update_layout(
        legend=dict(
                orientation = 'h',
                yanchor='bottom',
                xanchor='center',
                y=1.03,
                x=0.5,
        ),
        xaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
    )
    return bigrams_bar_insta_fig

#generate main model graph helper function
def main_model_fig_generator(dataframe):
    main_model_fig = go.Figure(
        data =  [
            {'x' : dataframe['Order Date'], 'y' : dataframe['Sale Price'], 'name': 'Sneaker Price'},
            {'x' : dataframe['Order Date'], 'y' : dataframe['Retail Price'], 'name': 'Retail Price'},
            {'x' : dataframe['Order Date'], 'y' : dataframe['Predicted Price'], 'name': 'Forecasted Price'},
        ],
        layout = {
            'title': 'Sneaker Forecasted Price Over Time (Size 9)',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Price($)'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': colors['plot_background'],
            'font': {
                'color': colors['plot_text']
            },
            'legend':dict(
                orientation = 'h',
                yanchor='bottom',
                xanchor='center',
                y=1.0,
                x=0.5,
            ),
        }
    )
    main_model_fig.update_layout(
        barmode='group',
        hovermode='x',
        legend=dict(
                orientation = 'h',
                yanchor='bottom',
                xanchor='center',
                y=1.03,
                x=0.5,
        ),
        xaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True
        ),
    )


    main_model_fig.add_trace(go.Scatter(
        x=dataframe['Order Date'],
        y=dataframe['Lower Predicted CI'],
        name='Confidence Bands',
        fill=None,
        mode='lines',
        line_color='gray',
        hoverinfo='none',
        legendgroup='Confidence Bands',
    ))
    main_model_fig.add_trace(go.Scatter(
        x=dataframe['Order Date'],
        y=dataframe['Upper Predicted CI'],
        name='Confidence Bands',
        fill='tonexty',
        mode='lines',
        line_color='gray',
        hoverinfo='none',
        legendgroup='Confidence Bands',
        showlegend=False,
    ))

    return main_model_fig

#add insta plots to figure
def add_insta_plots(dataframe, plot_column, figure):
    value_dict = {'likes_num' : ['Instagram Likes', 'Instagram Likes Over Time', 'Likes'],
                  'posts_num' : ['Instagram Posts', 'Instagram Posts Over Time', 'Posts'],
                  'posts_compound' : ['Instagram Sentiment', 'Instagram Sentiment Over Time', 'Sentiment Score'],
                  'comments_compound_real' : ['Instagram Comment Sentiment', 'Instagram Comment Sentiment Over Time', 'Sentimen Score'],
                  'num_ads' : ['Instagram Ads', 'Instagram Ads Over Time', 'Ads'],
                  'ads_likes_num' : ['Instagram Ad Likes', 'Instagram Ad Likes Over Time', 'Ad Likes']}
    figure.add_trace(
        go.Scatter(x=dataframe['Order Date'],
                   y=dataframe[plot_column],
                   name=value_dict[plot_column][0],
                   yaxis="y2"
                  ),
    )
    figure.update_layout(
#         legend=dict(
#             orientation = 'h',
#             yanchor='bottom',
#             y=1.0,
#             x=0.2,
#         ),
        yaxis=dict(
            title='Price ($)',
            anchor="x",
#             overlaying="y",
            side="left",
        ),
        yaxis2=dict(
            title=value_dict[plot_column][2],
            anchor="x",
            overlaying="y",
            side="right",
        )
    )

#Calculate RMSE helper function
def calculate_rmse(dataframe):
    mse = ((dataframe['Predicted Price'] - dataframe['Sale Price']) ** 2).mean()
    return round(np.sqrt(mse), 2)
