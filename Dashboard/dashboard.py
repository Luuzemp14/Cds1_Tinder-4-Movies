import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_renderer
from dash.dependencies import Input, Output, State
from flask import Flask, redirect

import pandas as pd

df = pd.read_csv('movies_hp.csv')

# Create the app object with bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Navigation
nav_menu = dbc.Navbar(
    children = [
        dbc.Nav([
                dbc.NavItem(dbc.NavLink('Select Genres', href='/select-genres')),
                dbc.NavItem(dbc.NavLink('Tinder', href='/tinder')),
                dbc.NavItem(dbc.NavLink('Recommendations', href='/recommendations')),
            ],
            vertical = False,
        ),
    ],
    color='primary',
    dark=True,
    className='mb-5 justify-content-center',
)

# Define the app layout
app.layout = html.Div([
    dcc.Location(id = 'url', refresh = False),
    html.Div(id = 'page-content', style = {'textAlign': 'center', 'font-family': 'Arial'}),
])

# Create Layout for page 1
page_1_layout = html.Div(
    children = [
        nav_menu,
        html.H1('Select your favorite Genres', style = {'margin-top': '20px'}),
        html.H3('Multiple selections are possible', style = {'margin-top': '20px'}),
        dcc.Checklist(
            style = {'margin-top': '20px'},
            id = 'checklist',        
            options = [
                {'label': ' Action', 'value': 'Action'},
                {'label': ' Adventure', 'value': 'Adventure'},
                {'label': ' Animation', 'value': 'Animation'},
                {'label': ' Comedy', 'value': 'Comedy'},
                {'label': ' Crime', 'value': 'Crime'},
                {'label': ' Documentary', 'value': 'Documentary'},
                {'label': ' Drama', 'value': 'Drama'},
                {'label': ' Family', 'value': 'Family'},
                {'label': ' Fantasy', 'value': 'Fantasy'},
                {'label': ' History', 'value': 'History'},
                {'label': ' Horror', 'value': 'Horror'},
                {'label': ' Music', 'value': 'Music'},
                {'label': ' Mystery', 'value': 'Mystery'},
                {'label': ' Romance', 'value': 'Romance'},
                {'label': ' Science Fiction', 'value': 'Science Fiction'},
                {'label': ' TV Movie', 'value': 'TV Movie'},
                {'label': ' Thriller', 'value': 'Thriller'},
                {'label': ' War', 'value': 'War'},
                {'label': ' Western', 'value': 'Western'}
            ],
            labelStyle = {'display': 'block'},
            value = []
        ),

        # text of selected genres
        html.Div(
            id = 'selected_genres', 
            style = {'margin-top': '20px'}
        ),

        # start button
        dcc.Link(html.Button(
            'Start swiping',
            id = 'start-button',
            n_clicks = 0, 
            style = {'margin-top': '20px'}),
            href = "/tinder"),
    ]
)

# Update chosen genres
@app.callback(
    Output('selected_genres', 'children'),
    [Input('checklist', 'value')])
def update_output(value):
    print(value)
    text = 'You have selected {}'.format(value)
    return text

# Page for Tinder - Like or dislike movies
page_2_layout = html.Div(
    children = [
        nav_menu,

        # show movie poster without formatting image
        html.Img(id='movie-poster', style = {'margin-top': '20px'}, height = '300px'),

        #movie details
        html.Div([
            html.H2(id='movie-title', style = {'margin-top': '20px'}),
            html.P(id='movie-overview', style = {'margin-top': '20px', 'margin-left': '15%', 'margin-right': '15%'}),
            dbc.Button('Like', id='like-button', style = {'margin-right': '10px'}, color = 'success'),
            dbc.Button('Have-not seen', id='not_seen-button', color = 'secondary'),
            dbc.Button('Dislike', id = 'dislike-button', style = {'margin-left': '10px'}, color = 'danger')            
        ]),

        # button to start recommendations and redirect to page3
        # hide button until user has liked at least 5 movies
        dcc.Link(dbc.Button(
            'Show recommendations',
            id = 'recommendations-button',
            n_clicks = 0, 
            style = {'margin-top': '40px', 'display': 'none'}),
            href = "/recommendations"),
])

@app.callback(
    [Output(component_id='movie-poster', component_property='src'),
     Output(component_id='movie-title', component_property='children'),
     Output(component_id='movie-overview', component_property='children')],
    [Input('like-button', 'n_clicks'),
     Input('dislike-button', 'n_clicks'),
     Input('not_seen-button', 'n_clicks')]
)
def update_image(like_clicks, dislike_clicks, not_seen_clicks):
    if 'df_likes' not in globals():
        globals()['df_likes'] = pd.DataFrame(columns=['title', 'overview'])

    if 'df_dislikes' not in globals():
        globals()['df_dislikes'] = pd.DataFrame(columns=['title', 'overview'])

    random_movie = df.sample(1) 
    poster_url = random_movie['backdrop_path'].values[0]
    poster_url = 'https://www.themoviedb.org/t/p/original'+ poster_url
    title = random_movie['title'].values[0]
    overview = random_movie['overview'].values[0]

    # Add the movie to the dataframe
    if like_clicks is not None:
        df_likes = pd.DataFrame({'title': [title], 'overview': [overview]})
        globals()['df_likes'] = globals()['df_likes'].append(df_likes, ignore_index=True)

    elif dislike_clicks is not None:
        df_dislikes = pd.DataFrame({'title': [title], 'overview': [overview]})
        globals()['df_dislikes'] = globals()['df_dislikes'].append(df_dislikes, ignore_index=True)

    return poster_url, title, overview


# Page for recommendations - show the 5 recommended movies for the user
page_3_layout = html.Div(
    children = [
        nav_menu,
])


# Create the callback for the page content based on the url
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/select-genres':
        return page_1_layout
    elif pathname == '/tinder':
        return page_2_layout
    elif pathname == '/recommendations':
        return page_3_layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server()
