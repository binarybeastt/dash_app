import dash
from dash import dcc, html, no_update
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import plotly.figure_factory as ff
import numpy as np

df = pd.read_csv('chart-device.csv')
df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = df['Time'].dt.hour
df['Weekday'] = df['Time'].dt.day_name()
now = df['Time'].max()
past_24_hours = now - timedelta(hours=24)
past_6_hours = now - timedelta(hours=6)

current_index = 0
def get_next_data_chunk(chunk_size=10):
    global current_index
    if current_index + chunk_size >= len(df):
        chunk = df[current_index:] 
        current_index = 0  
    else:
        chunk = df[current_index:current_index + chunk_size]
        current_index += chunk_size
    return chunk

app = dash.Dash(__name__, suppress_callback_exceptions=True)
dark_mode_styles = {
    "background-color": "#1e1e1e",
    "color": "#ffffff",
    "font-family": "Arial, sans-serif"  
}
tabs_styles = {
    'height': '30px',
    "font-size": "10px", 
    "text-align": "center"
    }

def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        style={
            "flex": "0 0 15%", 
            "height": "100vh",
            "border-right": "2px solid #333",  
            "padding": "10px",
            "overflow": "auto", 
            **dark_mode_styles
        },
        children=[
            html.Div(
                id="temperature-indicator",
                style={"margin-bottom": "20px", "border": "1px solid #333", "padding": "10px"},
                children=[
                    html.P("Current Temperature", style={"font-size": "15px"}),
                    daq.LEDDisplay(
                        id="temperature-led",
                        color="#ff5733",
                        backgroundColor="#1e2130",
                        size=15,  
                    ),
                    html.P("Current H2s", style={"font-size": "15px"}),
                    daq.LEDDisplay(
                        id="h2s-led",
                        color="#ff5733",
                        backgroundColor="#1e2130",
                        size=15,  
                    ),
                    html.P("Current NO2", style={"font-size": "15px"}),
                    daq.LEDDisplay(
                        id="No2-led",
                        color="#ff5733",
                        backgroundColor="#1e2130",
                        size=15,  
                    ),
                    html.P("Current Voc", style={"font-size": "15px"}),
                    daq.LEDDisplay(
                        id="Voc-led",
                        color="#ff5733",
                        backgroundColor="#1e2130",
                        size=15,  
                    ),
                    html.P("Current Pm10", style={"font-size": "15px"}),
                    daq.LEDDisplay(
                        id="Pm10-led",
                        color="#ff5733",
                        backgroundColor="#1e2130",
                        size=15,  
                    ),
                    html.P("Current Pm2", style={"font-size": "15px"}),
                    daq.LEDDisplay(
                        id="Pm2-led",
                        color="#ff5733",
                        backgroundColor="#1e2130",
                        size=15,  
                    ),
                ],
            ),
        ],
    )

app.layout = html.Div(
    [
        html.Div(
            [
                build_quick_stats_panel(), 
                html.Div(
                    [
                        dcc.Tabs(
                            id="tabs-example",
                            value='tab-1',
                            children=[
                                dcc.Tab(label='A', value='tab-1', style={"font-size":"16px", "text-align": "center", **dark_mode_styles}, selected_style={"background-color": "#333", "color": "#ffffff"}),
                                dcc.Tab(label='B', value='tab-2', style=dark_mode_styles, selected_style={"background-color": "#333", "color": "#ffffff"}),
                                dcc.Tab(label='C', value='tab-3', style=dark_mode_styles, selected_style={"background-color": "#333", "color": "#ffffff"}),
                            ],
                            colors={
                                "border": "#333",
                                "primary": "#ff5733",
                                "background": "#1e1e1e"
                            },
                        ),
                        html.Div(id='tabs-content-example', style={"flex": "1", **dark_mode_styles}),
                    ],
                    style={"flex": "1", "overflow": "auto"}
                )
            ],
            style={"display": "flex", "height": "100vh", "width": "100vw", "overflow": "hidden"}
            
        ),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
    ])
@app.callback(
    dash.dependencies.Output('tabs-content-example', 'children'),
    [dash.dependencies.Input('tabs-example', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(
            [
                dcc.Graph(id='time-series-chart', style={"width": "100%", "height": "33vh"}),
                dcc.Graph(id='temp-humidity-chart', style={"width": "100%", "height": "33vh"}),
                dcc.Graph(id='scatter-chart', style={"width": "100%", "height": "33vh"}),
            ]
        )
    elif tab == 'tab-2':
        return html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='ammonia-variations', style={"width": "48%", "height": "45vh"}),
                        dcc.Graph(id='No2-variations', style={"width": "48%", "height": "45vh"}),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "padding": "10px"}
                ),
                html.Div(
                    [
                        dcc.Graph(id='Pm-variations', style={"width": "48%", "height": "45vh"}),
                        dcc.Graph(id='VOC-variations', style={"width": "48%", "height": "45vh"}),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "padding": "10px"}
                ),
            ],
            style={"text-align": "center", "padding": "20px", **dark_mode_styles}
        )
    elif tab == 'tab-3':
        return html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='li-variations', style={"width": "48%", "height": "45vh"}),
                        dcc.Graph(id='Np-variations', style={"width": "48%", "height": "45vh"}),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "padding": "10px"}
                ),
                html.Div(
                    [
                        dcc.Graph(id='Na-variations', style={"width": "48%", "height": "45vh"}),
                        dcc.Graph(id='H2s-variations', style={"width": "48%", "height": "45vh"}),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "padding": "10px"}
                ),
            ],
            style={"text-align": "center", "padding": "20px", **dark_mode_styles}
        )

@app.callback(
    dash.dependencies.Output('time-series-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_time_series_chart(n_intervals, active_tab):
    if active_tab != 'tab-1':
        return no_update
    chunk = get_next_data_chunk()
    fig = px.line(chunk, x='Time', y='Tm', title='Temperature Over Time')
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True
        )
    )
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5m", step="minute", stepmode="backward"),
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor="#1e1e1e",
        )
    )
    return fig


@app.callback(
    dash.dependencies.Output('temp-humidity-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_temp_humidity_chart(n_intervals, active_tab):
    if active_tab!='tab-1':
        return no_update
    chunk = get_next_data_chunk()
    df_melted = chunk.melt(id_vars='Time', value_vars=['Tm', 'Rh'], 
                           var_name='Variable', value_name='Value')
    fig = px.line(df_melted, x='Time', y='Value', color='Variable',
                  
                  labels={'Value': 'Values', 'Time': 'Hour'},
                  title='Temperature and Humidity by Hour')
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True
        )
    )
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5m", step="minute", stepmode="backward"),
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor="#1e1e1e",
        )
    )
    return fig

@app.callback(
    dash.dependencies.Output('scatter-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_scatter_plot(n_intervals, active_tab):
    if active_tab!='tab-1':
        return no_update
    chunk = get_next_data_chunk()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chunk['Li'], 
        y=chunk['Tm'], 
        mode='markers', 
        name='Li vs. Tm',
        marker=dict(size=10, color='blue')
    ))
    slope, intercept = np.polyfit(chunk['Li'], chunk['Tm'], 1)
    trendline = np.array(chunk['Li']) * slope + intercept
    fig.add_trace(go.Scatter(
        x=chunk['Li'], 
        y=trendline, 
        mode='lines', 
        name='Trendline',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="Light Intensity vs. Temperature",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Light Intensity",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Temperature",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e",  
        paper_bgcolor="#1e1e1e",  
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True
    )
    return fig

@app.callback(
    dash.dependencies.Output('Pm-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_pm_chart(n_interval, active_tab):
    if active_tab!='tab-2':
        return no_update
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Pm2'],
        mode='lines',
        name='Pm2',
    ))
    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Pm10'],
        mode='lines',
        name='Pm10',
    ))

    fig.update_layout(
        title="Particulate Matter by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 100]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,  
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('VOC-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_VOC_chart(n_intervals, active_tab):
    if active_tab!='tab-2':
        return no_update
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['Voc'],
        mode='lines',
        name='Pm2',
    ))
    fig.update_layout(
        title="Volatile Organic Compounds by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 500]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True, 
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('No2-variations', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'),
     dash.dependencies.Input('tabs-example', 'value')]
)
def update_VOC_chart(n_intervals, active_tab):
    if active_tab!='tab-2':
        return no_update
    chunk = get_next_data_chunk()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk['Time'],
        y=chunk['No2'],
        mode='lines',
        name='Pm2',
    ))
    fig.update_layout(
        title="No2 by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 2]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('ammonia-variations', 'figure'),
    [dash.dependencies.Input('ammonia-variations', 'id')]
)
def update_VOC_chart(_):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Nh3'],
        mode='lines',
        name='Pm2',
    ))
    fig.update_layout(
        title="Ammonia by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 500]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            range=[past_24_hours, now],  
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('li-variations', 'figure'),
    [dash.dependencies.Input('li-variations', 'id')]
)
def update_Li_chart(_):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Li'],
        mode='lines',
        name='Li',
    ))
    fig.update_layout(
        title="Light Intensity by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 500]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            range=[past_24_hours, now],  
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('Np-variations', 'figure'),
    [dash.dependencies.Input('Np-variations', 'id')]
)
def update_Np_chart(_):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Np'],
        mode='lines',
        name='Np',
    ))
    fig.update_layout(
        title="Nitrogen Oxides by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 500]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            range=[past_24_hours, now],  
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig


@app.callback(
    dash.dependencies.Output('Na-variations', 'figure'),
    [dash.dependencies.Input('Na-variations', 'id')]
)
def update_Na_chart(_):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Na'],
        mode='lines',
        name='Pm2',
    ))
    fig.update_layout(
        title="Na by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 70]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            range=[past_24_hours, now],  
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('H2s-variations', 'figure'),
    [dash.dependencies.Input('H2s-variations', 'id')]
)
def update_H2s_chart(_):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['H2s'],
        mode='lines',
        name='H2s',
    ))
    fig.update_layout(
        title="Hydrogen Sulphide by Hour",
        title_font_family="Arial, sans-serif",
        title_font_size=10,
        xaxis_title="Hour",
        xaxis_title_font_family="Arial, sans-serif",
        xaxis_title_font_size=9,
        yaxis_title="Values",
        yaxis_title_font_family="Arial, sans-serif",
        yaxis_title_font_size=9,
        font=dict(family="Arial, sans-serif", size=9, color="#ffffff"),
        template="plotly_dark",
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(range=[0, 2]), 
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            range=[past_24_hours, now],  
        )
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=3600000 
    )
    return fig

@app.callback(
    dash.dependencies.Output('temperature-led', 'value'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_temperature_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_temp = chunk['Tm'].iloc[-1]  
    return f"{latest_temp:.2f}"  

@app.callback(
    dash.dependencies.Output('ammonia-led', 'value'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_ammonia_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_nh3 = chunk['Nh3'].iloc[-1]  
    return f"{latest_nh3:.2f}" 

@app.callback(
    dash.dependencies.Output('h2s-led', 'value'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_h2s_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_h2s = chunk['H2s'].iloc[-1]  
    return f"{latest_h2s:.2f}" 

@app.callback(
    dash.dependencies.Output('No2-led', 'value'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_no2_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_no2 = chunk['No2'].iloc[-1]  
    return f"{latest_no2:.2f}" 

@app.callback(
    dash.dependencies.Output('Voc-led', 'value'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_voc_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_voc = chunk['Voc'].iloc[-1]  
    return f"{latest_voc:.2f}"

@app.callback(
    dash.dependencies.Output('Pm10-led', 'value'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_pm10_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_pm10 = chunk['Pm10'].iloc[-1]  
    return f"{latest_pm10:.2f}"

@app.callback(
    dash.dependencies.Output('Pm2-led', 'value'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_pm2_led(n_intervals):
    chunk = get_next_data_chunk()
    latest_pm2 = chunk['Pm2'].iloc[-1]  
    return f"{latest_pm2:.2f}"


if __name__ == '__main__':
    app.run_server(debug=True)