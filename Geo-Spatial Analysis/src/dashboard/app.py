import os
import math
import random
from datetime import datetime, date, timedelta
from typing import List, Dict

import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import dash_daq as daq
COUNTRY_COORDS = {
    'China': (35.8617, 104.1954),
    'Iran': (32.4279, 53.6880),
    'Italy': (41.8719, 12.5674),
    'Korea, South': (35.9078, 127.7669),
    'India': (20.5937, 78.9629),
    'United States': (37.0902, -95.7129),
    'Brazil': (-14.2350, -51.9253),
    'Russia': (61.5240, 105.3188),
    'Spain': (40.4637, -3.7492),
    'Germany': (51.1657, 10.4515),
    'United Kingdom': (55.3781, -3.4360),
    'France': (46.2276, 2.2137),
    'Turkey': (38.9637, 35.2433),
    'Japan': (36.2048, 138.2529),
    'Australia': (-25.2744, 133.7751),
    'Canada': (56.1304, -106.3468),
    'Mexico': (23.6345, -102.5528),
    'Indonesia': (-0.7893, 113.9213),
    'Argentina': (-38.4161, -63.6167),
    'South Africa': (-30.5595, 22.9375),
}

METRICS = ['Confirmed', 'Recovered', 'Deaths', 'Active']
BLOOD_TYPES = ['(All)', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
def synthesize_hospitals() -> Dict[str, List[Dict]]:
    hospitals_by_country: Dict[str, List[Dict]] = {}
    for country, (lat, lon) in COUNTRY_COORDS.items():
        rnd = random.Random(hash(country) & 0xFFFFFFFF)
        hospitals: List[Dict] = []
        for i in range(12):
            dlat = (rnd.uniform(-0.7, 0.7))
            dlon = (rnd.uniform(-0.7, 0.7))
            hospitals.append({
                'name': f"{country} General Hospital #{i+1}",
                'city': f"City {i+1}",
                'latitude': lat + dlat,
                'longitude': lon + dlon,
            })
        hospitals_by_country[country] = hospitals
    return hospitals_by_country
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
HOSPITALS = synthesize_hospitals()
def synthesize_dataset() -> List[Dict]:
    start = date(2020, 1, 22)
    end = date(2020, 3, 11)
    days = (end - start).days + 1
    countries = list(COUNTRY_COORDS.keys())
    blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    data: List[Dict] = []
    totals = {
        'China': 80000,
        'Iran': 12000,
        'Italy': 15000,
        'Korea, South': 8000,
        'India': 5000,
        'United States': 120000,
        'Brazil': 25000,
        'Russia': 18000,
        'Spain': 20000,
        'Germany': 16000,
        'United Kingdom': 14000,
        'France': 12000,
        'Turkey': 9000,
        'Japan': 7000,
        'Australia': 3000,
        'Canada': 8000,
        'Mexico': 6000,
        'Indonesia': 4000,
        'Argentina': 3500,
        'South Africa': 2000,
    }
    immigrant_pct = {
        'China': 0.15,
        'Iran': 0.03,
        'Italy': 0.10,
        'Korea, South': 0.05,
        'India': 0.02,
        'United States': 0.25,
        'Brazil': 0.08,
        'Russia': 0.12,
        'Spain': 0.15,
        'Germany': 0.18,
        'United Kingdom': 0.20,
        'France': 0.16,
        'Turkey': 0.05,
        'Japan': 0.03,
        'Australia': 0.22,
        'Canada': 0.28,
        'Mexico': 0.04,
        'Indonesia': 0.02,
        'Argentina': 0.06,
        'South Africa': 0.07,
    }
    blood_type_dist = {
        'A+': 0.35, 'A-': 0.06, 'B+': 0.08, 'B-': 0.02,
        'AB+': 0.03, 'AB-': 0.01, 'O+': 0.37, 'O-': 0.08
    }    
    for c in countries:
        lat, lon = COUNTRY_COORDS[c]
        target = totals[c]
        for i in range(days):
            d = start + timedelta(days=i)
            x = i / days
            confirmed = int(target * (1 / (1 + math.exp(-10 * (x - 0.5)))))
            recovered = int(confirmed * 0.55)
            deaths = int(confirmed * 0.03)            
            for blood_type in blood_types:
                blood_cases = int(confirmed * blood_type_dist[blood_type])
                blood_recovered = int(blood_cases * 0.55)
                blood_deaths = int(blood_cases * 0.03)
                immigrant_cases = int(blood_cases * immigrant_pct[c])
                data.append({
                    'date': d,
                    'country': c,
                    'latitude': lat,
                    'longitude': lon,
                    'blood_type': blood_type,
                    'Confirmed': blood_cases,
                    'Recovered': blood_recovered,
                    'Deaths': blood_deaths,
                    'Active': blood_cases - blood_recovered - blood_deaths,
                    'Immigrant': immigrant_cases,
                })
    return data
DATA = synthesize_dataset()
ALL_COUNTRIES = ['(All)'] + sorted(set(row['country'] for row in DATA))
DATE_MIN = min(row['date'] for row in DATA)
DATE_MAX = max(row['date'] for row in DATA)
def aggregate_timeseries(data: List[Dict], country: str, metric: str, blood_type: str = '(All)') -> List[Dict]:
    sums: Dict[date, int] = {}
    for row in data:
        if country != '(All)' and row['country'] != country:
            continue
        if blood_type != '(All)' and row.get('blood_type', '') != blood_type:
            continue
        d = row['date']
        sums[d] = sums.get(d, 0) + int(row[metric])
    series = [{'date': d, metric: v} for d, v in sorted(sums.items())]
    return series
def top_countries(data: List[Dict], as_of: date, metric: str, blood_type: str = '(All)', limit: int = 10) -> List[Dict]:
    country_data: Dict[str, Dict[str, Dict[str, int]]] = {}
    for row in data:
        if row['date'] == as_of:
            if blood_type != '(All)' and row.get('blood_type', '') != blood_type:
                continue
            country = row['country']
            blood_type_row = row.get('blood_type', 'Unknown')
            deaths = int(row['Deaths'])
            immigrant_cases = int(row['Immigrant'])
            if country not in country_data:
                country_data[country] = {'blood_types': {}, 'immigrant_total': 0}
            if blood_type_row not in country_data[country]['blood_types']:
                country_data[country]['blood_types'][blood_type_row] = 0            
            country_data[country]['blood_types'][blood_type_row] += deaths
            country_data[country]['immigrant_total'] += immigrant_cases
    results = []
    for country, data in country_data.items():
        blood_types = data['blood_types']
        total_deaths = sum(blood_types.values())
        immigrant_total = data['immigrant_total']
        majority_blood_group = max(blood_types.items(), key=lambda x: x[1])[0] if blood_types else 'Unknown'
        majority_deaths = blood_types.get(majority_blood_group, 0)        
        blood_group_desc = ", ".join([f"{bt}: {deaths}" for bt, deaths in sorted(blood_types.items())])        
        results.append({
            'country': country,
            'total_deaths': total_deaths,
            'majority_blood_group': majority_blood_group,
            'majority_deaths': majority_deaths,
            'blood_groups_breakdown': blood_group_desc,
            'immigrant_cases': immigrant_total
        })
    results.sort(key=lambda x: x['total_deaths'], reverse=True)
    return results[:limit]
def world_map_figure(data: List[Dict], as_of: date, metric: str) -> go.Figure:
    rows = [row for row in data if row['date'] == as_of]
    values = [max(0, int(row[metric])) for row in rows]
    if values:
        maxv = max(values)
    else:
        maxv = 1
    sizes = [math.sqrt(v) / math.sqrt(maxv) * 20 + 5 for v in values]
    colors = {
        'Confirmed': '#2a6fdb',
        'Recovered': '#2ab673', 
        'Deaths': '#7b3fa0',
        'Active': '#3fa0db'
    }
    fig = go.Figure(
        data=go.Scattergeo(
            lat=[row['latitude'] for row in rows],
            lon=[row['longitude'] for row in rows],
            text=[f"{row['country']}: {int(row[metric])}" for row in rows],
            marker=dict(
                size=sizes,
                color=colors[metric],
                opacity=0.8,
                line=dict(width=1, color='rgba(0,0,0,0.2)'),
            ),
            mode='markers'
        )
    )
    fig.update_geos(
        showcountries=True, 
        showland=True, 
        landcolor='#f5f5f5',
        projection_scale=1.2,
        showframe=False,
        coastlinecolor='#cccccc',
        countrycolor='#cccccc',
        showocean=True,
        oceancolor='#ffffff'
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        font=dict(color='#000000')
    )
    return fig
def metric_timeseries_figure(series: List[Dict], metric: str) -> go.Figure:
    x = [row['date'] for row in series]
    y = [row[metric] for row in series]
    colors = {
        'Confirmed': '#2a6fdb',
        'Recovered': '#2ab673',
        'Deaths': '#7b3fa0',
        'Active': '#3fa0db'
    }
    
    fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color=colors[metric])])
    fig.update_layout(
        margin=dict(l=20, r=10, t=30, b=0), 
        height=300, 
        title=metric, 
        xaxis_title=None, 
        yaxis_title=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10, color='#000000'),
        xaxis=dict(
            gridcolor='#e0e0e0',
            gridwidth=0.5,
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#e0e0e0',
            gridwidth=0.5,
            showgrid=True
        )
    )
    return fig
app = dash.Dash(__name__)
app.title = 'Geo-Spatial COVID-19 Dashboard'
app.layout = html.Div([
    html.Div([
        html.H1('Geo-Spatial COVID-19 Dashboard', style={'margin': '0'}),
        html.Div(f"{DATE_MIN:%A, %B %d, %Y} - {DATE_MAX:%A, %B %d, %Y}", style={'color': '#ddd', 'fontSize': '14px'}),
    ], style={'background': '#23344d', 'color': 'white', 'padding': '12px 16px'}),
    html.Div([
        html.Div([
            html.Label('Date'),
            dcc.DatePickerRange(id='date-range', min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX, start_date=DATE_MIN, end_date=DATE_MAX),
        ], style={'marginRight': '24px'}),
        html.Div([
            html.Label('Select Metric'),
            dcc.Dropdown(id='metric', options=[{'label': m, 'value': m} for m in METRICS], value='Confirmed', clearable=False, style={'width': '220px'}),
        ], style={'marginRight': '24px'}),
        html.Div([
            html.Label('Select Country'),
            dcc.Dropdown(id='country', options=[{'label': c, 'value': c} for c in ALL_COUNTRIES], value='(All)', clearable=False, style={'width': '260px'}),
        ], style={'marginRight': '24px'}),
        html.Div([
            html.Label('Select Blood Type'),
            dcc.Dropdown(id='blood-type', options=[{'label': bt, 'value': bt} for bt in BLOOD_TYPES], value='(All)', clearable=False, style={'width': '200px'}),
        ]),
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '16px', 'padding': '12px 16px'}),
    html.Div([
        html.Div([dcc.Graph(id='ts-confirmed', style={'height': '300px'})], style={'flex': 1, 'padding': '8px'}),
        html.Div([dcc.Graph(id='ts-recovered', style={'height': '300px'})], style={'flex': 1, 'padding': '8px'}),
        html.Div([dcc.Graph(id='ts-deaths', style={'height': '300px'})], style={'flex': 1, 'padding': '8px'}),
        html.Div([dcc.Graph(id='ts-active', style={'height': '300px'})], style={'flex': 1, 'padding': '8px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '0 16px'}),
    html.Div([
        html.Div([dcc.Graph(id='map-confirmed', style={'height': '400px'})], style={'flex': 1, 'padding': '8px', 'minWidth': '300px'}),
        html.Div([dcc.Graph(id='map-recovered', style={'height': '400px'})], style={'flex': 1, 'padding': '8px', 'minWidth': '300px'}),
        html.Div([dcc.Graph(id='map-deaths', style={'height': '400px'})], style={'flex': 1, 'padding': '8px', 'minWidth': '300px'}),
        html.Div([dcc.Graph(id='map-active', style={'height': '400px'})], style={'flex': 1, 'padding': '8px', 'minWidth': '300px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '0 16px'}),
    html.Div([
        html.H4('Top Countries'),
        dash_table.DataTable(
            id='top-table', 
            columns=[
                {'name': 'Country', 'id': 'country'},
                {'name': 'Total Deaths', 'id': 'total_deaths'},
                {'name': 'Majority Blood Group', 'id': 'majority_blood_group'},
                {'name': 'Majority Deaths', 'id': 'majority_deaths'},
                {'name': 'Blood Groups Breakdown', 'id': 'blood_groups_breakdown'},
                {'name': 'Immigrant Cases', 'id': 'immigrant_cases'}
            ], 
            style_as_list_view=True, 
            style_cell={'padding': '6px', 'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
            style_header={'fontWeight': 'bold'}, 
            style_data_conditional=[
                {
                    'if': {'column_id': 'total_deaths'},
                    'backgroundColor': '#fff3cd',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'majority_blood_group'},
                    'backgroundColor': '#d1ecf1',
                    'fontWeight': 'bold'
                }
            ],
            page_size=10
        ),
    ], style={'padding': '16px'}),

    html.Div([
        html.H4('Hospitals Near Hotzones'),
        dash_table.DataTable(
            id='hospital-table',
            columns=[
                {'name': 'Hospital', 'id': 'name'},
                {'name': 'City', 'id': 'city'},
                {'name': 'Distance (km)', 'id': 'distance_km'},
            ],
            style_as_list_view=True,
            style_cell={'padding': '6px', 'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
            style_header={'fontWeight': 'bold'},
            page_size=8
        ),
        html.Div('(Select a country to see nearby hospitals)', id='hospital-hint', style={'fontSize': '12px', 'color': '#666', 'marginTop': '6px'})
    ], style={'padding': '16px'}),
])
@app.callback(
    [Output('ts-confirmed', 'figure'),
     Output('ts-recovered', 'figure'),
     Output('ts-deaths', 'figure'),
     Output('ts-active', 'figure'),
     Output('map-confirmed', 'figure'),
     Output('map-recovered', 'figure'),
     Output('map-deaths', 'figure'),
     Output('map-active', 'figure'),
     Output('top-table', 'data'),
     Output('hospital-table', 'data')],
    [Input('date-range', 'start_date'), Input('date-range', 'end_date'), Input('country', 'value'), Input('metric', 'value'), Input('blood-type', 'value')]
)
def update_dashboard(start_date, end_date, country, metric, blood_type):
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    filtered = [row for row in DATA if start <= row['date'] <= end]
    ts_conf = aggregate_timeseries(filtered, country, 'Confirmed', blood_type)
    ts_rec = aggregate_timeseries(filtered, country, 'Recovered', blood_type)
    ts_dea = aggregate_timeseries(filtered, country, 'Deaths', blood_type)
    ts_act = aggregate_timeseries(filtered, country, 'Active', blood_type)
    fig_ts_conf = metric_timeseries_figure(ts_conf, 'Confirmed')
    fig_ts_rec = metric_timeseries_figure(ts_rec, 'Recovered')
    fig_ts_dea = metric_timeseries_figure(ts_dea, 'Deaths')
    fig_ts_act = metric_timeseries_figure(ts_act, 'Active')
    as_of = max(row['date'] for row in filtered) if filtered else DATE_MAX
    fig_map_conf = world_map_figure(filtered, as_of, 'Confirmed')
    fig_map_rec = world_map_figure(filtered, as_of, 'Recovered')
    fig_map_dea = world_map_figure(filtered, as_of, 'Deaths')
    fig_map_act = world_map_figure(filtered, as_of, 'Active')
    table_data = top_countries(filtered, as_of, metric, blood_type)
    hospitals_data: List[Dict] = []
    if country and country != '(All)' and country in COUNTRY_COORDS:
        c_lat, c_lon = COUNTRY_COORDS[country]
        hospitals = HOSPITALS.get(country, [])
        ranked = []
        for h in hospitals:
            dist = haversine_km(c_lat, c_lon, h['latitude'], h['longitude'])
            ranked.append({
                'name': h['name'],
                'city': h['city'],
                'distance_km': round(dist, 1)
            })
        ranked.sort(key=lambda x: x['distance_km'])
        hospitals_data = ranked[:8]
    return (fig_ts_conf, fig_ts_rec, fig_ts_dea, fig_ts_act,
            fig_map_conf, fig_map_rec, fig_map_dea, fig_map_act,
            table_data, hospitals_data)
if __name__ == '__main__':
    app.run_server(debug=True)