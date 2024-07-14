import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import string

import callbacks

#####################################
########### Online Assets ###########
#####################################

# FONT_AWESOME = "https://use.fontawesome.com/releases/v5.13.0/css/all.css"

######################################
########## Global Variables ##########
######################################

rows = list(string.ascii_uppercase[:8])# A to H
cols = list(range(1, 13)) # 1 to 12
wellplate_ids = [f"{row}{col}"
    for row in rows
    for col in cols]

plt_options = [
    {"label": "All Wells", "value": "plt-all"},
    {"label": "Rows View", "value": "plt-rows"},
    {"label": "Columns View", "value": "plt-cols"},
    {"label": "Custom Wells", "value": "plt-custom"}
]
time_unit_options = [
    {"label": "Seconds", "value": "unit-sec"},
    {"label": "Minutes", "value": "unit-min"}
]
download_data = [
    {"label": "Original Data", "value": "select-full-raw"}, # Raw for full dataframe
    {"label": "Full Temperature Rate", "value": "select-full-rate"},
    {"label": "Selected Temperature", "value": "select-current-temp"}, # for custom selection temp
    {"label": "Selected Rate", "value": "select-current-rate"} # for custom selection rate
]
adv_options = [
    {"label": "Comparision", "value": "adv-compare"},
    {"label": "Analysis", "value": "adv-analysis"}
]
adv_operation_options = [
    {"label": "Fit Simple Linear Regression", "value": "fit-sim-lin"},
    {"label": "Fit Exponential Growth", "value": "fit-expo-growth"},
    {"label": "Fit Exponential Decay", "value": "fit-expo-decay"},
    {"label": "Fit Polynomial", "value": "fit-poly"},
    {"label": "Fit Newtonian Cooling", "value": "fit-new-cool"},
    # {"label": "Forecast Smoothing", "value": "fore-smooth"},
    # {"label": "Forecast Moving-average", "value": "fore-mov-avg"},
    # {"label": "Forecast Exponential", "value": "fore-expo"},
]
download_adv_data = [
    {"label": "Plot Data", "value": "select-plt-data"},
    {"label": "Analysis Results", "value": "select-ana-res"}
]

######################################
######### Define Application ######### 
######################################

app = dash.Dash(__name__, title="Thermal Gradient Dashboard", external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server # required by dash-tools to host on Render

######### Dashboard Elements ########
DASHBOARD_DETAILS = """This is a dashboard application to view temperature changes versus time."""

UPLOAD_SECTION = dbc.Card([
    dbc.CardHeader(html.H4("Upload File", className="text-center")),
    dbc.CardBody([
        html.Div([
            dcc.Upload(dbc.Button("Upload", className="upload-button"), id="upload-data", multiple=True),
            html.Div("No files uploaded", id="upload-status-txt", className="upload-status-text")
        ], className="upload-container")
    ])
], className="upload-section mb-3")

## Static Plot
MULTI_PLOT_SELECTION = dbc.Card([
    dbc.CardHeader(html.H4("Multiple Wells View", className="text-center")),
    dbc.CardBody([
        html.Div([
            html.Label("Select File:"),
            dcc.Dropdown(id="upload-file-dropdown", 
                placeholder="Select File(s)", 
                multi=False)
        ], className="pt-1"),
        html.Div([
            html.Label("Plot Option:"),
            dcc.Dropdown(id="plt-option", 
                options=plt_options, 
                value="plt-all", 
                searchable=False)
        ], className="pt-1"),
        html.Div([
            html.Label("Select Wells Group:"),
            dcc.Dropdown(id="select-row-column", 
                multi=True, 
                searchable=False)
        ], className="pt-1"),
        html.Div([
            html.Label("Select Time Unit:"),
            dcc.Dropdown(id="select-time-unit", 
                options=time_unit_options, 
                value="unit-sec", 
                searchable=False)
        ], className="pt-1"),
        html.Div([
            html.Label("Select Data to Download:"),
            dcc.Dropdown(id="select-download-data", 
                options=download_data, 
                value="select-full-raw",
                searchable=False,
                clearable=False)
        ], className="pt-1"),
        html.Div([
            dbc.Button("Download CSV",
                id="btn-download-multi", 
                className="download-btn ml-auto mt-3"),
            dcc.Download(id="download-multi")
        ], className="d-flex justify-content-end")
    ])
], className="mb-3")

MULTI_PLOT_AREA = dbc.Card([
    dbc.CardHeader(
        dbc.Tabs([
            dbc.Tab(label="Temperature vs Time", tab_id="tab-temp-time"),
            dbc.Tab(label="Rate of Change vs Time", tab_id="tab-rate-time"),
            dbc.Tab(label="Temperature Data", tab_id="tab-data-table", id="tab-data-table"),
            dbc.Tab(label="Rate Data", tab_id="tab-rate-table", id="tab-rate-table")], 
            id="multi-tabs", active_tab="tab-temp-time")
    ),
    dbc.CardBody(
        dbc.Spinner([html.Div(id="multi-plot-area", className="p-2")])
    )
], className="plot-area mb-3")

## Customizable Plot
CUSTOM_PLOT_SELECTION = dbc.Card([
    dbc.CardHeader(html.H4("Data Analysis", className="text-center")),
    dbc.CardBody([
        html.Div([
            html.Label("Select Analysis Mode:"),
            dcc.Dropdown(id="adv-mode",
                options=adv_options,
                value="adv-compare",
                searchable=False
            )
        ], className="pt-1"),
        html.Div([
            html.Label("Select Files:"),
            dcc.Dropdown(id="adv-file-dropdown",
                placeholder="Select File(s)",
                multi=True
            )
        ], className="pt-1"),
        html.Div([
            html.Label("Select Wells:"),
            dcc.Dropdown(id="adv-well-select",
                options=wellplate_ids,
                placeholder="Select well(s)",
                multi=True,
                searchable=False
            )
        ], className="pt-1"),
        html.Div([
            html.Label("Select Time Unit:"),
            dcc.Dropdown(id="adv-time-unit",
                options=time_unit_options,
                value="unit-sec",
                searchable=False
            )
        ], className="pt-1"),
        html.Div([
            html.Label("Select Operation:"),
            dcc.Dropdown(id="adv-operation",
                options=adv_operation_options,
                searchable=False
            )
        ], className="pt-1"),
        ## Hidden Divs
        html.Div([
            html.Div([
                html.Label("Enter Polynomial Degree (2-10):"),
                dbc.Input(id="poly-deg", type="number", min=2, max=10, step=1, value=2)
                ], className="pt-1", id="container-poly-deg",style={"display": "none"}),
            html.Div([
                html.Label("Enter Room Temperature:"),
                dbc.Input(id="room-temp", type="number", min=10, max=30, step=0.1, value=20)
                ], className="pt-1", id="container-room-temp", style={"display": "none"}),
        ],id="hidden-container-parent"),
        html.Div([
            html.Label("Select to Download:"),
            dcc.Dropdown(id="select-adv-dl-data",
                options=download_adv_data,
                value="select-plt-data",
                searchable=False,
                clearable=False
            )
        ]),
        ## Button
        html.Div([
            dbc.Button("Download CSV",
            id="btn-download-adv",
            className="download-btn ml-auto mt-3"
            ),
            dcc.Download(id="download-adv")
        ], className="d-flex justify-content-end")
    ])
], className="mb-3")

CUSTOM_PLOT_AREA = dbc.Card([
    dbc.CardHeader(
        dbc.Tabs([
            dbc.Tab(label="Plot Area", tab_id="tab-adv-plot", id="tab-adv-plt"),
            dbc.Tab(label="Plot Analysis", tab_id="tab-adv-analysis", id="tab-ana"),
            dbc.Tab(label="Plot Data", tab_id="tab-adv-plot-data", id="tab-adv-data"),
            dbc.Tab(label="Analysis Results", tab_id="tab-adv-ana-res", id="tab-ana-res")],
            id="adv-tabs", 
            active_tab="tab-adv-plot")
    ),
    dbc.CardBody(
        dbc.Spinner([html.Div(id="custom-plot-area", className="p-2")])
    )
], className="custom-plot-area mb-3")

## Footer
FOOTER = html.P(["© Copyright 2024 Jira Leelasoontornwatana"], className="text-center pt-3")

## Application Layout
app.layout = dbc.Container([

    ## Header and brief details
    dbc.Row(dbc.Col(html.H1("Thermal Gradient Dashboard", className="text-center my-4"), width=12)),
    dcc.Markdown(DASHBOARD_DETAILS),
    html.Hr(),

    ## Upload Data Storage
    dcc.Store(id="upload-data-storage", storage_type="session"),

    ## Control and Plots
    dbc.Row([
        dbc.Col([UPLOAD_SECTION, # Upload Section
            MULTI_PLOT_SELECTION], md=3), # Control for Multiple plots
        dbc.Col(MULTI_PLOT_AREA, md=9)
    ]),
    
    # Plot Area 2 Customizable Plot
    dbc.Row([
        dbc.Col([CUSTOM_PLOT_SELECTION], md=3),
        dbc.Col([CUSTOM_PLOT_AREA], md=9)
    ]),

    ## Footer
    dbc.Row([
        dbc.Col([html.Footer([FOOTER])
        ])
    ])
    
],fluid=True)

## Start Application
if __name__ == "__main__":
    app.run(debug=True)
