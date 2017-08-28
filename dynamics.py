''' Programs to analyze the damped single degree of freedom system
    Date: 26 August 2017    Time:   7:43 PM
    Author: Ajit Sharma
'''

import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go


# displacement at any time t
def disp_damped_free(wn, v0, x0, _zeta, t):
    wd = wn*np.sqrt(1 - _zeta*_zeta)
    A = np.sqrt((np.square(v0 + _zeta*wn*x0) + np.square(x0*wd))/np.square(wd))
    phi = np.arctan(x0*wd/(v0 + _zeta*wn*x0))

    disp = A*np.exp(-_zeta*wn*t)*np.sin(wd*t + phi)

    return disp

# velocity at any time t
def vel_damped_free(wn, v0, x0, _zeta, t):
    wd = wn*np.sqrt(1 - _zeta*_zeta)
    A = np.sqrt((np.square(v0 + _zeta*wn*x0) + np.square(x0*wd))/np.square(wd))
    phi = np.arctan(x0*wd/(v0 + _zeta*wn*x0))

    vel = A*np.exp(-_zeta*wn*t)*np.sin(wd*t + phi)*(-1*_zeta*wn) + A*np.exp(-1*_zeta*wn*t)*np.cos(wd*t + phi)*wd

    return vel

# overdamped: displacement at any time t
def disp_over_damped_free(wn, v0, x0, _zeta, t):
    wf = wn*np.sqrt(_zeta*_zeta - 1)
    lambda_pos = -_zeta*wn + wf
    lambda_neg = -_zeta*wn - wf
    a1 = (-v0 + lambda_pos*x0)/(2*wf)
    a2 = (-v0 - lambda_neg*x0)/(2*wf)

    disp = np.exp(-_zeta*wn*t)*(a1 * np.exp(wf*t) + a2 * np.exp(-wf*t))

    return disp

# overdamped: velocity at any time t
def vel_over_damped_free(wn, v0, x0, _zeta, t):
    wf = wn*np.sqrt(_zeta*_zeta - 1)
    lambda_pos = -_zeta*wn + wf
    lambda_neg = -_zeta*wn - wf
    a1 = (-v0 + lambda_pos*x0)/(2*wf)
    a2 = (-v0 - lambda_neg*x0)/(2*wf)

    vel = (a1 * (wf - wn*_zeta)* np.exp((wf - _zeta*wn)*t) - a2 *(wf+_zeta*wn)* np.exp(-(wf+_zeta*wn)*t))

    return vel

# critically damped : displacement at any time t
def disp_critical_damped_free(wn, v0, x0, _zeta, t):
    disp = (x0 + (v0 + wn*x0)*t)*np.exp(-wn*t)

    return disp

# critically damped : velocity at any time t
def vel_critical_damped_free(wn, v0, x0, _zeta, t):
    vel = (v0 + wn*x0)*np.exp(-wn*t) - wn*((x0 + (v0 + wn*x0)*t)*np.exp(-wn*t))
    return vel

# Forced undamped vibrations: transient displacement at time t
def disp_trans_harmonic_undamped(wn, v0, x0, f0, w, t):
    disp = 0
    if w == wn:
        a1 = v0/wn
        a2 = x0
        disp = a1*np.sin(wn*t) + a2*np.cos(wn*t)
    else:
        a1 = v0/wn
        a2 = x0 - f0/(wn*wn - w*w)
        disp = a1*np.sin(wn*t) + a2*np.cos(wn*t)
    return disp

# Forced undamped vibrations: transient displacement at time t
def vel_trans_harmonic_undamped(wn, v0, x0, f0, w, t):
    vel = 0
    if w == wn:
        a1 = v0/wn
        a2 = x0
        vel = a1*np.cos(wn*t) - a2*np.sin(wn*t)
    else:
        a1 = v0/wn
        a2 = x0 - f0/(wn*wn - w*w)
        vel = a1*wn*np.cos(wn*t) - a2*wn*np.sin(wn*t)
    return vel

# forced damped vibrations: transient displacement at any time t
def disp_trans_harmonic_damped(wn, v0, x0, f0, w, zeta, t):
    disp = 0
    if zeta < 1:
        a0 = f0/np.sqrt((wn*wn - w*w)*(wn*wn -w*w) + (2 * zeta * wn * w)*(2 * zeta * wn * w))
        try:
            phi = np.arctan2((2*zeta*wn*w)/(wn*wn - w*w))
        except:
            phi = 3.14159/2
        wd = wn*np.sqrt(1 - zeta * zeta)
        a2 = x0 - a0*np.cos(phi) + a0*np.sin(phi)
        a1 = (v0 + zeta*wn*a2 - a0 * w *np.sin(phi) - a0*w*np.cos(phi))/(wd)
        disp = (a1 * np.sin(wd * t) + a2*np.cos(wd *t))*np.exp(-zeta*wn * t)
    return disp

# forced damped vibrations: transient displacement at any time t
def vel_trans_harmonic_damped(wn, v0, x0, f0, w, zeta, t):
    vel = 0
    if zeta < 1:
        a0 = f0/np.sqrt((wn*wn - w*w)*(wn*w - w*w) + (2 * zeta * wn * w)*(2 * zeta * wn * w))
        try:
            phi = np.arctan2((2*zeta*wn*w)/(wn*wn - w*w))
        except:
            phi = 3.14159/2
        wd = wn*np.sqrt(1 - zeta * zeta)
        a2 = x0 - a0*np.cos(phi) + a0*np.sin(phi)
        a1 = (v0 + zeta*wn*a2 - a0 * w *np.sin(phi) - a0*w*np.cos(phi))/(wd)

        vel = (a1 * np.sin(wd * t)*(-zeta*wn) + a1*wd*np.cos(wd*t) + a2*np.cos(wd *t)*(-zeta*wn) - a2*wd*np.sin(wd*t))*np.exp(-zeta*wn*t)
    return vel

# Forced undamped vibrations: steady displacement at time t
def disp_steady_harmonic_undamped(wn, v0, x0, f0, w, t):
    disp = 0
    if w == wn:
        disp = (f0/2*wn)*t*np.sin(wn*t)
    else:
        disp = (f0/(wn*wn - w*w))*np.cos(w*t)
    return disp

# Forced undamped vibrations: steady displacement at time t
def vel_steady_harmonic_undamped(wn, v0, x0, f0, w, t):
    vel = 0
    if w == wn:
        vel = (f0/2*wn)*np.sin(wn*t) + (f0/2)*t*np.cos(wn*t)
    else:
        vel = (f0/(wn*wn - w*w))*np.sin(w*t) * (-w)
    return vel

# forced damped vibrations: steady displacement at any time t
def disp_steady_harmonic_damped(wn, v0, x0, f0, w, zeta, t):
    disp = 0
    a0 = f0/np.sqrt((wn*wn - w*w) + (2 * zeta * wn * w)*(2 * zeta * wn * w))
    try:
        phi = np.arctan2((2*zeta*wn*w)/(wn*wn - w*w))
    except:
        phi = 3.14159/2
    disp = a0 * np.cos(w*t - phi)
    return disp

# forced damped vibrations: steady velocity at any time t
def vel_steady_harmonic_damped(wn, v0, x0, f0, w, zeta, t):
    a0 = f0/np.sqrt((wn*wn - w*w) + (2 * zeta * wn * w)*(2 * zeta * wn * w))
    try:
        phi = np.arctan2((2*zeta*wn*w)/(wn*wn - w*w))
    except:
        phi = 3.14159/2
    vel = a0 * np.sin(w*t - phi) * (-w)
    return vel

# Forced undamped vibrations: total displacement at time t
def disp_total_harmonic_undamped(wn, v0, x0, f0, w, t):
    disp = disp_trans_harmonic_undamped(wn, v0, x0, f0, w, t) + disp_steady_harmonic_undamped(wn, v0, x0, f0, w, t)
    return disp

# Forced undamped vibrations: total displacement at time t
def vel_total_harmonic_undamped(wn, v0, x0, f0, w, t):
    vel = vel_trans_harmonic_undamped(wn, v0, x0, f0, w, t) + vel_steady_harmonic_undamped(wn, v0, x0, f0, w, t)
    return vel

# forced damped vibrations: total displacement at any time t
def disp_total_harmonic_damped(wn, v0, x0, f0, w, zeta, t):
    disp = disp_trans_harmonic_damped(wn, v0, x0, f0, w, zeta, t) + disp_steady_harmonic_damped(wn, v0, x0, f0, w,zeta, t)
    return disp

# forced damped vibrations: total displacement at any time t
def vel_total_harmonic_damped(wn, v0, x0, f0, w, zeta, t):
    vel = vel_trans_harmonic_damped(wn, v0, x0, f0, w, zeta, t) + vel_steady_harmonic_damped(wn, v0, x0, f0, w,zeta, t)
    return vel

# Calculation of Displacement Response Fcators
def displacement_factor(r, zeta):
    try:
        rd = 1/np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
    except:
        rd = 100
    return rd


app = dash.Dash()

# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app.config.supress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    html.H1('Analysis of SDOF System'),
    dcc.Link('Free Vibrations', href='/free', style = {'text-decoration': 'none', 'font-size': '24px'}),
    html.Br(),
    dcc.Link('Forced Vibrations', href='/forced', style = {'text-decoration': 'none', 'font-size': '24px'}),
    html.Br(),
    dcc.Link('Response Factors', href = '/response-factor', style = {'text-decoration': 'none', 'font-size': '24px'})
], style = {'width': '75%', 'margin':'auto', 'text-align': 'center', 'font-family': 'Impact', 'padding-top': '200'})

free_layout = html.Div([
    html.H1('Free Vibration of SDOF System', style = {'margin':'auto', 'text-align': 'center', 'font-family': 'Impact'}),
    html.Br(),

    html.Div([
        html.Label('Natural Frequency'),
        dcc.Input(id = 'wn', value='50'),

        html.Label('Initial Displacement'),
        dcc.Input(id = 'x0', value='1'),

        html.Label('Initial Velocity'),
        dcc.Input(id = 'v0', value='1'),

        html.Label('Zeta'),
        dcc.Input(id = 'zeta', value='0.1'),

        html.Br(),
        html.Div([
            html.Label('Response Time'),
            dcc.Slider(
                id = 'time_duration',
                min=0,
                max=20,
                marks={i: 'Time {}'.format(i) if i == 1 else str(i) for i in range(1, 21)},
                value=1,
            )
        ], style = {'width': '40%'}),
    ], style = {'width':'75%', 'margin': 'auto'}),

    html.Br(),
    html.Div([
        html.Label('Check Responses'),
        dcc.Checklist(
            id = 'disp_list',
            options=[
                {'label': 'Displacement', 'value': 'disp'},
                {'label': 'Velocity', 'value': 'vel'},
                {'label': 'Acceleration', 'value': 'accn'}
                ],
            values=['disp', 'vel'],
            labelStyle={'display': 'inline-block'})
    ], style = {'width': '75%', 'margin': 'auto'}),

    html.Br(),
    html.Div([
        dcc.Graph(id='free-response-graph', animate=True, style={ 'border': '3px solid #1a53af'})
    ], style={'width': '75%', 'margin': 'auto'}),

    html.Br(),
    html.Div([
        dcc.Link('Forced Vibrations', href='/forced', style = {'text-decoration': 'none', 'font-size': '18px'}),
        html.Br(),
        dcc.Link('Response factors', href='/response-factor', style = {'text-decoration': 'none', 'font-size': '18px'}),
        html.Br(),
        dcc.Link('Go back to home', href='/', style = {'text-decoration': 'none', 'font-size': '18px'}),
    ], style ={'columnCount': '3', 'text-align': 'center', 'width': '75%', 'margin': 'auto'})

])

@app.callback(dash.dependencies.Output('free-response-graph', 'figure'),
              [dash.dependencies.Input('time_duration', 'value'),
               dash.dependencies.Input('x0', 'value'),
               dash.dependencies.Input('v0', 'value'),
               dash.dependencies.Input('wn', 'value'),
               dash.dependencies.Input('zeta', 'value'),
               dash.dependencies.Input('disp_list', 'values')])

def plot_response(time_duration, x0, v0, wn, zeta, disp_list):

    time = np.arange(0, time_duration, 0.001)

    displacement = []
    velocity = []
    acceleration =[]


    if float(zeta) < 1:
        for t in time:
            displacement.append(disp_damped_free(float(wn), float(v0), float(x0), float(zeta), t))
            velocity.append(vel_damped_free(float(wn), float(v0), float(x0), float(zeta), t))
            acceleration.append(vel_damped_free(float(wn), float(v0), float(x0), float(zeta), t))
    elif float(zeta) == 1:
        for t in time:
            displacement.append(disp_critical_damped_free(float(wn), float(v0), float(x0), float(zeta), t))
            velocity.append(vel_critical_damped_free(float(wn), float(v0), float(x0), float(zeta), t))
            acceleration.append(vel_critical_damped_free(float(wn), float(v0), float(x0), float(zeta), t))
    elif float(zeta) > 1:
        for t in time:
            displacement.append(disp_over_damped_free(float(wn), float(v0), float(x0), float(zeta), t))
            velocity.append(vel_over_damped_free(float(wn), float(v0), float(x0), float(zeta), t))
            acceleration.append(vel_over_damped_free(float(wn), float(v0), float(x0), float(zeta), t))

    max_velocity = np.amax(np.abs(velocity))
    max_disp = np.amax(np.abs(displacement))
    max_accn = np.amax(np.abs(acceleration))

    range_list = []
    plot_list = []

    for response in disp_list:
        if response == 'disp':
            plot_list.append(go.Scatter(x = time, y = displacement, mode = 'lines'))
            range_list.append(max_disp)
        elif response == 'vel':
            plot_list.append(go.Scatter(x = time, y = velocity, mode = 'lines'))
            range_list.append(max_velocity)
        elif response == 'accn':
            plot_list.append(go.Scatter(x = time, y = acceleration, mode = 'lines'))
            range_list.append(max_accn)

    max_range = max(range_list)

    return {
        'data': plot_list,
        'layout': go.Layout(
            xaxis={'title': 'Time (s)', 'range': [0, time_duration]},
            yaxis={'range': [-1.25 * max_range, 1.25 * max_range]},
            margin=go.Margin(l=100, r=100, t=50, b=50)
        )
    }



forced_layout = html.Div([
    html.H1('Forced Vibration of SDOF System',style = {'margin':'auto', 'text-align': 'center', 'font-family': 'Impact'}),
    html.Br(),

    html.Div([
        html.Label('Natural Frequency'),
        dcc.Input(id = 'wn', value='50'),

        html.Label('Initial Displacement'),
        dcc.Input(id = 'x0', value='1'),

        html.Label('Initial Velocity'),
        dcc.Input(id = 'v0', value='1'),

        html.Label('Zeta'),
        dcc.Input(id = 'zeta', value='0.1'),

        html.Label('Driving Frequency'),
        dcc.Input(id = 'w', value='30'),

        html.Label('Driving force per unit Mass'),
        dcc.Input(id = 'f0', value='1.0'),


    ], style={'width': '75%', 'margin': 'auto', 'columnCount': '2'}),



    html.Div([
        html.Div([
            html.Label('Response Time'),
            dcc.Slider(
                id = 'time_duration',
                min=0,
                max=20,
                marks={i: 'Time {}'.format(i) if i == 1 else str(i) for i in range(1, 21)},
                value=1,
            )
        ], style={'width': '40%'}),

        html.Br(),
        html.Label('Check Responses'),
        dcc.Checklist(
            id = 'disp_list',
            options=[
                {'label': 'Displacement', 'value': 'disp'},
                {'label': 'Velocity', 'value': 'vel'},
                {'label': 'Acceleration', 'value': 'accn'}
                ],
            values=['disp', 'vel'],
            labelStyle={'display': 'inline-block'}
)
    ], style={'width': '75%', 'margin': 'auto'} ),

    html.Br(),
    html.Div([
        html.Label('Transient Response'),
        dcc.Graph(id='forced-response-transient', animate=True, style={ 'border': '3px solid #1a53af'})
    ], style={'width': '75%', 'margin': 'auto'}),

    html.Br(),
    html.Div([
        html.Label('Steady State Response'),
        dcc.Graph(id='forced-response-steady', animate=True, style={ 'border': '3px solid #1a53af'})
    ], style={'width': '75%', 'margin': 'auto'}),

        html.Br(),
    html.Div([
        html.Label('Total Response'),
        dcc.Graph(id='forced-response-total', animate=True, style={ 'border': '3px solid #1a53af'})
    ], style={'width': '75%', 'margin': 'auto'}),

    html.Br(),
    html.Div([
        dcc.Link('Free Vibrations', href='/free', style = {'text-decoration': 'none', 'font-size': '18px'}),
        html.Br(),
        dcc.Link('Response Factors', href='/response-factor', style = {'text-decoration': 'none', 'font-size': '18px'}),
        html.Br(),
        dcc.Link('Go back to home', href='/', style = {'text-decoration': 'none', 'font-size': '18px'})
    ], style ={'columnCount': '3', 'text-align': 'center', 'width': '75%', 'margin': 'auto', 'paddingBottom': '100px'}),

])

@app.callback(dash.dependencies.Output('forced-response-transient', 'figure'),
                   [dash.dependencies.Input('time_duration', 'value'),
                    dash.dependencies.Input('x0', 'value'),
                    dash.dependencies.Input('v0', 'value'),
                    dash.dependencies.Input('wn', 'value'),
                    dash.dependencies.Input('zeta', 'value'),
                    dash.dependencies.Input('w', 'value'),
                    dash.dependencies.Input('f0', 'value'),
                    dash.dependencies.Input('disp_list', 'values')])

def plot_response_forced(time_duration, x0, v0, wn, zeta, w, f0, disp_list):

    time = np.arange(0, time_duration, 0.001)

    displacement_trans = []
    velocity_trans = []
    acceleration_trans =[]

    if (float(zeta) == 0):
        #forced harmonic Undamped vibrations
        for t in time:
            displacement_trans.append(disp_trans_harmonic_undamped(float(wn), float(v0), float(x0), float(f0), float(w), t))
            velocity_trans.append(vel_trans_harmonic_undamped(float(wn), float(v0), float(x0), float(f0), float(w), t))
            acceleration_trans.append(vel_trans_harmonic_undamped(float(wn), float(v0), float(x0), float(f0), float(w), t))

    elif float(zeta) < 1:
        #Forced Harmonic _transDamped vibrations
        for t in time:
            displacement_trans.append(disp_trans_harmonic_damped(float(wn), float(v0), float(x0), float(f0), float(w), float(zeta), t))
            velocity_trans.append(vel_trans_harmonic_damped(float(wn), float(v0), float(x0), float(f0), float(w), float(zeta), t))
            acceleration_trans.append(vel_trans_harmonic_damped(float(wn), float(v0), float(x0), float(f0), float(w), float(zeta), t))


    max_velocity = np.amax(np.abs(velocity_trans))
    max_disp = np.amax(np.abs(displacement_trans))
    max_accn = np.amax(np.abs(acceleration_trans))

    range_list = []
    plot_list = []

    for response in disp_list:
        if response == 'disp':
            plot_list.append(go.Scatter(x = time, y = displacement_trans, mode = 'lines'))
            range_list.append(max_disp)
        elif response == 'vel':
            plot_list.append(go.Scatter(x = time, y = velocity_trans, mode = 'lines'))
            range_list.append(max_velocity)
        elif response == 'accn':
            plot_list.append(go.Scatter(x = time, y = acceleration_trans, mode = 'lines'))
            range_list.append(max_accn)

    max_range = max(range_list)

    return {
        'data': plot_list,
        'layout': go.Layout(
            xaxis={'title': 'Time (s)', 'range': [0, time_duration]},
            yaxis={'range': [-1.25 * max_range, 1.25 * max_range]},
            margin=go.Margin(l=100, r=100, t=50, b=50)
        )
    }

@app.callback(dash.dependencies.Output('forced-response-steady', 'figure'),
                   [dash.dependencies.Input('time_duration', 'value'),
                    dash.dependencies.Input('x0', 'value'),
                    dash.dependencies.Input('v0', 'value'),
                    dash.dependencies.Input('wn', 'value'),
                    dash.dependencies.Input('zeta', 'value'),
                    dash.dependencies.Input('w', 'value'),
                    dash.dependencies.Input('f0', 'value'),
                    dash.dependencies.Input('disp_list', 'values')])

def plot_response_forced(time_duration, x0, v0, wn, zeta, w, f0, disp_list):

    time = np.arange(0, time_duration, 0.001)

    displacement_steady = []
    velocity_steady = []
    acceleration_steady =[]

    if (float(zeta) == 0):
        #harmonic Undamped vibrations
        for t in time:
            displacement_steady.append(disp_steady_harmonic_undamped(float(wn), float(v0), float(x0), float(f0),float(w), t))
            velocity_steady.append(vel_steady_harmonic_undamped(float(wn), float(v0), float(x0), float(f0),float(w), t))
            acceleration_steady.append(vel_steady_harmonic_undamped(float(wn), float(v0), float(x0), float(f0),float(w), t))
    elif float(zeta)< 1:
        #Harmonic Damped vibrations
        for t in time:
            displacement_steady.append(disp_steady_harmonic_damped(float(wn), float(v0), float(x0), float(f0),float(w), float(zeta), t))
            velocity_steady.append(vel_steady_harmonic_damped(float(wn), float(v0), float(x0), float(f0),float(w), float(zeta), t))
            acceleration_steady.append(vel_steady_harmonic_damped(float(wn), float(v0), float(x0), float(f0),float(w), float(zeta), t))


    max_velocity = np.amax(np.abs(velocity_steady))
    max_disp = np.amax(np.abs(displacement_steady))
    max_accn = np.amax(np.abs(acceleration_steady))

    range_list = []
    plot_list = []

    for response in disp_list:
        if response == 'disp':
            plot_list.append(go.Scatter(x = time, y = displacement_steady, mode = 'lines'))
            range_list.append(max_disp)
        elif response == 'vel':
            plot_list.append(go.Scatter(x = time, y = velocity_steady, mode = 'lines'))
            range_list.append(max_velocity)
        elif response == 'accn':
            plot_list.append(go.Scatter(x = time, y = acceleration_steady, mode = 'lines'))
            range_list.append(max_accn)

    max_range = max(range_list)

    return {
        'data': plot_list,
        'layout': go.Layout(
            xaxis={'title': 'Time (s)', 'range': [0, time_duration]},
            yaxis={'range': [-1.25 * max_range, 1.25 * max_range]},
            margin=go.Margin(l=100, r=100, t=50, b=50)
        )
    }

@app.callback(dash.dependencies.Output('forced-response-total', 'figure'),
                   [dash.dependencies.Input('time_duration', 'value'),
                    dash.dependencies.Input('x0', 'value'),
                    dash.dependencies.Input('v0', 'value'),
                    dash.dependencies.Input('wn', 'value'),
                    dash.dependencies.Input('zeta', 'value'),
                    dash.dependencies.Input('w', 'value'),
                    dash.dependencies.Input('f0', 'value'),
                    dash.dependencies.Input('disp_list', 'values')])

def plot_response_forced(time_duration, x0, v0, wn, zeta, w, f0, disp_list):

    time = np.arange(0, time_duration, 0.001)

    displacement_total = []
    velocity_total = []
    acceleration_total =[]

    if (float(zeta) == 0):
        #forced harmonic Undamped vibrations
        for t in time:
            displacement_total.append(disp_total_harmonic_undamped(float(wn), float(v0), float(x0), float(f0), float(w), t))
            velocity_total.append(vel_total_harmonic_undamped(float(wn), float(v0), float(x0), float(f0), float(w), t))
            acceleration_total.append(vel_total_harmonic_undamped(float(wn), float(v0), float(x0), float(f0), float(w), t))
    elif float(zeta)<1:
        #forced Harmonic Damped vibrations
        for t in time:
            displacement_total.append(disp_total_harmonic_damped(float(wn), float(v0), float(x0), float(f0), float(w),float(zeta), t))
            velocity_total.append(vel_total_harmonic_damped(float(wn), float(v0), float(x0), float(f0), float(w),float(zeta), t))
            acceleration_total.append(vel_total_harmonic_damped(float(wn), float(v0), float(x0), float(f0), float(w),float(zeta), t))

    max_velocity = np.amax(np.abs(velocity_total))
    max_disp = np.amax(np.abs(displacement_total))
    max_accn = np.amax(np.abs(acceleration_total))

    range_list = []
    plot_list = []

    for response in disp_list:
        if response == 'disp':
            plot_list.append(go.Scatter(x = time, y = displacement_total, mode = 'lines'))
            range_list.append(max_disp)
        elif response == 'vel':
            plot_list.append(go.Scatter(x = time, y = velocity_total, mode = 'lines'))
            range_list.append(max_velocity)
        elif response == 'accn':
            plot_list.append(go.Scatter(x = time, y = acceleration_total, mode = 'lines'))
            range_list.append(max_accn)

    max_range = max(range_list)

    return {
        'data': plot_list,
        'layout': go.Layout(
            xaxis={'title': 'Time (s)', 'range': [0, time_duration]},
            yaxis={'range': [-1.25 * max_range, 1.25 * max_range]},
            margin=go.Margin(l=100, r=100, t=50, b=50)
        )
    }

# app layout for analysing the displacement, velocity and acceleration factors

response_factor_layout = html.Div([
    html.H1('Response factor of SDOF System', style = {'margin':'auto', 'text-align': 'center', 'font-family': 'Impact'}),
    html.Br(),

    html.Div([
        html.Label('Zeta'),
        dcc.Input(id = 'zeta', value='0.1'),

        html.Br(),
        html.Div([
            html.Label('Check Responses'),
            dcc.Checklist(
                id = 'factor_disp_list',
                options=[
                    {'label': 'Displacement Factor', 'value': 'disp'},
                    {'label': 'Velocity Factor', 'value': 'vel'},
                    {'label': 'Acceleration Factor', 'value': 'accn'}
                    ],
                values=['disp', 'vel'],
                labelStyle={'display': 'inline-block'})
        ]),
    ], style = {'width': '60%', 'margin': 'auto'}),

    html.Br(),
    html.Div([
        dcc.Graph(id='response-factor-graph', animate=True, style={ 'border': '2px solid #1a53af'})
    ], style={'width': '60%', 'margin': 'auto'}),

    html.Br(),
    html.Div([
        dcc.Link('Forced Vibrations', href='/forced', style = {'text-decoration': 'none', 'font-size': '18px'}),
        html.Br(),
        dcc.Link('Free Vibrations', href = '/free', style = {'text-decoration': 'none', 'font-size': '18px'}),
        html.Br(),
        dcc.Link('Go back to home', href='/', style = {'text-decoration': 'none', 'font-size': '18px'}),
    ], style ={'columnCount': '3', 'text-align': 'center', 'width': '75%', 'margin': 'auto'}),


])

@app.callback(dash.dependencies.Output('response-factor-graph', 'figure'),
              [dash.dependencies.Input('zeta', 'value'),
               dash.dependencies.Input('factor_disp_list', 'values')
              ])

def plot_factor_response(zeta, factor_disp_list):

    ratio = np.arange(0,11, 0.001)

    disp_factor = []
    vel_factor = []
    accn_factor =[]

    for r in ratio:
        disp_factor.append(displacement_factor(r, float(zeta)))
        vel_factor.append(r*displacement_factor(r, float(zeta)))
        accn_factor.append(r**2*displacement_factor(r, float(zeta)))

    display_plot = []
    factor_max = []

    for factor in factor_disp_list:
        if factor == 'disp':
            display_plot.append(go.Scatter(x = ratio, y = disp_factor, mode = 'lines'))
            factor_max.append(np.amax(disp_factor))
        if factor == 'vel':
            display_plot.append(go.Scatter(x = ratio, y = vel_factor, mode = 'lines'))
            factor_max.append(np.amax(vel_factor))
        if factor == 'accn':
            display_plot.append(go.Scatter(x = ratio, y = accn_factor, mode = 'lines'))
            factor_max.append(np.amax(accn_factor))

    max_disp_factor = np.amax(factor_max)

    return {
        'data': display_plot,
        'layout': go.Layout(
            xaxis={'title': 'Driving Frequency / Natural Frequency', 'range': [0, 10]},
            yaxis={'range': [0, max_disp_factor],'type': 'linear'},
            margin=go.Margin(l=100, r=100, t=50, b=50)
        )
    }


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/free':
        return free_layout
    elif pathname == '/forced':
        return forced_layout
    elif pathname == '/response-factor':
        return response_factor_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    app.run_server(debug=True)
