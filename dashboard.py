import numpy as np
import pandas as pd
import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

#################################################

################
# Haile's work #
################

# load data from csv file
df_data = pd.read_csv('data/crop_data.csv')

X = np.array(pd.concat([pd.DataFrame({'DUMMY': list(np.ones(48))}), df_data[['HUM_PCT', 'TEMP_C', 'SMOIST_G_CM3', 'ET_MM']]], axis = 1))

Y = np.array(df_data[['CWD_M3_HA']])

# calculate the parameters of the linear regression model
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# compute the R_squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.T.dot(d1) / d2.T.dot(d2)
print(f"The r-quared value of the linear regression model is: {r2[0][0]}")

# plot the observations and prediction
# trace for observations
trace1 = go.Scatter(
    x = np.array(range(48)),
    y = Y.T[0],
    mode = 'markers',
    marker = dict(
        size = 12,
        color = 'rgb(255, 0, 0)',
        symbol = 'circle',
        line = dict(
            width = 2
        )
    ),
    name = 'observations'
)

# trace for predictions
trace2 = go.Scatter(
    x = np.array(range(48)),
    y = Yhat.T[0],
    mode = 'lines',
    marker = dict(
        color = 'rgb(0, 0, 255)'
    ),
    name = 'predictions'
)

data1 = [trace1, trace2]

layout1 = go.Layout(
    title = 'Crop Water Demand (CWD) variation',
    font = dict(
        size = 15
    ),
    xaxis = dict(
        title = 'Time (month #)',
        rangeslider = dict(
            visible = True
            )
        ),
    yaxis = dict(
        title = 'CWD (m3/ha)'
        ),
    hovermode = 'closest'
)

#################################################

################
# Sarah's work #
################

data2 = dict(
    type='sankey',
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(
        color = "black",
        width = 0.5
      ),
      label = ["A1", "A2", "B1", "B2", "C1", "C2"],
      color = ["blue", "blue", "blue", "blue", "blue", "blue"]
    ),
    link = dict(
      source = [0,1,0,2,3,3],
      target = [2,3,3,4,4,5],
      value = [8,4,2,8,4,2]
  ))

layout2 =  dict(
    title = "Resource Allocations",
    font = dict(
      size = 15
    )
)

#################################################

###################
# Geospatial Viz  #
###################

mapbox_access_token = 'pk.eyJ1IjoicmdvdmluZGFuIiwiYSI6ImNqcDlodXJsbjJhd2EzcW8xeXdwYm9jem4ifQ.VpJ1X26USunx-0UOUPz9zA'

data3 = go.Scattermapbox(
    lat = ['24.998425'],
    lon = ['51.189871'],
    mode = 'markers',
    marker = dict(
        size = 14
    ),
    text=['Alfalfa Fields']
)

layout3 = go.Layout(
    title = "Alfalfa Field Location",
    font = dict(
      size = 15
    ),
    autosize = True,
    hovermode = 'closest',
    mapbox = dict(
        accesstoken = mapbox_access_token,
        bearing = 0,
        center = dict(
            lat = 24.998425,
            lon = 51.189871
        ),
        pitch=0,
        zoom=5
    )
)

#################################################

#################
# RSI Dashboard #
#################

app = dash.Dash()

image_file = 'data/RSI.png'
encoded_image = base64.b64encode(open(image_file, 'rb').read())

app.layout = html.Div(children = [
    html.Img(
        src = 'data:image/png;base64,{}'.format(encoded_image.decode()),
        style = dict(
            width = 250,
            display = 'block',
            marginLeft = 'auto',
            marginRight = 'auto',
            padding = 10
        )),
    html.H1(children = [
        'Test dashboard for Agriculture'
        ],
        style = dict(
            textAlign = 'center'
        )
    ),
    html.Div(id = 'div-haile', children = [
        dcc.Graph(
            id = 'plot-haile',
            figure = dict(
                data = data1,
                layout = layout1
            )
        )], style = dict(
            float = 'left',
            padding = 0,
            margin = 0
        )),
    html.Div(id = 'div-geo', children = [
        dcc.Graph(
            id = 'plot-geo',
            figure = dict(
                data = [data3],
                layout = layout3
            )
        )], style = dict(
            float = 'right',
            padding = 0,
            margin = 0
        )),
    html.Div(id = 'div-sarah', children = [
        dcc.Graph(
            id = 'plot-sarah',
            figure = dict(
                data = [data2],
                layout = layout2
                )
        )], style = dict(
            clear = 'both',
            padding = 0,
            margin = 0
        ))
    ])

if __name__ == '__main__':
    app.run_server(debug = True)

#################################################