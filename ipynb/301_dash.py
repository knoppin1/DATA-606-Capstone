import pandas as pd
import pickle
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as pgo

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter


from urllib.request import urlopen
import json

# -----------------

#external_stylesheets = [dbc.themes.BOOTSTRAP]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

print("Working directory = ", os. getcwd()) 
pickle_file = os. getcwd() + '\\all_data_fips.pkl'
print("Pickle file = ", pickle_file)
df = pd.read_pickle(pickle_file)
rows = df.shape[0]
print("Dataframe rows = ",rows)

#available_indicators = df['Indicator Name'].unique()
feature_list = ['Land_Area', 'Population', 'Pop_Sq_Mile', 'Income', 
				'Confirmed', 'Cases_Sq_Mile', 'Deaths', 'Deaths_Sq_Mile', 
				'Vaccinated', 'Vax_Pct','Vax_Sq_Mile']

app.layout = html.Div([
	html.Div([
		html.H2(children='Capstone: Wealth and Population Influences on COVID-19 Cases and Vaccinations'),
		
		html.H4(children='Select features to cluster:'),
		html.Div([
			dcc.Dropdown(
				id='xaxis-column',
				options=[{'label': i, 'value': i} for i in feature_list],
				value='Income'
			),
			dcc.RadioItems(
				id='xaxis-type',
				options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
				value='Linear',
				labelStyle={'display': 'inline-block'}
			)
		], style={'width': '48%', 'display': 'inline-block'}),
		html.Div([
			dcc.Dropdown(
				id='yaxis-column',
				options=[{'label': i, 'value': i} for i in feature_list],
				value='Vax_Pct'
			),
			dcc.RadioItems(
				id='yaxis-type',
				options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
				value='Linear',
				labelStyle={'display': 'inline-block'}
			)
		], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
	]),

	html.Div([html.Br()]),
	
	dcc.Graph(id='all-counties-scatter-plot'),
	
	html.Div([html.Br()]),
	html.H4(children='Select states to limit the counties clustered:'),
	html.Div([
		html.Div([
			html.Label('States'),
			dcc.Dropdown(
				id='states',
				options=[{'label': i, 'value': i} for i in df['State'].unique()],
				value=['California', 'Florida'],
				multi=True)
		], style={'width': '48%', 'display': 'inline-block'}),
		html.Div([
			html.Label('Bubble Size Feature'),
			dcc.Dropdown(
				id='bubble-column',
				options=[{'label': i, 'value': i} for i in feature_list],
				value='Cases_Sq_Mile')
		], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
	]),
	dcc.Graph(id='selected-states-scatter-plot'),
	html.Div([html.Br()]),
	html.H4(children='Select optimum number of clusters:'),
	html.Div([
		html.Div([
			html.Div([html.Br()]),
			html.Div([html.Br()]),
			dcc.Graph(id='elbow-chart')
		], className="six columns"),
		html.Div([
			html.Label('Optimum Clusters (K)'),
			dcc.Dropdown(
				id='clusters_k',
				options=[
					{'label': '1', 'value': 1}, {'label': '2', 'value': 2}, {'label': '3', 'value': 3}, 
					{'label': '4', 'value': 4}, {'label': '5', 'value': 5}, {'label': '6', 'value': 6}, 
					{'label': '7', 'value': 7}, {'label': '8', 'value': 8},	{'label': '9', 'value': 9}, 
					{'label': '10', 'value': 10}
					],
				value=3),
			dcc.Graph(id='cluster-bar-chart')
		], className="six columns")
	], className="row"),
	html.Div([html.Br()]),	
	html.H4(children='Review Plot of Clusters and Centroids:'),
	dcc.Graph(id='cluster-scatter-plot'),
	html.Div([html.Br()]),	
	html.H4(children='Review Choropleth Map of Clusters:'),
	dcc.Graph(id='cluster-chloropleth'),
])

@app.callback(
	Output('all-counties-scatter-plot', 'figure'),
	Input('xaxis-column', 'value'),
	Input('yaxis-column', 'value'),
	Input('xaxis-type', 'value'),
	Input('yaxis-type', 'value'))
def update_all_counties_scatter_plot(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type):

	fig = px.scatter(x=df[xaxis_column_name], y=df[yaxis_column_name], 
					 hover_name=df["Place"], title='All US Counties')
	fig.update_layout(margin={'l': 40, 'b': 40, 't': 50, 'r': 0}, hovermode='closest')
	fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
	fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

	return fig

@app.callback(
	Output('selected-states-scatter-plot', 'figure'),
	Input('xaxis-column', 'value'),
	Input('yaxis-column', 'value'),
	Input('xaxis-type', 'value'),
	Input('yaxis-type', 'value'),
	Input('states', 'value'),
	Input('bubble-column', 'value'))
def update_select_states_scatter_plot(xaxis_column_name, yaxis_column_name, 
									  xaxis_type, yaxis_type, states, bubble_column):
    
	dff = df[df['State'].isin(states)]
	fig = px.scatter(dff, x=xaxis_column_name, y=yaxis_column_name, 
					 size=bubble_column, color="State", hover_name="Place")
	fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
	fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
	fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

	return fig

@app.callback(
	Output('elbow-chart', 'figure'),
	Input('xaxis-column', 'value'),
	Input('yaxis-column', 'value'),
	Input('states', 'value'))
def update_elbow_graph(xaxis_column_name, yaxis_column_name, states):

	# Filter data down to states selected.
	dff = df[df['State'].isin(states)]

	# Scale selected features.
	col_names = [xaxis_column_name, yaxis_column_name]
	features = dff[col_names]
	scaler = StandardScaler().fit(features.values)
	features = scaler.transform(features.values)

	# Loop ten times to:
	#  - fit the k-means algorithm to data 
	#  - use kmean++ random initialization method 
	#  - allow for a maximum of 300 iterations to find final clusters 
	#  - run the algorithm for 10 different initial centroids 
	wcss =[]
	for i in range(1,11):
		kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10, random_state=0)
		kmeans.fit(features)

		# Compute the WCSS (within cluster sum of squares) and append it to a WCSS list.
		wcss.append(kmeans.inertia_)

	fig = px.line(x=range(1,11), y=wcss, title='Elbow Curve')
	fig.update_xaxes(title='Number of Clusters')
	fig.update_yaxes(title='Inertia')
	fig.update_layout(margin={'l': 40, 'b': 20, 't': 50, 'r': 0}, hovermode='closest')
	return fig

@app.callback(
	Output('cluster-bar-chart', 'figure'),
	Input('xaxis-column', 'value'),
	Input('yaxis-column', 'value'),
	Input('states', 'value'),
	Input('clusters_k', 'value'))
def update_cluster_bar_chart(xaxis_column_name, yaxis_column_name, states, clusters_k):

	# Filter data down to states selected.
	dff = df[df['State'].isin(states)]

	# Scale selected features.
	col_names = [xaxis_column_name, yaxis_column_name]
	features = dff[col_names]
	scaler = StandardScaler().fit(features.values)
	features = scaler.transform(features.values)

	kmeans = KMeans(n_clusters=clusters_k, init ='k-means++', max_iter=300,  n_init=10, random_state=0)
	#sse = kmeans.inertia_
	#iterations = kmeans.n_iter_
	labels = kmeans.fit_predict(features)
	counts_dict = dict(Counter(labels))
	counts_df = pd.DataFrame(list(counts_dict.items()),columns = ['Cluster','Counties']).sort_values(by='Cluster')

	colors = {'0':'blue','1':'red','2':'orange','3':'darkseagreen','4':'deeppink',
			  '5':'gray','6':'brown','7':'purple','8':'yellow','9':'lightblue'}
	counts_df["Cluster"] = counts_df["Cluster"].astype(str)
	fig = px.bar(counts_df, x='Cluster', y='Counties', 
				 color='Cluster',
				 color_discrete_map=colors,
				 title='Cluster Sizes')
	fig.update_xaxes(tickmode='array',tickvals=list(range(0,clusters_k)))
	fig.update_xaxes(title='Cluster')
	fig.update_yaxes(title='Counties')
	fig.update_layout(margin={'l': 40, 'b': 40, 't': 50, 'r': 0}, hovermode='closest')
	return fig

@app.callback(
	Output('cluster-scatter-plot', 'figure'),
	Input('xaxis-column', 'value'),
	Input('yaxis-column', 'value'),
	Input('xaxis-type', 'value'),
	Input('yaxis-type', 'value'),
	Input('states', 'value'),
	Input('clusters_k', 'value'))
def update_cluster_scatter_plot(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type, states, clusters_k):
	
	# Filter data down to states selected.
	dff = df[df['State'].isin(states)].copy()

	# Scale selected features.
	col_names = [xaxis_column_name, yaxis_column_name]
	features = dff[col_names]
	scaler = StandardScaler().fit(features.values)
	features = scaler.transform(features.values)

	kmeans = KMeans(n_clusters=clusters_k, init ='k-means++', max_iter=300,  n_init=10, random_state=0)
	labels = kmeans.fit_predict(features)
	dff['Cluster'] = labels
	dff['Cluster']=dff['Cluster'].astype(str)
	centroids = scaler.inverse_transform(kmeans.cluster_centers_)
	
	color_dict = {'0':'blue','1':'red','2':'orange','3':'darkseagreen','4':'deeppink',
				  '5':'gray','6':'brown','7':'purple','8':'yellow','9':'lightblue'}
	fig = px.scatter(dff, x=xaxis_column_name, y=yaxis_column_name, color='Cluster', color_discrete_map=color_dict, 
					 category_orders={'Cluster':['0','1','2','3','4','5','6','7','8','9']}, 
					 hover_name='Place', hover_data=col_names, title="K-Means Clustering")
	fig.add_trace(pgo.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
							  marker=pgo.Marker(symbol='star-dot', size=12, color='black'),
							  showlegend=False))
	fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
	fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')
	fig.update_layout(margin={'l': 40, 'b': 40, 't': 50, 'r': 0}, hovermode='closest')
	
	return fig

@app.callback(
	Output('cluster-chloropleth', 'figure'),
	Input('xaxis-column', 'value'),
	Input('yaxis-column', 'value'),
	Input('xaxis-type', 'value'),
	Input('yaxis-type', 'value'),
	Input('states', 'value'),
	Input('clusters_k', 'value'))
def update_choropleth(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type, states, clusters_k):

	# Filter data down to states selected.
	dff = df[df['State'].isin(states)].copy()

	# Scale selected features.
	col_names = [xaxis_column_name, yaxis_column_name]
	features = dff[col_names]
	scaler = StandardScaler().fit(features.values)
	features = scaler.transform(features.values)

	kmeans = KMeans(n_clusters=clusters_k, init ='k-means++', max_iter=300,  n_init=10, random_state=0)
	labels = kmeans.fit_predict(features)
	dff['Cluster'] = labels
	dff['Cluster']=dff['Cluster'].astype(str)
	
	
	color_map = {'0':'blue','1':'red','2':'orange','3':'darkseagreen','4':'deeppink',
				 '5':'gray','6':'brown','7':'purple','8':'yellow','9':'lightblue'}
	cluster_order = {'Cluster':['0','1','2','3','4','5','6','7','8','9']}
	hover_columns = col_names

	# Load a GEOJSON file containing the polygon definitions for counties by FIPS code.
	with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
		counties = json.load(response)
	
	map_title = "Cluster Map for " + xaxis_column_name + " and " + yaxis_column_name
	fig = px.choropleth(dff, geojson=counties, locations='FIPS_Code', color='Cluster', 
						color_discrete_map=color_map, category_orders=cluster_order,
						scope="usa", hover_name='Place', hover_data=hover_columns, title=map_title
						)
	fig.update_geos(fitbounds="locations", visible=False)
	fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, title_x=.5)	
	return fig


if __name__ == '__main__':
	app.run_server(debug=True)