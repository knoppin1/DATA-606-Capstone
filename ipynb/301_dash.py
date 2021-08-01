###########################################################################
# DATA 606 Capstone 
# Ken Noppinger - Phase 3, Python Program 1
# Wealth and Population Density Influences on COVID-19 Cases and Vaccinations Using Machine Learning
#
# This python program uses the Dash framework to provide a user with dynamic
# K-Means clustering capability on any two of the following county-level features:
# - Land Area
# - Population
# - Population Per Square Mile 
# - Median Income
# - Confirmed COVID-19 Cases 
# - Confirmed COVID-19 Cases Per Square Mile 
# - COVID-19 Deaths
# - COVID-19 Deaths Per Square Mile
# - COVID-19 Vaccinated Population
# - COVID-19 Vaccinated Percentage of Population
# - COVID-19 Vaccinations Per Square Mile
#
# The user works their way down the dashboard web page selecting the following:
# - Two features to cluster (as listed above) with linear or log scaling
# - Multi-select any number of states to include in the K-Means clustering
# - One feature to scale bubble size on the scatter plot for the data filtered by selected states
# - The optimum number of clusters (K) after reviewing an elbow curve chart
# 
# The resulting charts are:
# - Bar chart showing K-Means cluster distribution of counties
# - Scatter plot of K-Means clusters with centroid markings
# - Choropleth map showing counties by cluster color for each state selected 
#
# Note: The "all_data_fips.pkl" pickle file is loaded for use by this 
# program and represents the consolidated and comprehenisve data for use 
# in the study.
#
###########################################################################
# Import packages
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

###########################################################################
# Set style sheet
###########################################################################
#external_stylesheets = [dbc.themes.BOOTSTRAP]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

###########################################################################
# Load cleaned study data from pickle file
###########################################################################
print("Working directory = ", os. getcwd()) 
pickle_file = os. getcwd() + '\\all_data_fips.pkl'
print("Pickle file = ", pickle_file)
df = pd.read_pickle(pickle_file)
rows = df.shape[0]
print("Dataframe rows = ",rows)

###########################################################################
# Load a GEOJSON file containing the polygon definitions for counties by 
# FIPS code.
###########################################################################
json_file = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
with urlopen(json_file) as response:
	counties = json.load(response)
	
###########################################################################
# Set the list of features used to populate drop downs
###########################################################################
feature_list = ['Land_Area', 'Population', 'Pop_Sq_Mile', 'Income', 
				'Confirmed', 'Cases_Sq_Mile', 'Deaths', 'Deaths_Sq_Mile', 
				'Vaccinated', 'Vax_Pct','Vax_Sq_Mile']

###########################################################################
# Set cluster colors dictionary mapping used when charting
###########################################################################
colors_map = {'0':'blue','1':'red','2':'orange','3':'darkseagreen','4':'deeppink',
			  '5':'gray','6':'brown','7':'purple','8':'yellow','9':'lightblue'}

###########################################################################
# Define the layout of the dynamic dashboard controls and charts
###########################################################################
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
				value=['Maryland', 'Virginia','Pennsylvania','New Jersey'],
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
				value=5),
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

###########################################################################
# Callback function - update_all_counties_scatter_plot
# This function reacts to changes in feature selections for the x and y columns
# to use in scatter plots.  The scatter plot displayed includes those features 
# for all US Counties according to the scaling values for each respective axis.
###########################################################################
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

###########################################################################
# Callback function - update_select_states_scatter_plot
# This function reacts to changes in states, bubble size column and feature 
# selections and scaling for the x and y columns.  The scatter plot displayed 
# includes the data points for the features selected for the states selected
# according to the scaling values provided for each respective axis.  Each
# data point is sized based on the value of bubble feature selected.
###########################################################################
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

###########################################################################
# Callback function - update_elbow_graph
# This function reacts to changes in states and clustering feature selections.  
# The line chart displayed represents the elbow curve of the sum of squares error
# values when the K-Means algorithm is run for cluster values between 1 and 10.
###########################################################################
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
	#  - Fit the k-means algorithm to data 
	#  - Use kmean++ random initialization method 
	#  - Allow for a maximum of 300 iterations to find final clusters 
	#  - Run the algorithm for 10 different initial centroids 
	wcss =[]
	for i in range(1,11):
		kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10, random_state=0)
		kmeans.fit(features)

		# Compute the WCSS (within cluster sum of squares) and append it to a WCSS list.
		wcss.append(kmeans.inertia_)

	# Create the line chart
	fig = px.line(x=range(1,11), y=wcss, title='Elbow Curve')
	fig.update_xaxes(title='Number of Clusters')
	fig.update_yaxes(title='Inertia')
	fig.update_layout(margin={'l': 40, 'b': 20, 't': 50, 'r': 0}, hovermode='closest')
	return fig

###########################################################################
# Callback function - update_cluster_bar_chart
# This function reacts to changes in states, clustering feature selections,
# and the number of clusters.  The bar chart displayed shows the counts of
# counties in each cluster after running the K-Means algorithm for the
# specified number of clusters.
###########################################################################
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

	# Run the K-Means
	kmeans = KMeans(n_clusters=clusters_k, init ='k-means++', max_iter=300,  n_init=10, random_state=0)
	#sse = kmeans.inertia_
	#iterations = kmeans.n_iter_
	
	# Predict the clusters
	labels = kmeans.fit_predict(features)
	
	# Get the count of counties in each cluster.
	counts_dict = dict(Counter(labels))
	counts_df = pd.DataFrame(list(counts_dict.items()),columns = ['Cluster','Counties']).sort_values(by='Cluster')
	counts_df["Cluster"] = counts_df["Cluster"].astype(str)
	
	# Create the bar chart
	fig = px.bar(counts_df, x='Cluster', y='Counties', 
				 color='Cluster',
				 color_discrete_map=colors_map,
				 title='Cluster Sizes')
	fig.update_xaxes(tickmode='array',tickvals=list(range(0,clusters_k)))
	fig.update_xaxes(title='Cluster')
	fig.update_yaxes(title='Counties')
	fig.update_layout(margin={'l': 40, 'b': 40, 't': 50, 'r': 0}, hovermode='closest')
	return fig

###########################################################################
# Callback function - update_cluster_scatter_plot
# This function reacts to changes in states, clustering feature selections,
# and the number of clusters.  The scatter plot chart created shows cluster
# colored data points with black star centroids for the counties of selected 
# states after running the K-Means algorithm for the selected features and
# specified number of clusters.
###########################################################################
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

	# Run the K-Means
	kmeans = KMeans(n_clusters=clusters_k, init ='k-means++', max_iter=300,  n_init=10, random_state=0)
	
	# Predict the clusters
	labels = kmeans.fit_predict(features)
	dff['Cluster'] = labels
	dff['Cluster']=dff['Cluster'].astype(str)
	
	# Get the centroids
	centroids = scaler.inverse_transform(kmeans.cluster_centers_)
	
	# Create the scatter plot
	fig = px.scatter(dff, x=xaxis_column_name, y=yaxis_column_name, 
					 color='Cluster', color_discrete_map=colors_map, 
					 category_orders={'Cluster':['0','1','2','3','4','5','6','7','8','9']}, 
					 hover_name='Place', hover_data=col_names, title="K-Means Clustering")
	fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
	fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')
	fig.update_layout(margin={'l': 40, 'b': 40, 't': 50, 'r': 0}, hovermode='closest')
	
	# Add the centroids to the scatter plot
	fig.add_trace(pgo.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
							  marker=pgo.Marker(symbol='star-dot', size=12, color='black'),
							  showlegend=False))	
	return fig

###########################################################################
# Callback function - update_choropleth
# This function reacts to changes in states, clustering feature selections,
# and the number of clusters.  The choropleth map created shows cluster
# colored counties for the selected states after running the K-Means algorithm 
# for the selected features and specified number of clusters.
###########################################################################
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

	# Run the K-Means
	kmeans = KMeans(n_clusters=clusters_k, init ='k-means++', max_iter=300,  n_init=10, random_state=0)
	
	# Predict the clusters
	labels = kmeans.fit_predict(features)
	dff['Cluster'] = labels
	dff['Cluster']=dff['Cluster'].astype(str)
	
	# Set cluster order for the legend, and hover columns	
	cluster_order = {'Cluster':['0','1','2','3','4','5','6','7','8','9']}
	hover_columns = col_names

	# Create the choropleth map.
	map_title = "Cluster Map for " + xaxis_column_name + " and " + yaxis_column_name
	fig = px.choropleth(dff, geojson=counties, locations='FIPS_Code', color='Cluster', 
						color_discrete_map=colors_map, category_orders=cluster_order,
						scope="usa", hover_name='Place', hover_data=hover_columns, title=map_title
						)
	fig.update_geos(fitbounds="locations", visible=False)
	fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, title_x=.5)	
	return fig

###########################################################################
# Run Main
###########################################################################
if __name__ == '__main__':
	app.run_server(debug=True)