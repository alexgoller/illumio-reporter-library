#!/usr/bin/env python3

import os
import json
import click
from functools import wraps
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from illumio import PolicyComputeEngine, TrafficQuery
from collections import deque, defaultdict
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from kaleido.scopes.plotly import PlotlyScope
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend
import pygraphviz as pgv
import networkx as nx
import io
import logging
import sys
import subprocess
import math
from scipy.optimize import minimize
import tempfile
import textwrap


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def global_options(f):
	@click.option('--pce-host', envvar="ILLUMIO_PCE_HOST", required=True, help='PCE host')
	@click.option('--port', envvar="ILLUMIO_PCE_PORT", required=True, type=int, help='PCE port')
	@click.option('--org-id', envvar="ILLUMIO_PCE_ORG_ID", required=True, help='Organization ID')
	@click.option('--api-key', envvar="ILLUMIO_PCE_API_KEY", required=True, help='API key')
	@click.option('--api-secret', envvar="ILLUMIO_PCE_API_SECRET", required=True, help='API secret')
	@click.option('--start', default='30 days ago', help='Start date (YYYY-MM-DD or "X days ago")')
	@click.option('--end', default='today', help='End date (YYYY-MM-DD or "X days ago")')
	@click.option('--limit', type=int, default=2000, help='Maximum number of traffic flows to fetch')
	@wraps(f)
	def wrapper(*args, **kwargs):
		return f(*args, **kwargs)
	return wrapper

label_href_map = {}
value_href_map = {}

def parse_date(date_string):
	if date_string.lower() == 'today':
		return datetime.now()
	if date_string.lower().endswith(' ago'):
		days = int(date_string.split()[0])
		return datetime.now() - timedelta(days=days)
	return datetime.strptime(date_string, "%Y-%m-%d")

def traffic_flow_unique_name(flow):
	return "{}-{}_{}-{}_{}".format(
		flow.src.ip,
		flow.dst.ip,
		flow.service.port,
		flow.service.proto,
		flow.flow_direction
	)

def to_dataframe(flows, save_temp=False):
	global label_href_map
	global value_href_map

	series_array = []
	
	for flow in flows:
		f = {
			'src_ip': flow.src.ip,
			'src_hostname': flow.src.workload.name if flow.src.workload is not None else None,
			'dst_ip': flow.dst.ip,
			'dst_hostname': flow.dst.workload.name if flow.dst.workload is not None else None,
			'proto': flow.service.proto,
			'port': flow.service.port,
			'process_name': flow.service.process_name,
			'service_name': flow.service.service_name,
			'user_name': flow.service.user_name,
			'windows_service_name': flow.service.windows_service_name,
			'policy_decision': flow.policy_decision,
			'flow_direction': flow.flow_direction,
			'num_connections': flow.num_connections,
			'first_detected': flow.timestamp_range.first_detected,
			'last_detected': flow.timestamp_range.last_detected,
			'src_iplist': '',
			'dst_iplist': ''
		}
		
	   # Check if src is an IP list
		if flow.src.ip_lists:
			# logging.debug(f'SRC IP Lists: {flow.dst.ip_lists}')
			f['src_iplist'] = flow.src.ip_lists[0].name if flow.src.ip_lists else None

		# Check if dst is an IP list
		if flow.dst.ip_lists:
			# logging.debug(f'DST IP Lists: {flow.dst.ip_lists}')
			f['dst_iplist'] = flow.dst.ip_lists[0].name if flow.dst.ip_lists else None

		if flow.src.workload:
			# print(f'Found workload: {flow.src.workload.name}, {flow.src.workload.labels}')
			for l in flow.src.workload.labels:
				if l.href in label_href_map:
					f['src_' + label_href_map[l.href]['key']] = label_href_map[l.href]['value']

		if flow.dst.workload:
			# print(f'Found workload: {flow.dst.workload.name}, {flow.dst.workload.labels}')
			for l in flow.dst.workload.labels:
				if l.href in label_href_map:
					f['dst_' + label_href_map[l.href]['key']] = label_href_map[l.href]['value']
		
		series_array.append(f)

	df = pd.DataFrame(series_array)
	print(type(df))

	if save_temp:
		save_dataframe_to_temp(df)

	return df

def save_dataframe_to_temp(df):
	with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
		logging.debug(f'Saving DataFrame to temporary file: {tmp.name}')
		df.to_parquet(tmp.name, index=False)
		return tmp.name

def load_dataframe_from_temp(file_path):
	if os.path.exists(file_path):
		return pd.read_parquet(file_path)
	else:
		raise FileNotFoundError(f"Temporary file not found: {file_path}")

def generate_top_x(df, column, n=10, title=""):
	top_x = df[column].value_counts().nlargest(n)
	fig = go.Figure(data=[go.Bar(x=top_x.index, y=top_x.values)])
	fig.update_layout(title=title, xaxis_title=column, yaxis_title="Count", width=4096, height=2160)
	return fig

def generate_treemap(df, title=""):
	protocol_port = df.groupby(['proto', 'port']).size().reset_index(name='count')
	fig = px.treemap(protocol_port, path=['proto', 'port'], values='count')
	fig.update_layout(title=title, width=4096, height=2160)
	return fig

def generate_top_talkers(df, n=10):
	return generate_top_x(df, 'src_ip', n, f"Top {n} Talkers")

def generate_top_destinations(df, n=10):
	return generate_top_x(df, 'dst_ip', n, f"Top {n} Destinations")

def generate_top_ports(df, n=10):
	return generate_top_x(df, 'port', n, f"Top {n} Ports")

def generate_ip_protocol_treemap(df):
	return generate_treemap(df, "IP Protocols and Most Used Ports")

def generate_top_app_group_sources(df, n=10):
	df['src_app_group'] = df['src_app'] + ' (' + df['src_env'] + ')'
	return generate_top_x(df, 'src_app_group', n, f"Top {n} App Group Sources")

def generate_top_app_group_destinations(df, n=10):
	df['dst_app_group'] = df['dst_app'] + ' (' + df['dst_env'] + ')'
	return generate_top_x(df, 'dst_app_group', n, f"Top {n} App Group Destinations")

def generate_traffic_graph(df, diagram_type, output_format, direction):
	connections = defaultdict(lambda: defaultdict(int))

	for _, row in df.iterrows():
		src_app = row.get('src_app', 'No App')
		src_env = row.get('src_env', 'No Env')
		dst_app = row.get('dst_app', 'No App')
		dst_env = row.get('dst_env', 'No Env')

		src = f"{src_app} ({src_env})"
		dst = f"{dst_app} ({dst_env})"
		if src != dst:
			connections[src][dst] += 1
	
	if diagram_type == 'sankey':
		return generate_sankey_diagram(connections, output_format)
	elif diagram_type == 'sunburst':
		return generate_sunburst_diagram(connections, output_format)
	elif diagram_type == 'graphviz':
		return generate_graphviz_diagram(connections, output_format, direction)
	else:
		raise ValueError(f"Unsupported diagram type: {diagram_type}")

def generate_sankey_diagram(connections, output_format):
	sources, targets, values = [], [], []
	labels = set()
	
	for source, destinations in connections.items():
		for target, value in destinations.items():
			sources.append(source)
			targets.append(target)
			values.append(value)
			labels.add(source)
			labels.add(target)
	
	labels = list(labels)
	label_to_index = {label: i for i, label in enumerate(labels)}
	
	fig = go.Figure(data=[go.Sankey(
		node = dict(
			pad = 15,
			thickness = 20,
			line = dict(color = "black", width = 0.5),
			label = labels,
			color = "blue"
		),
		link = dict(
			source = [label_to_index[s] for s in sources],
			target = [label_to_index[t] for t in targets],
			value = values
		)
	)])
	
	fig.update_layout(title_text="Application Flow Sankey Diagram", font_size=10, width=4096, height=2160)
	
	return export_plotly(fig, output_format)

def generate_sunburst_diagram(connections, output_format):
	data = []
	for source, destinations in connections.items():
		for target, value in destinations.items():
			data.append({
				'source': source,
				'target': target,
				'value': value
			})
	
	df = pd.DataFrame(data)
	
	fig = px.sunburst(
		df,
		path=['source', 'target'],
		values='value',
		title="Application Flow Sunburst Diagram"
	)
	
	return export_plotly(fig, output_format)

def export_plotly(fig, output_format):
	if output_format == 'html':
		return fig.to_html(include_plotlyjs=True, full_html=True)
	else:
		scope = PlotlyScope()
		img_bytes = scope.transform(fig, format=output_format)
		return img_bytes

def generate_app_env_treemap(df, column_prefix, title):
	required_columns = [f'{column_prefix}_app', f'{column_prefix}_env']
	
	# Check if required columns exist
	missing_columns = [col for col in required_columns if col not in df.columns]
	if missing_columns:
		click.echo(f"Error: The following required columns are missing: {', '.join(missing_columns)}")
		click.echo(f"Available columns: {', '.join(df.columns)}")
		return None

	df[f'{column_prefix}_app_env'] = df[f'{column_prefix}_app'] + ' | ' + df[f'{column_prefix}_env']
	app_env_counts = df.groupby([f'{column_prefix}_env', f'{column_prefix}_app', f'{column_prefix}_app_env']).size().reset_index(name='count')
	fig = px.treemap(app_env_counts, 
					 path=[f'{column_prefix}_env', f'{column_prefix}_app', f'{column_prefix}_app_env'], 
					 values='count',
					 title=title)
	fig.update_traces(textinfo="label+value+percent parent")
	return fig
def generate_plotly_directed_graph(connections, width=800, height=600):
	# Create a NetworkX graph
	G = nx.DiGraph()

	# Add edges to the graph
	for source, targets in connections.items():
		for target, weight in targets.items():
			G.add_edge(source, target, weight=weight)

	# Use Fruchterman-Reingold force-directed algorithm
	pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)

	# Create edge trace
	edge_trace = go.Scatter(
		x=[],
		y=[],
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		mode='lines')

	for edge in G.edges():
		x0, y0 = pos[edge[0]]
		x1, y1 = pos[edge[1]]
		edge_trace['x'] += (x0, x1, None)
		edge_trace['y'] += (y0, y1, None)

	# Create node trace
	node_trace = go.Scatter(
		x=[],
		y=[],
		text=[],
		mode='markers',
		hoverinfo='text',
		marker=dict(
			showscale=True,
			colorscale='YlGnBu',
			reversescale=True,
			color=[],
			size=10,
			colorbar=dict(
				thickness=15,
				title='Node Connections',
				xanchor='left',
				titleside='right'
			),
			line_width=2))

	# Add node positions
	for node in G.nodes():
		x, y = pos[node]
		node_trace['x'] += (x,)
		node_trace['y'] += (y,)

	# Color node points by the number of connections
	for node, adjacencies in G.adjacency():
		node_trace['marker']['color'] += (len(adjacencies),)
		node_info = f'{node}<br># of connections: {len(adjacencies)}'
		node_trace['text'] += (node_info,)

	# Create annotations for node labels
	annotations = []
	for node, (x, y) in pos.items():
		annotations.append(dict(
			x=x, y=y,
			xref='x', yref='y',
			text=node,
			showarrow=False,
			font=dict(size=8),
			bgcolor='rgba(255, 255, 255, 0.7)'
		))

	# Create the figure
	fig = go.Figure(data=[edge_trace, node_trace],
					layout=go.Layout(
						title='Network Graph',
						titlefont_size=16,
						showlegend=False,
						hovermode='closest',
						margin=dict(b=20,l=5,r=5,t=40),
						annotations=annotations,
						xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
						yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
						width=width,
						height=height
					))

	# Update layout to prevent clipping of node labels
	fig.update_layout(clickmode='event+select')
	fig.update_xaxes(range=[min(node_trace['x'])-0.1, max(node_trace['x'])+0.1])
	fig.update_yaxes(range=[min(node_trace['y'])-0.1, max(node_trace['y'])+0.1])

	return fig

def get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit):
	global label_href_map
	global value_href_map

	pce = PolicyComputeEngine(pce_host, port=port, org_id=org_id)
	pce.set_credentials(api_key, api_secret)

	if not pce.check_connection():
		click.echo("Connection to PCE failed.")
		return None

	# todo: add label caching and lookup if there is a miss
	for l in pce.labels.get(params={'max_results': 10000}):
		label_href_map[l.href] = {"key": l.key, "value": l.value}
		value_href_map["{}={}".format(l.key, l.value)] = l.href

	d_end = parse_date(end) if end != 'today' else datetime.now()
	d_start = parse_date(start)

	traffic_query = TrafficQuery.build(
		start_date=d_start.strftime("%Y-%m-%d"),
		end_date=d_end.strftime("%Y-%m-%d"),
		include_services=[],
		exclude_services=[
			{"port": 53},
			{"port": 137},
			{"port": 138},
			{"port": 139},
			{"proto": "udp"}
		],
		exclude_destinations=[
			{"transmission": "broadcast"},
			{"transmission": "multicast"}
		],
		policy_decisions=['allowed', 'potentially_blocked', 'unknown'],
		max_results=limit
	)

	all_traffic = pce.get_traffic_flows_async(
		query_name='all-traffic',
		traffic_query=traffic_query
	)

	print(f'get_traffic_data: {type(all_traffic)}')
	print(f'Length of all_traffic: {len(all_traffic)}')

	df = to_dataframe(all_traffic, save_temp=True)
	print(f'get_traffic_data: {type(df)}')
	return(df)

@click.group()
def cli():
	"""Illumio CLI tool for traffic analysis and visualization."""
	pass

@cli.command()
@global_options
@click.option('--output', default='traffic_analysis', help='Output filename prefix')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
@click.option('--top-n', default=10, help='Number of top items to show')
def analyze(pce_host, port, org_id, api_key, api_secret, start, end, output, limit, format, top_n):
	"""Analyze traffic data and generate Top X views and treemap."""
	global label_href_map
	global value_href_map

	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	views = {
		'top_talkers': generate_top_talkers(df, top_n),
		'top_destinations': generate_top_destinations(df, top_n),
		'top_ports': generate_top_ports(df, top_n),
		'ip_protocol_treemap': generate_ip_protocol_treemap(df),
		'top_app_group_sources': generate_top_app_group_sources(df, top_n),
		'top_app_group_destinations': generate_top_app_group_destinations(df, top_n)
	}
	
	# Save all views
	for name, fig in views.items():
		filename = f"{output}_{name}.{format}"
		if format == 'html':
			fig.write_html(filename)
		else:
			fig.write_image(filename)
		click.echo(f"Saved {name} as {filename}")

def save_figure(fig, output, format):
	filename = f"{output}.{format}"
	if format == 'html':
		fig.write_html(filename)
	else:
		fig.write_image(filename)
	click.echo(f"Saved graph as {filename}")


@cli.command()
@global_options
@click.option('--output', default='top_talkers', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
@click.option('--top-n', default=10, help='Number of top items to show')
def top_talkers(pce_host, port, org_id, api_key, api_secret, start, end, output, limit, format, top_n):
	"""Generate a graph of top talkers."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		fig = generate_top_x(df, 'src_ip', top_n, f"Top {top_n} Talkers")
		save_figure(fig, output, format)

@cli.command()
@global_options
@click.option('--output', default='top_destinations', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
@click.option('--top-n', default=10, help='Number of top items to show')
def top_destinations(pce_host, port, org_id, api_key, api_secret, start, end, output, format, top_n):
	"""Generate a graph of top destinations."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		fig = generate_top_x(df, 'dst_ip', top_n, f"Top {top_n} Destinations")
		save_figure(fig, output, format)

@cli.command()
@global_options
@click.option('--output', default='top_ports', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
@click.option('--top-n', default=10, help='Number of top items to show')
def top_ports(pce_host, port, org_id, api_key, api_secret, start, end, output, limit, format, top_n):
	"""Generate a graph of top ports used in the environment."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		fig = generate_top_x(df, 'port', top_n, f"Top {top_n} Ports")
		save_figure(fig, output, format)

@cli.command()
@global_options
@click.option('--output', default='ip_protocol_treemap', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
def ip_protocol_treemap(pce_host, port, org_id, api_key, api_secret, start, end, limit, output, format):
	"""Generate a treemap for IP protocols containing the most used ports."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		fig = generate_treemap(df, "IP Protocols and Most Used Ports")
		save_figure(fig, output, format)

@cli.command()
@global_options
@click.option('--output', default='top_app_group_sources', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
@click.option('--top-n', default=10, help='Number of top items to show')
def top_app_group_sources(pce_host, port, org_id, api_key, api_secret, start, end, limit, output, format, top_n):
	"""Generate a graph of top app group sources."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		df['src_app_group'] = df['src_app'] + ' (' + df['src_env'] + ')'
		fig = generate_top_x(df, 'src_app_group', top_n, f"Top {top_n} App Group Sources")
		save_figure(fig, output, format)

@cli.command()
@global_options
@click.option('--output', default='top_app_group_destinations', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
@click.option('--top-n', default=10, help='Number of top items to show')
def top_app_group_destinations(pce_host, port, org_id, api_key, api_secret, start, end, limit, output, format, top_n):
	"""Generate a graph of top app group destinations."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		df['dst_app_group'] = df['dst_app'] + ' (' + df['dst_env'] + ')'
		fig = generate_top_x(df, 'dst_app_group', top_n, f"Top {top_n} App Group Destinations")

@cli.command()
@global_options
@click.option('--output', default='top_talking_app_env_treemap', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
def top_talking_app_env_treemap(pce_host, port, org_id, api_key, api_secret, start, end, limit, output, format):
	"""Generate a treemap of the app/env tuples talking the most."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		fig = generate_app_env_treemap(df, 'src', "Top Talking App/Env Tuples")
		save_figure(fig, output, format)

@cli.command()
@global_options
@click.option('--output', default='top_receiving_app_env_treemap', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
def top_receiving_app_env_treemap(pce_host, port, org_id, api_key, api_secret, start, end, limit, output, format):
	"""Generate a treemap of the app/env tuples receiving the most traffic."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		fig = generate_app_env_treemap(df, 'dst', "Top Receiving App/Env Tuples")
		save_figure(fig, output, format)

@cli.command()
@global_options
@click.option('--output', default='traffic_graph', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg']), default='html', help='Output format')
@click.option('--diagram-type', type=click.Choice(['sankey', 'sunburst', 'graphviz', 'plotly']), default='sankey', help='Diagram type')
@click.option('--direction', type=click.Choice(['LR', 'TB']), default='LR', help='Flow directed graph orientation (LR left-right, TB top-bottom)')
@click.option('--width', type=int, default=800, help='Width of the graph in pixels')
@click.option('--height', type=int, default=600, help='Height of the graph in pixels')
def traffic(pce_host, port, org_id, api_key, api_secret, start, end, output, format, diagram_type, direction, width, height, limit):
	"""Generate traffic graph based on Illumio PCE data."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	
	connections = defaultdict(lambda: defaultdict(int))
	for _, row in df.iterrows():
		src = f"{row.get('src_app', 'Unknown')} ({row.get('src_env', 'Unknown')})"
		dst = f"{row.get('dst_app', 'Unknown')} ({row.get('dst_env', 'Unknown')})"
		if src != dst:
			connections[src][dst] += 1

	if diagram_type == 'plotly':
		fig = generate_plotly_directed_graph(connections, width, height)
		if format == 'html':
			fig.write_html(f"{output}.html")
		else:
			fig.write_image(f"{output}.{format}")
	elif diagram_type == 'graphviz':
		content = generate_graphviz_diagram(connections, format, direction)
		with open(f"{output}.{format}", 'wb') as f:
			f.write(content)
	else:
		content = generate_traffic_graph(df, diagram_type, format, direction)
		if isinstance(content, str):
			with open(f"{output}.{format}", 'w') as f:
				f.write(content)
		else:
			with open(f"{output}.{format}", 'w') as f:
				f.write(content)
	
	click.echo(f"Traffic graph saved as {output}.{format}")

def filter_external_connections(df, center_app, center_env):
	"""
	Filter the DataFrame to remove connections where the center app is both source and destination.
	
	:param df: pandas DataFrame containing the traffic data
	:param center_app: The name of the center application
	:param center_env: The environment of the center application
	:return: Filtered pandas DataFrame
	"""
	# Create a mask for connections that are not self-connections of the center app
	mask = ~((df['src_app'] == center_app) & (df['src_env'] == center_env) & 
			 (df['dst_app'] == center_app) & (df['dst_env'] == center_env))
	
	# Apply the mask to filter the DataFrame
	filtered_df = df[mask]
	
	# Log the filtering results
	total_connections = len(df)
	external_connections = len(filtered_df)
	removed_connections = total_connections - external_connections
	
	logging.debug(f"Total connections: {total_connections}")
	logging.debug(f"External connections: {external_connections}")
	logging.debug(f"Removed self-connections: {removed_connections}")
	
	return filtered_df

def generate_app_dependency_graph(df, src_app, src_env, dst_app, dst_env, direction='LR'):
	# Filter the dataframe for the specific app and env
	center_app = f"{src_app} ({src_env})"
	df_filtered = df[(df['src_app'] == src_app) & (df['src_env'] == src_env) |
					 (df['dst_app'] == dst_app) & (df['dst_env'] == dst_env)]

	# Create a directed graph
	G = nx.DiGraph()

	# Add edges to the graph
	for _, row in df_filtered.iterrows():
		source = f"{row['src_app']} ({row['src_env']})"
		target = f"{row['dst_app']} ({row['dst_env']})"
		G.add_edge(source, target)

	# Get all connected nodes
	connected_nodes = set(nx.node_connected_component(G.to_undirected(), center_app))

	# Create subgraph with only connected nodes
	H = G.subgraph(connected_nodes)

	# Compute positions
	if direction == 'LR':
		pos = nx.spring_layout(H, k=1, iterations=50)
		# Adjust x-coordinates
		center_x = pos[center_app][0]
		for node in pos:
			if node != center_app:
				if node in G.predecessors(center_app):
					pos[node] = (pos[node][0] - center_x - 0.1, pos[node][1])
				elif node in G.successors(center_app):
					pos[node] = (pos[node][0] - center_x + 0.1, pos[node][1])
	else:  # 'TB'
		pos = nx.spring_layout(H, k=1, iterations=50)
		# Adjust y-coordinates
		center_y = pos[center_app][1]
		for node in pos:
			if node != center_app:
				if node in G.predecessors(center_app):
					pos[node] = (pos[node][0], pos[node][1] - center_y + 0.1)
				elif node in G.successors(center_app):
					pos[node] = (pos[node][0], pos[node][1] - center_y - 0.1)

	# Create edge trace
	edge_x, edge_y = [], []
	for edge in H.edges():
		x0, y0 = pos[edge[0]]
		x1, y1 = pos[edge[1]]
		edge_x.extend([x0, x1, None])
		edge_y.extend([y0, y1, None])

	edge_trace = go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		mode='lines')

	# Create node trace
	node_x, node_y = [], []
	for node in H.nodes():
		x, y = pos[node]
		node_x.append(x)
		node_y.append(y)

	node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode='markers',
		hoverinfo='text',
		marker=dict(
			showscale=True,
			colorscale='YlGnBu',
			reversescale=True,
			color=[],
			size=10,
			colorbar=dict(
				thickness=15,
				title='Node Connections',
				xanchor='left',
				titleside='right'
			),
			line_width=2))

	# Color node points by the number of connections
	node_adjacencies = []
	node_text = []
	for node, adjacencies in H.adjacency():
		node_adjacencies.append(len(adjacencies))
		node_text.append(f'{node}<br># of connections: {len(adjacencies)}')

	node_trace.marker.color = node_adjacencies
	node_trace.text = node_text

	# Create the figure
	fig = go.Figure(data=[edge_trace, node_trace],
					layout=go.Layout(
						title=f'Dependency Graph for {center_app}',
						titlefont_size=16,
						showlegend=False,
						hovermode='closest',
						margin=dict(b=20,l=5,r=5,t=40),
						annotations=[dict(
							text="",
							showarrow=False,
							xref="paper", yref="paper",
							x=0.005, y=-0.002)],
						xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
						yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
					)

	# Highlight the center node
	center_node_trace = go.Scatter(
		x=[pos[center_app][0]],
		y=[pos[center_app][1]],
		mode='markers',
		marker=dict(
			color='red',
			size=15,
			line=dict(width=2)
		),
		text=[center_app],
		hoverinfo='text'
	)
	fig.add_trace(center_node_trace)

	return fig

def generate_graphviz_diagram(df, src_app, src_env, dst_app, dst_env, output_format, direction, max_levels=2, coalesce_threshold=None, node_limit=100):
	print(type(df))
	center_app = f"{src_app} ({src_env})"
	
	# Create a directed graph
	G = pgv.AGraph(strict=True, directed=True)  # Using strict=True to merge multiple edges

	# Set graph attributes
	G.graph_attr.update({
		'rankdir': direction,
		'splines': 'polyline',  # Changed from 'ortho' for faster rendering
		'nodesep': '0.3',
		'ranksep': '0.5',
		'concentrate': 'true'
	})

	# Create dictionaries to store connections and weights
	connections = defaultdict(lambda: {'incoming': set(), 'outgoing': set()})
	edge_weights = {}

	# Process the dataframe to build connection dictionaries
	for _, row in df.iterrows():
		if row['src_app'] != None:
			source = f"{row['src_app']} ({row['src_env']})"
		elif row['src_env'] != None:
			source = f"NoApp {row['src_env']}"
		else:
			source = f"IPL: {row['src_iplist']}"

		if row['dst_app'] != None:
			target = f"{row['dst_app']} ({row['dst_env']})"
		elif row['dst_env'] != None:
			target = f"NoApp {row['dst_env']}"
		else:
			target = f"IPL: {row['dst_iplist']}"

		if source != target:
			connections[source]['outgoing'].add(target)
			connections[target]['incoming'].add(source)
			edge_weights[(source, target)] = row['num_connections']

	# Function to get important nodes at multiple levels
	def get_important_nodes(start_node, direction, max_level):
		levels = {start_node: 0}
		queue = deque([(start_node, 0)])
		while queue and len(levels) < node_limit:
			node, level = queue.popleft()
			if level >= max_level:
				continue
			neighbors = sorted(connections[node][direction], 
							   key=lambda x: edge_weights.get((node, x) if direction == 'outgoing' else (x, node), 0),
							   reverse=True)
			for neighbor in neighbors[:min(5, len(neighbors))]:  # Limit to top 5 connections
				if neighbor not in levels:
					levels[neighbor] = level + 1
					queue.append((neighbor, level + 1))
		return levels

	# Get incoming and outgoing levels
	incoming_levels = get_important_nodes(center_app, 'incoming', max_levels)
	outgoing_levels = get_important_nodes(center_app, 'outgoing', max_levels)

	# Add nodes to the graph
	G.add_node(center_app, level="center")
	for node, level in incoming_levels.items():
		G.add_node(node, level=f"incoming_{level}")
	for node, level in outgoing_levels.items():
		G.add_node(node, level=f"outgoing_{level}")

	# Add edges to the graph
	for source in incoming_levels:
		for target in connections[source]['outgoing']:
			if target in incoming_levels or target == center_app:
				weight = edge_weights.get((source, target), 1)
				if weight > coalesce_threshold:
					G.add_edge(source, target, weight=weight, color="blue")
	
	for target in outgoing_levels:
		for source in connections[target]['incoming']:
			if source in outgoing_levels or source == center_app:
				weight = edge_weights.get((source, target), 1)
				if weight > coalesce_threshold:
					G.add_edge(source, target, weight=weight, color="green")

	# Set node attributes
	for node in G.nodes():
		attrs = {
			'shape': 'box',
			'style': 'filled',
			'fontsize': '8',
			'margin': '0.1',
			'height': '0.3',
			'width': '0.8'
		}
		if node == center_app:
			attrs.update({
				'shape': 'doubleoctagon',
				'fillcolor': 'red',
				'fontsize': '10',
				'width': '1.2',
				'height': '0.6'
			})
		elif G.get_node(node).attr['level'].startswith('incoming'):
			attrs['fillcolor'] = 'lightblue'
		elif G.get_node(node).attr['level'].startswith('outgoing'):
			attrs['fillcolor'] = 'lightgreen'
		G.get_node(node).attr.update(attrs)

	# Set edge attributes
	for edge in G.edges():
		weight = int(edge_weights.get((edge[0], edge[1]), 1))
		G.get_edge(*edge).attr['penwidth'] = str(min(3, 1 + weight / 1000))

	# Apply the layout
	G.layout(prog='dot')

	# Render the graph
	if output_format == 'svg':
		G.draw(f'app_dependency.svg', format='svg')
		with open('app_dependency.svg', 'rb') as f:
			return f.read()
	elif output_format == 'png':
		G.draw(f'app_dependency.png', format='png')
		with open('app_dependency.png', 'rb') as f:
			return f.read()
	else:
		raise ValueError(f"Unsupported output format: {output_format}")
def generate_destination_treemap(df, src_app, src_env, title):
	# Filter the dataframe for the specific source app and environment
	df_mask = ~((df['src_app'] == src_app) & (df['src_env'] == src_env) & (df['dst_app'] == src_app) & (df['dst_env'] == src_env))

	df_filtered = df[(df['src_app'] == src_app) & (df['src_env'] == src_env)]
	df_same_app_traffic = df_filtered
	df_filtered = df_same_app_traffic[df_mask]
	
	# Group by destination app and environment, and count the connections
	dst_counts = df_filtered.groupby(['dst_env', 'dst_app']).size().reset_index(name='count')
	
	# Create the treemap
	fig = px.treemap(dst_counts, 
					 path=['dst_env', 'dst_app'], 
					 values='count',
					 title=title)
	fig.update_traces(textinfo="label+value+percent parent")
	return fig

def generate_source_treemap(df, dst_app, dst_env, title):
	# Filter the dataframe for the specific destination app and environment
	df_filtered = df[(df['dst_app'] == dst_app) & (df['dst_env'] == dst_env)]
	df_mask = ~((df['src_app'] == dst_app) & (df['src_env'] == dst_env) & (df['dst_app'] == dst_app) & (df['dst_env'] == dst_env))

	df_same_app_traffic = df_filtered
	df_filtered = df_same_app_traffic[df_mask]
	
	# Group by source app and environment, and count the connections
	src_counts = df_filtered.groupby(['src_env', 'src_app']).size().reset_index(name='count')
	
	# Create the treemap
	fig = px.treemap(src_counts, 
					 path=['src_env', 'src_app'], 
					 values='count',
					 title=title)
	fig.update_traces(textinfo="label+value+percent parent")
	return fig

@cli.command()
@global_options
@click.option('--src-app', required=True, help='Source application name')
@click.option('--src-env', required=True, help='Source environment name')
@click.option('--output', default='destination_treemap', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg', 'pdf']), default='html', help='Output format')
@click.option('--width', type=int, default=1200, help='Width of the diagram in pixels')
@click.option('--height', type=int, default=800, help='Height of the diagram in pixels')
def destination_treemap(pce_host, port, org_id, api_key, api_secret, start, end, limit, src_app, src_env, output, format, width, height):
	"""Generate a treemap of destinations for a specific source app and environment."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		title = f"Destinations for {src_app} ({src_env})"
		fig = generate_destination_treemap(df, src_app, src_env, title)
		fig.update_layout(width=width, height=height)
		if format == 'html':
			fig.write_html(f"{output}.html")
		else:
			fig.write_image(f"{output}.{format}")
		click.echo(f"Destination treemap saved as {output}.{format}")
	else:
		click.echo("Failed to retrieve traffic data.")

@cli.command()
@global_options
@click.option('--dst-app', required=True, help='Destination application name')
@click.option('--dst-env', required=True, help='Destination environment name')
@click.option('--output', default='source_treemap', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg', 'pdf']), default='html', help='Output format')
@click.option('--width', type=int, default=1200, help='Width of the diagram in pixels')
@click.option('--height', type=int, default=800, help='Height of the diagram in pixels')
def source_treemap(pce_host, port, org_id, api_key, api_secret, start, end, limit, dst_app, dst_env, output, format, width, height):
	"""Generate a treemap of sources for a specific destination app and environment."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		title = f"Sources for {dst_app} ({dst_env})"
		fig = generate_source_treemap(df, dst_app, dst_env, title)
		fig.update_layout(width=width, height=height)
		if format == 'html':
			fig.write_html(f"{output}.html")
		else:
			fig.write_image(f"{output}.{format}")
		click.echo(f"Source treemap saved as {output}.{format}")
	else:
		click.echo("Failed to retrieve traffic data.")

@cli.command()
@global_options
@click.option('--src-app', required=True, help='Source application name')
@click.option('--src-env', required=True, help='Source environment name')
@click.option('--dst-app', required=True, help='Destination application name')
@click.option('--dst-env', required=True, help='Destination environment name')
@click.option('--output', default='app_dependency_graphviz', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['svg', 'png']), default='svg', help='Output format')
@click.option('--direction', type=click.Choice(['LR', 'TB']), default='LR', help='Graph orientation (LR: left-right, TB: top-bottom)')
@click.option('--levels', type=int, default=3, help='Maximum number of levels to show')
@click.option('--coalesce', type=int, default=10, help='Threshold for coalescing nodes')
@click.option('--node-limit', type=int, default=100, help='Node limit for graph')
def app_dependency_graphviz(pce_host, port, org_id, api_key, api_secret, start, end, limit, src_app, src_env, dst_app, dst_env, output, format, direction, levels, coalesce, node_limit):
	"""Generate a GraphViz diagram for app dependencies."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		content = generate_graphviz_diagram(df, src_app, src_env, dst_app, dst_env, format, direction, max_levels=levels, coalesce_threshold=coalesce, node_limit=node_limit)
		
		with open(f"{output}.{format}", 'wb') as f:
			f.write(content)
		
		click.echo(f"App dependency GraphViz diagram saved as {output}.{format}")
	else:
		click.echo("Failed to retrieve traffic data.")

def generate_app_dependency_sankey(df, src_app, src_env, dst_app, dst_env, coalesce_threshold=None, max_link_value=None):
	center_app = f"{src_app} ({src_env})"
	
	# Data filtering
	df = df[((df['src_app'] == src_app) & (df['src_env'] == src_env)) |
			((df['dst_app'] == dst_app) & (df['dst_env'] == dst_env))]
	df = df[(df['src_app'] != df['dst_app']) | (df['src_env'] != df['dst_env'])]
	
	# Create dictionaries for incoming and outgoing connections
	incoming = defaultdict(int)
	outgoing = defaultdict(int)
	
	for _, row in df.iterrows():
		source = f"{row['src_app']} ({row['src_env']})"
		target = f"{row['dst_app']} ({row['dst_env']})"
		
		if target == center_app and source != center_app:
			incoming[source] += row['num_connections']
		elif source == center_app and target != center_app:
			outgoing[target] += row['num_connections']

	# Apply coalescing if threshold is set
	if coalesce_threshold is not None:
		incoming = coalesce_nodes(incoming, coalesce_threshold, "Other Incoming")
		outgoing = coalesce_nodes(outgoing, coalesce_threshold, "Other Outgoing")

	# Prepare Sankey data
	nodes = list(incoming.keys()) + [center_app] + list(outgoing.keys())
	node_indices = {node: i for i, node in enumerate(nodes)}

	# Node positioning
	node_x, node_y = calculate_node_positions(nodes, center_app, incoming, outgoing)

	# Color scheme
	node_color = generate_color_scheme(nodes, center_app)

	links = {'source': [], 'target': [], 'value': [], 'color': [], 'label': []}

	# Function to cap link values
	def cap_value(value):
		return min(value, max_link_value) if max_link_value is not None else value

	# Add incoming connections
	for source, value in incoming.items():
		links['source'].append(node_indices[source])
		links['target'].append(node_indices[center_app])
		capped_value = cap_value(value)
		links['value'].append(capped_value)
		links['color'].append('rgba(0, 0, 255, 0.2)')  # Light blue
		links['label'].append(f"{source} → {center_app}: {value}")

	# Add outgoing connections
	for target, value in outgoing.items():
		links['source'].append(node_indices[center_app])
		links['target'].append(node_indices[target])
		capped_value = cap_value(value)
		links['value'].append(capped_value)
		links['color'].append('rgba(0, 255, 0, 0.2)')  # Light green
		links['label'].append(f"{center_app} → {target}: {value}")

	# Create the Sankey diagram
	fig = go.Figure(data=[go.Sankey(
		node = dict(
		  pad = 15,
		  thickness = 20,
		  line = dict(color = "black", width = 0.5),
		  label = nodes,
		  color = node_color,
		  x = node_x,
		  y = node_y
		),
		link = dict(
		  source = links['source'],
		  target = links['target'],
		  value = links['value'],
		  color = links['color'],
		  label = links['label'],
		  hovertemplate = '%{label}<extra></extra>'
	  ))])

	# Layout and interactivity
	fig.update_layout(
		title_text=f"Dependency Diagram for {center_app}",
		font_size=10,
		autosize=False,
		width=1200,
		height=800,
		hoverlabel=dict(
			bgcolor="white",
			font_size=12,
			font_family="Rockwell"
		)
	)
	
	# Handle large datasets
	if len(nodes) > 50:
		fig.update_layout(showlegend=False)
	
	return fig

# Helper functions remain the same
def coalesce_nodes(node_dict, threshold, other_label):
	coalesced = {k: v for k, v in node_dict.items() if v > threshold}
	other_sum = sum(v for k, v in node_dict.items() if v <= threshold)
	if other_sum > 0:
		coalesced[other_label] = other_sum
	return coalesced

def generate_color_scheme(nodes, center_app):
	color_scale = px.colors.qualitative.Plotly
	node_color = []
	
	for node in nodes:
		if node == center_app:
			node_color.append('red')
		elif "Other" in node:
			node_color.append('gray')
		else:
			node_color.append(color_scale[hash(node) % len(color_scale)])
	
	return node_color

def calculate_node_positions(nodes, center_app, incoming, outgoing):
	node_x = []
	node_y = []
	incoming_count = len(incoming)
	outgoing_count = len(outgoing)
	total_nodes = len(nodes)

	# Set center node position
	center_index = nodes.index(center_app)
	initial_positions = np.zeros((total_nodes, 2))
	initial_positions[center_index] = [0.5, 0.5]

	# Set initial positions for incoming and outgoing nodes
	for i, node in enumerate(nodes):
		if node in incoming:
			angle = 2 * math.pi * (i / incoming_count)
			initial_positions[i] = [0.25 + 0.25 * math.cos(angle), 0.5 + 0.4 * math.sin(angle)]
		elif node in outgoing and node != center_app:
			angle = 2 * math.pi * ((i - incoming_count) / outgoing_count)
			initial_positions[i] = [0.75 + 0.25 * math.cos(angle), 0.5 + 0.4 * math.sin(angle)]

	# Define the objective function to minimize
	def objective(positions):
		pos = positions.reshape(-1, 2)
		total_distance = 0
		for i in range(total_nodes):
			for j in range(i+1, total_nodes):
				dist = np.linalg.norm(pos[i] - pos[j])
				total_distance += 1 / (dist ** 2)  # Use inverse square to create stronger repulsion
		return total_distance

	# Define constraints to keep nodes within bounds and maintain left/right separation
	def constraint(positions):
		pos = positions.reshape(-1, 2)
		constraints = []
		for i, node in enumerate(nodes):
			if node in incoming:
				constraints.append(pos[i][0] - 0.45)  # x <= 0.45
			elif node in outgoing and node != center_app:
				constraints.append(0.55 - pos[i][0])  # x >= 0.55
			constraints.extend([pos[i][0], pos[i][1], 1 - pos[i][0], 1 - pos[i][1]])  # 0 <= x,y <= 1
		return np.array(constraints)

	# Optimize node positions
	result = minimize(
		objective,
		initial_positions.flatten(),
		method='SLSQP',
		constraints={'type': 'ineq', 'fun': constraint},
		options={'maxiter': 1000}
	)

	optimized_positions = result.x.reshape(-1, 2)

	# Extract x and y coordinates
	node_x = optimized_positions[:, 0].tolist()
	node_y = optimized_positions[:, 1].tolist()

	return node_x, node_y

def create_circular_dependency_diagram(df, center_app, output_file='circular_dependency.html', max_connections=10, max_levels=3):
    # Create a graph of connections
    G = nx.DiGraph()
    
    def add_connections(app, level=0):
        if level >= max_levels:
            return
        outgoing = df[df['src_app'] == app.split(' (')[0]]['dst_app'] + ' (' + df[df['src_app'] == app.split(' (')[0]]['dst_env'] + ')'
        incoming = df[df['dst_app'] == app.split(' (')[0]]['src_app'] + ' (' + df[df['dst_app'] == app.split(' (')[0]]['src_env'] + ')'
        
        for target in outgoing.unique():
            if target != app and target not in G.nodes():
                G.add_edge(app, target)
                add_connections(target, level + 1)
        
        for source in incoming.unique():
            if source != app and source not in G.nodes():
                G.add_edge(source, app)
                add_connections(source, level + 1)

    G.add_node(center_app)
    add_connections(center_app)

    # Limit to top N connections for the center app
    center_edges = list(G.in_edges(center_app)) + list(G.out_edges(center_app))
    center_edges = sorted(center_edges, key=lambda x: df[(df['src_app'] == x[0].split(' (')[0]) & (df['dst_app'] == x[1].split(' (')[0])]['num_connections'].sum(), reverse=True)[:max_connections]
    G = G.edge_subgraph(center_edges).copy()

    # Calculate positions
    pos = nx.spring_layout(G)

    # Function to wrap text and determine font size
    def wrap_text_and_size(text, max_width=15):
        wrapped_text = textwrap.fill(text, width=max_width)
        lines = wrapped_text.count('\n') + 1
        font_size = max(8, min(12, 20 // lines))
        return wrapped_text, font_size

    # Create edge traces with arrows
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Calculate the angle of the edge
        angle = math.atan2(y1 - y0, x1 - x0)
        
        # Shorten the edge to make room for the arrow
        shorten = 0.05
        x1, y1 = x1 - shorten * math.cos(angle), y1 - shorten * math.sin(angle)
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines+markers',
            marker=dict(
                symbol='arrow',
                size=10,
                angleref='previous',
                standoff=5
            )
        )
        edge_traces.append(edge_trace)

    # Create node trace with wrapped text and adaptive font size
    node_x, node_y, node_text, node_font_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        wrapped_text, font_size = wrap_text_and_size(node)
        node_text.append(wrapped_text)
        node_font_size.append(font_size)

    node_colors = ['red' if node == center_app else 'lightblue' for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=node_font_size),
        marker=dict(
            color=node_colors,
            size=30,
            line_width=2
        ))

    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=f'Dependency Diagram for {center_app}',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Adjust layout to provide more space for nodes
    fig.update_layout(
        autosize=False,
        width=1200,
        height=1200,
    )

    # Save the figure
    fig.write_html(output_file)
    print(f"Diagram saved to {output_file}")


@cli.command()
@global_options
@click.option('--src-app', required=True, help='Source application name')
@click.option('--src-env', required=True, help='Source environment name')
@click.option('--output', default='circular_dependency.html', help='Output filename')
@click.option('--max-connections', default=10, type=int, help='Maximum number of connections to show per direction')
def circular_dependency(pce_host, port, org_id, api_key, api_secret, start, end, limit, src_app, src_env, output, max_connections):
    """Generate a circular dependency diagram for a specific app."""
    df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
    if df is not None:
        center_app = f"{src_app} ({src_env})"
        create_circular_dependency_diagram(df, center_app, output, max_connections)
        click.echo(f"Circular dependency diagram saved as {output}")
    else:
        click.echo("Failed to retrieve traffic data.")

@cli.command()
@global_options
@click.option('--src-app', required=True, help='Source application name')
@click.option('--src-env', required=True, help='Source environment name')
@click.option('--dst-app', required=True, help='Destination application name')
@click.option('--dst-env', required=True, help='Destination environment name')
@click.option('--output', default='app_dependency_sankey', help='Output filename (without extension)')
@click.option('--format', type=click.Choice(['html', 'png', 'jpg', 'svg', 'pdf']), default='html', help='Output format')
@click.option('--width', type=int, default=1200, help='Width of the diagram in pixels')
@click.option('--height', type=int, default=800, help='Height of the diagram in pixels')
@click.option('--coalesce', type=int, default=None, help='Threshold for coalescing nodes')
@click.option('--max-link-value', type=int, default=None, help='Maximum value for link thickness')
def app_dependency_sankey(pce_host, port, org_id, api_key, api_secret, start, end, limit, src_app, src_env, dst_app, dst_env, output, format, width, height, coalesce, max_link_value):
	"""Generate a Sankey diagram for app dependencies."""
	df = get_traffic_data(pce_host, port, org_id, api_key, api_secret, start, end, limit)
	if df is not None:
		fig = generate_app_dependency_sankey(df, src_app, src_env, dst_app, dst_env, coalesce_threshold=coalesce, max_link_value=max_link_value)
		fig.update_layout(width=width, height=height)
		if format == 'html':
			fig.write_html(f"{output}.html")
		else:
			fig.write_image(f"{output}.{format}")
		click.echo(f"App dependency Sankey diagram saved as {output}.{format}")
	else:
		click.echo("Failed to retrieve traffic data.")

if __name__ == '__main__':
	cli()
