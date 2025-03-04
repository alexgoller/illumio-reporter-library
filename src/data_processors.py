import pandas as pd
from collections import Counter

class BaseProcessor:
    def __init__(self, data):
        self.df = pd.DataFrame(data)

    def get_top_n(self, column, n=10):
        top_n = self.df[column].value_counts().head(n).reset_index()
        top_n.columns = [column, 'Count']
        return top_n

class WorkloadProcessor(BaseProcessor):
    def __init__(self, data):
        super().__init__(data)
        print("Columns in WorkloadProcessor DataFrame:", self.df.columns)

    def get_os_summary(self):
        os_summary = self.df['os_id'].value_counts().reset_index()
        os_summary.columns = ['Operating System', 'Count']
        return os_summary

    def get_online_status(self):
        online_status = self.df['online'].value_counts().reset_index()
        online_status.columns = ['Status', 'Count']
        return online_status

    def get_top_hostnames(self, n=10):
        return self.get_top_n('hostname', n)

    def get_os_summary_for_graph(self):
        return self.df['os_id'].value_counts()

    def get_enforcement_state_summary(self):
        enforcement_summary = self.df['enforcement_state'].value_counts()
        return enforcement_summary

    def get_enforcement_state_for_graph(self):
        return self.get_enforcement_state_summary()

    def get_enforcement_mode_summary(self):
        enforcement_summary = self.df['enforcement_mode'].value_counts()
        return enforcement_summary

    def get_enforcement_mode_for_graph(self):
        return self.get_enforcement_mode_summary()

    def get_workloads_by_network(self):
        """
        Group workloads by their network and return a summary DataFrame
        """
        # Group workloads by network
        network_groups = self.df.groupby('network')
        
        # Create summary for each network
        network_summaries = []
        for network, group in network_groups:
            summary = {
                'Network': network if network else 'No Network',
                'Total Workloads': len(group),
                'Online Workloads': len(group[group['online'] == True]),
                'Offline Workloads': len(group[group['online'] == False]),
                'Unique OS Types': group['os_id'].nunique(),
                'Enforcement Modes': group['enforcement_mode'].value_counts().to_dict()
            }
            network_summaries.append(summary)
            
        # Convert to DataFrame and sort by total workloads
        summary_df = pd.DataFrame(network_summaries)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('Total Workloads', ascending=False)
            
        return summary_df

class WorkloadServiceProcessor(BaseProcessor):
    def __init__(self, detailed_workloads):
        service_data = []
        for workload in detailed_workloads:
            if workload.services is not None:
                if workload.services.open_service_ports is not None:
                    for port in workload.services.open_service_ports:
                        service_data.append({
                            'hostname': workload.hostname,
                            'port': f"{port.port}/{port.protocol}",  # Combine port and protocol
                            'process_name': port.process_name,
                        })
        super().__init__(service_data)

    def get_top_ports(self, n=10):
        return self.get_top_n('port', n)

    def get_protocol_summary(self):
        # Extract protocol from the 'port' column
        self.df['protocol'] = self.df['port'].apply(lambda x: x.split('/')[1])
        protocols = self.df['protocol'].value_counts().reset_index()
        protocols.columns = ['Protocol', 'Count']
        return protocols

    def get_open_ports_summary(self):
        open_ports = self.df['port'].value_counts().sort_values(ascending=False)
        open_ports = open_ports.reset_index()
        open_ports.columns = ['Port', 'Count']
        
        # Add a percentage column
        total = open_ports['Count'].sum()
        open_ports['Percentage'] = (open_ports['Count'] / total * 100).round(2)
        
        # Create a formatted label
        open_ports['Label'] = open_ports.apply(lambda row: f"{row['Port']} ({row['Count']} - {row['Percentage']}%)", axis=1)
        
        return open_ports

class TrafficProcessor(BaseProcessor):
    def __init__(self, data, label_href_map=None, value_href_map=None):
        super().__init__(data)
        self.label_href_map = label_href_map or {}
        self.value_href_map = value_href_map or {}
        self.df = self.to_dataframe(data)
        print(f"TrafficProcessor initialized with {len(data)} flows")
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Available columns: {self.df.columns.tolist()}")

    def get_top_src_ips(self, n=10):
        return self.get_top_n('src_ip', n)

    def get_top_dst_ips(self, n=10):
        return self.get_top_n('dst_ip', n)

    def get_top_ports(self, n=10):
        if self.df.empty:
            print(f"Warning: DataFrame is empty. Cannot get top {n} ports.")
            return pd.DataFrame(columns=['port', 'connections'])
        if 'port' not in self.df.columns:
            print(f"Warning: 'port' column not found. Available columns: {self.df.columns.tolist()}")
            return pd.DataFrame(columns=['port', 'connections'])
        return (self.df.groupby('port')['num_connections']
                .sum()
                .sort_values(ascending=False)
                .head(n)
                .reset_index()
                .rename(columns={'num_connections': 'connections'}))

    def get_protocol_summary(self):
        if self.df.empty:
            print("Warning: DataFrame is empty. Cannot get protocol summary.")
            return pd.DataFrame(columns=['protocol', 'connections'])
        if 'proto' not in self.df.columns:
            print(f"Warning: 'proto' column not found. Available columns: {self.df.columns.tolist()}")
            return pd.DataFrame(columns=['protocol', 'connections'])
        return (self.df.groupby('proto')['num_connections']
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={'proto': 'protocol', 'num_connections': 'connections'}))

    def to_dataframe(self, flows):
        
        print(f'Label href map: {self.label_href_map}')
        print(f'Value href map: {self.value_href_map}')

        if not flows:
            print("Warning: Empty flows list received.")
            return pd.DataFrame()

        series_array = []
        for flow in flows:
            try:
                f = {
                    'src_ip': flow.src.ip,
                    'src_hostname': flow.src.workload.name if flow.src.workload is not None else None,
                    'dst_ip': flow.dst.ip,
                    'dst_hostname': flow.dst.workload.name if flow.dst.workload is not None else None,
                    'proto': flow.service.proto,
                    'port': flow.service.port,
                    'process_name': flow.service.process_name,
                    'service_name': flow.service.service_name,
                    'policy_decision': flow.policy_decision,
                    'flow_direction': flow.flow_direction,
                    'num_connections': flow.num_connections,
                    'first_detected': flow.timestamp_range.first_detected,
                    'last_detected': flow.timestamp_range.last_detected,
                }

                # Add src and dst app and env labels
                if flow.src.workload:
                    for l in flow.src.workload.labels:
                        if l.href in self.label_href_map:
                            key = self.label_href_map[l.href]['key']
                            value = self.label_href_map[l.href]['value']
                            f[f'src_{key}'] = value

                if flow.dst.workload:
                    for l in flow.dst.workload.labels:
                        if l.href in self.label_href_map:
                            key = self.label_href_map[l.href]['key']
                            value = self.label_href_map[l.href]['value']
                            f[f'dst_{key}'] = value

                series_array.append(f)
            except AttributeError as e:
                print(f"Error processing flow: {e}")
                print(f"Flow object: {flow}")

        df = pd.DataFrame(series_array)
        print(f"DataFrame info:\n{df.info()}")
        return df

    def summarize_traffic_for_ai_advisor(self):
        print("Starting summarize_traffic_for_ai_advisor method")
        
        # Use available columns for grouping
        group_columns = ['src_app', 'src_env', 'dst_app', 'dst_env', 'proto', 'port']
        
        print(f"Group columns: {group_columns}")
        print(f"DataFrame shape before grouping: {self.df.shape}")
        print(f"DataFrame columns: {self.df.columns.tolist()}")
        print(f"First few rows of DataFrame:\n{self.df.head()}")

        # Group by available columns
        summary = self.df.groupby(group_columns)['num_connections'].sum().reset_index()
        
        print(f"Summary shape after grouping: {summary.shape}")
        print(f"Summary columns: {summary.columns.tolist()}")
        print(f"First few rows of summary:\n{summary.head()}")

        # Sort by number of connections in descending order
        summary = summary.sort_values('num_connections', ascending=False)

        # Convert to a more readable format
        summary_list = []
        for _, row in summary.iterrows():
            src_info = f"{row['src_app']} ({row['src_env']})" if 'src_app' in row else row['src_ip']
            dst_info = f"{row['dst_app']} ({row['dst_env']})" if 'dst_app' in row else row['dst_ip']

            if src_info != dst_info:
                summary_list.append(
                    f"From {src_info} to {dst_info} on port {row['port']}: {row['num_connections']} connections"
                )

        return "\n".join(summary_list)

    # Add more methods for traffic-specific statistics as needed

    def get_cross_environment_traffic(self):
        # Group by source and destination environments
        cross_env_traffic = self.df.groupby(['src_env', 'dst_env'])['num_connections'].sum().reset_index()
        
        # Create a hierarchical structure for the treemap
        cross_env_traffic['total'] = 'Total'
        cross_env_traffic = cross_env_traffic.rename(columns={'src_env': 'Source', 'dst_env': 'Destination', 'num_connections': 'Connections'})
        
        return cross_env_traffic

    def create_app_group(self, app, env):
        return f"{app} ({env})"

    def get_top_app_groups(self, n=10, source=True):
        if source:
            app_column, env_column = 'src_app', 'src_env'
        else:
            app_column, env_column = 'dst_app', 'dst_env'
        
        app_groups = self.df.apply(lambda row: self.create_app_group(row[app_column], row[env_column]), axis=1)
        return app_groups.value_counts().nlargest(n).to_dict()

    def get_top_source_app_groups(self, n=10):
        return self.get_top_app_groups(n, source=True)

    def get_top_destination_app_groups(self, n=10):
        return self.get_top_app_groups(n, source=False)

    def create_app_group_summary(self):
        top_sources = self.get_top_source_app_groups()
        top_destinations = self.get_top_destination_app_groups()
        
        summary = "Top 10 Source App Groups:\n"
        for app_group, count in top_sources.items():
            summary += f"- {app_group}: {count} connections\n"
        
        summary += "\nTop 10 Destination App Groups:\n"
        for app_group, count in top_destinations.items():
            summary += f"- {app_group}: {count} connections\n"
        
        return summary

    def get_top_processes(self, n=10):
        # Group by process_name and sum the num_connections
        process_traffic = self.df.groupby('process_name')['num_connections'].sum().sort_values(ascending=False)
        return process_traffic.nlargest(n).to_dict()

    def create_process_summary(self):
        top_processes = self.get_top_processes()
        
        summary = "## Top 10 Processes by Traffic Volume\n\n"
        for process, connections in top_processes.items():
            summary += f"- **{process}**: {connections} connections\n"
        
        return summary