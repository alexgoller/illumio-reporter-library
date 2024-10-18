from src.illumio_client import IllumioClient
from src.config import config
from illumio import TrafficQuery
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        self.client = IllumioClient(
            hostname=config.hostname,
            port=config.port,
            org_id=config.org_id,
            api_key_id=config.api_key_id,
            api_key_secret=config.api_key_secret,
            ignore_tls=config.ignore_tls
        )

    def fetch_workload_data(self):
        pce = self.client.get_pce()
        workloads = pce.workloads.get_all({ 'representation': 'workloads_and_services' })
        return workloads

    def fetch_workload_services(self):
        workloads = self.client.pce.workloads.get()
        detailed_workloads = []
        for workload in workloads:
            detailed_workload = self.client.pce.workloads.get_by_reference({'href': workload.href})
            detailed_workloads.append(detailed_workload)
        return detailed_workloads
    
    def fetch_labels(self):
        pce = self.client.get_pce()
        labels = pce.labels.get(params={'max_results': 10000})
        return labels

    def fetch_label_href_map(self):
        labels = self.fetch_labels()
        label_href_map = {}
        for label in labels:
            label_href_map[label.href] = {"key": label.key, "value": label.value}
        return label_href_map

    def fetch_value_href_map(self):
        labels = self.fetch_labels()
        value_href_map = {}
        for label in labels:
            value_href_map["{}={}".format(label.key, label.value)] = label.href
        return value_href_map

    def fetch_traffic_data(self):
        pce = self.client.get_pce()

        if not pce.check_connection():
            click.echo("Connection to PCE failed.")
            return None

        label_href_map = {}
        value_href_map = {}

        # todo: add label caching and lookup if there is a miss
        for l in pce.labels.get(params={'max_results': 10000}):
            label_href_map[l.href] = {"key": l.key, "value": l.value}
            value_href_map["{}={}".format(l.key, l.value)] = l.href

        d_start = datetime.now() - timedelta(days=90)
        d_end = datetime.now()

        print(f"d_start: {d_start}")
        print(f"d_end: {d_end}")

        traffic_query = TrafficQuery.build(
            start_date=d_start.strftime("%Y-%m-%d"),
            end_date=d_end.strftime("%Y-%m-%d"),
            include_services=[],
            exclude_services=[
                {"port": 53},
                {"port": 137},
                {"port": 138},
                {"port": 139}
            ],
            exclude_destinations=[
                {"transmission": "broadcast"},
                {"transmission": "multicast"}
            ],
            policy_decisions=['allowed', 'potentially_blocked', 'unknown'],
            max_results=10000
        )

        all_traffic = pce.get_traffic_flows_async(
            query_name='all-traffic',
            traffic_query=traffic_query
        )

        print(f'get_traffic_data: {type(all_traffic)}')
        print(f'Length of all_traffic: {len(all_traffic)}')

        return all_traffic

    def fetch_events(self):
        pce = self.client.get_pce()
        events = pce.events.get()
        return events

    def fetch_rulesets(self):
        pce = self.client.get_pce()
        rulesets = pce.rulesets.get()
        return rulesets

    def fetch_rules(self):
        pce = self.client.get_pce()
        rules = pce.rules.get()
        return rules
