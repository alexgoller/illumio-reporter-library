from src.data_fetcher import DataFetcher
from src.data_processors import TrafficProcessor, WorkloadProcessor, WorkloadServiceProcessor
from src.report_generator import ReportGenerator
from src.graph_generator import GraphGenerator
from src.ai_models import AnthropicModel, OpenAIModel, OllamaModel
from src.ai_advisor import TrafficAIAdvisor, TrafficGraphAdvisor, ATTACKAdvisor, AIAdvisor
import os
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
from datetime import datetime, timedelta


def illumio_color_scheme():
    return {
        'title': colors.Color(0.39, 0.41, 0.45),  # Server Slate
        'heading': colors.Color(0.96, 0.39, 0.21),  # Illumio Orange
        'table_header_bg': colors.Color(0.39, 0.41, 0.45),  # Server Slate
        'table_header_text': colors.white,
        'table_body_bg': colors.white,
        'table_body_text': colors.Color(0.39, 0.41, 0.45),  # Server Slate
        'table_grid': colors.Color(0.63, 0.64, 0.67),  # Server Slate 75%
    }

def generate_test_traffic_report(output_file):
    # Initialize DataFetcher
    data_fetcher = DataFetcher()
    graph_generator = GraphGenerator()


    # Initialize report generator with customization
    report_generator = ReportGenerator(
        output_file,
        color_scheme=illumio_color_scheme(),
        header_text="Confidential - Internal Use Only",
        footer_text="Generated on " + datetime.now().strftime("%Y-%m-%d"),
        logo_path="examples/illumio-logo.png"
    )

    # Create report
    report_generator.add_title("Risk analysis report")

    # Generate traffic graphs

    workloads = data_fetcher.fetch_workload_data()
    detailed_workloads = data_fetcher.fetch_workload_services()

    # Fetch traffic data
    try:
        traffic_flows = data_fetcher.fetch_traffic_data()
        # Fetch traffic data
        label_href_map = data_fetcher.fetch_label_href_map()
        value_href_map = data_fetcher.fetch_value_href_map()    

        print(f"Fetched {len(traffic_flows)} traffic flows")
    except Exception as e:
        print(f"Error fetching traffic data: {str(e)}")
        traffic_flows = []

    # Initialize processors
    workload_processor = WorkloadProcessor(workloads)
    service_processor = WorkloadServiceProcessor(detailed_workloads)

    # Initialize GraphGenerator
    graph_generator = GraphGenerator()

    # Choose your model
    model = AnthropicModel(api_key=os.getenv('ANTHROPIC_API_KEY'), model="claude-3-5-sonnet-20240620")
    # model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini")
    # Or use one of these:
    # model = OllamaModel(model="llama3.1:8b")

    ai_advisor = AIAdvisor(model)
    traffic_ai_advisor = TrafficAIAdvisor(model)
    traffic_graph_advisor = TrafficGraphAdvisor(model)
    attack_advisor = ATTACKAdvisor(model)  # Initialize the ATTACKAdvisor
    # Initialize AIAdvisor

    # Generate OS summary graph
    os_summary_data = workload_processor.get_os_summary_for_graph()
    os_summary_graph = graph_generator.generate_pie_chart(
        os_summary_data, 
        "OS Distribution", 
        "os_summary.png"
    )

    # Generate enforcement mode summary graph
    enforcement_mode_data = workload_processor.get_enforcement_mode_for_graph()
    enforcement_mode_graph = graph_generator.generate_pie_chart(
        enforcement_mode_data,
        "Workload Enforcement Mode Distribution",
        "enforcement_mode_summary.png"
    )

    report_generator.add_section("Workload Summary")
    report_generator.add_explanation(
        "This section provides an overview of workloads in the environment.",
        icon_path="examples/info_icon.png",
        icon_position='left'
    )
    report_generator.add_table(workload_processor.get_os_summary())
    report_generator.add_table(workload_processor.get_online_status())
    
    report_generator.add_section("Top 10 Hostnames")
    report_generator.add_explanation(
        "This list represents the most frequently occurring hostnames in your environment.",
        icon_path="examples/info_icon.png",
        icon_position='left'
    )
    report_generator.add_table(workload_processor.get_top_hostnames())

    report_generator.add_section("Service Summary")
    report_generator.add_explanation(
        "This section provides an overview of services and open ports across all workloads.",
        icon_path="examples/info_icon.png",
        icon_position='left'
    )
    report_generator.add_table(service_processor.get_top_ports())
    report_generator.add_table(service_processor.get_protocol_summary())

    # Generate open ports summary graph
    open_ports_data = service_processor.get_open_ports_summary()
    open_ports_graph = graph_generator.generate_horizontal_bar_chart(
        open_ports_data,
        "Top 20 Open Ports",
        "open_ports_summary.png",
        "Number of Machines",
        "Port/Protocol (Count - Percentage)"
    )

    # Add the graph to the report
    report_generator.add_section("Open Ports Summary")
    report_generator.add_explanation("This chart shows the distribution of the top 20 open ports across all workloads, sorted by the number of machines with each port open. Ports are represented as 'port_number/protocol', followed by the count and percentage of machines with this port open.")
    report_generator.add_graph(open_ports_graph)

    # Add OS distribution graph to the report
    report_generator.add_section("OS Distribution")
    report_generator.add_explanation("This chart shows the distribution of operating systems across all workloads.")
    report_generator.add_graph(os_summary_graph)

    # Add enforcement mode distribution graph to the report
    report_generator.add_section("Workload Enforcement Mode Distribution")
    report_generator.add_explanation("This chart shows the distribution of enforcement modes across all workloads.")
    report_generator.add_graph(enforcement_mode_graph)

    # Add enforcement mode summary table
    report_generator.add_section("Enforcement Mode Summary")
    report_generator.add_explanation("This table provides a summary of workload enforcement modes.")
    enforcement_summary = workload_processor.get_enforcement_mode_summary().reset_index()
    enforcement_summary.columns = ['Enforcement Mode', 'Count']
    report_generator.add_table(enforcement_summary)

    if not traffic_flows:
        print("No traffic data available. Skipping traffic section.")
        report_generator.add_section("No Traffic Data")
        report_generator.add_paragraph("No traffic data is available for analysis.")
    else:
        # Initialize traffic processor
        traffic_processor = TrafficProcessor(traffic_flows, label_href_map, value_href_map)

        # Generate traffic summary
        report_generator.add_section("Traffic Summary")
        report_generator.add_explanation(
            "This section provides an overview of network traffic in the environment, "
            "highlighting the top 10 source and destination app groups.",
            icon_path="examples/info_icon.png",
            icon_position='left'
        )

        # Add top app groups summary
        app_group_summary = traffic_processor.create_app_group_summary()
        report_generator.add_paragraph(app_group_summary)

        # Generate and add bar charts for top app groups
        top_source_app_groups = traffic_processor.get_top_source_app_groups()
        if top_source_app_groups:  # Check if the dictionary is not empty
            source_chart = graph_generator.generate_bar_chart(
                top_source_app_groups, 
                "Top 10 Source App Groups", 
                "App Group", 
                "Number of Connections"
            )
            report_generator.add_plotly_figure(source_chart)
        else:
            print("No source app groups data available for graph generation.")

        top_dest_app_groups = traffic_processor.get_top_destination_app_groups()
        if top_dest_app_groups:  # Check if the dictionary is not empty
            dest_chart = graph_generator.generate_bar_chart(
                top_dest_app_groups, 
                "Top 10 Destination App Groups", 
                "App Group", 
                "Number of Connections"
            )
            report_generator.add_plotly_figure(dest_chart)
        else:
            print("No destination app groups data available for graph generation.")

        top_ports = traffic_processor.get_top_ports(10)
        if not top_ports.empty:
            top_ports_graph = graph_generator.generate_horizontal_bar_chart(
                top_ports,
                "Top 20 Ports",
                "top_ports.png",
                "Number of Connections",
                "Port"
            )
            if top_ports_graph:
                report_generator.add_graph(top_ports_graph)
        else:
            print("No top ports data available for graph generation.")

        protocol_summary = traffic_processor.get_protocol_summary()
        if not protocol_summary.empty:
            protocol_summary_graph = graph_generator.generate_pie_chart(
                protocol_summary,
                "Protocol Distribution",
                "protocol_distribution.png"
            )
            if protocol_summary_graph:
                report_generator.add_graph(protocol_summary_graph)
        else:
            print("No protocol summary data available for graph generation.")

        # Add process traffic summary
        report_generator.add_section("Process Traffic Summary")
        report_generator.add_explanation(
            "This section provides an overview of network traffic by process name.",
            icon_path="examples/info_icon.png",
            icon_position='left'
        )

        process_summary = traffic_processor.create_process_summary()
        report_generator.add_markdown(process_summary)

        # Generate and add bar chart for top processes
        top_processes = traffic_processor.get_top_processes()
        if top_processes:
            process_chart = graph_generator.generate_bar_chart(
                top_processes,
                "Top 10 Processes by Traffic Volume",
                "Process Name",
                "Number of Connections"
            )
            report_generator.add_plotly_figure(process_chart)
        else:
            print("No process traffic data available for graph generation.")

        # Generate traffic summary for AI advisor
        traffic_summary = traffic_processor.summarize_traffic_for_ai_advisor()

        # Choose your model
        model = AnthropicModel(api_key=os.getenv('ANTHROPIC_API_KEY'), model="claude-3-5-sonnet-20240620")
        # model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini")
        # Or use one of these:
        # model = OllamaModel(model="llama3.1:8b")

        traffic_ai_advisor = TrafficAIAdvisor(model)
        traffic_graph_advisor = TrafficGraphAdvisor(model)
        attack_advisor = ATTACKAdvisor(model)  # Initialize the ATTACKAdvisor

        # Get AI advice for traffic
        traffic_ai_advice = traffic_ai_advisor.get_traffic_advice(traffic_summary)

        # Add AI-generated advice to the report
        report_generator.add_section("AI-Generated Traffic Analysis")
        report_generator.add_explanation(
            "This section provides AI-generated analysis and recommendations based on the network traffic data.",
            icon_path="examples/info_icon.png",
            icon_position='left'
        )
        report_generator.add_markdown(traffic_ai_advice)

       # Generate traffic graph
        try:
            graph_image_path = os.path.abspath("traffic_graph.png")
            graph_image_path = traffic_graph_advisor.generate_mermaid_image(traffic_summary, graph_image_path)
            if graph_image_path:
                report_generator.add_section("ZTS Advisor findings")
                report_generator.add_explanation(
                    """
                    This graph visualizes the network connections, potential issues, and recommendations.
                    The graph is generated using a large language model and using traffic information for augmentend generation. 
                    """,
                    icon_path="examples/info_icon.png",
                    icon_position='left'
                )
                report_generator.add_image(graph_image_path, max_width=6.5*inch)  # Adjust max_width as needed
            else:
                print("Traffic graph generation failed. Skipping graph section.")
        except Exception as e:
            print(f"Error generating or adding traffic graph: {e}")
            print("Exception details:")
            import traceback
            traceback.print_exc()
            print("Skipping graph section.")

        # Generate cross-environment traffic data
        cross_env_traffic = traffic_processor.get_cross_environment_traffic()
        
        # Create treemap for cross-environment traffic
        cross_env_treemap = graph_generator.generate_cross_environment_treemap(cross_env_traffic, width_cm=20, height_cm=10)
        
        if cross_env_treemap:
            report_generator.add_section("Cross-Environment Traffic")
            report_generator.add_explanation(
                """
                Cross environment traffic can be an indicator of misconfiguration or malicious activity and should be monitored.
                Auditors and security teams should review this traffic to ensure that it is compliant and secure.

                The below treemap visualizes the traffic flow between different environments.
                The size and color of each box represent the number of connections between environments.""",
                icon_path="examples/info_icon.png",
                icon_position='left'
            )
            try:
                treemap_image_path = os.path.abspath("cross_env_treemap.png")
                graph_generator.save_plotly_figure(cross_env_treemap, treemap_image_path)
                report_generator.add_image(treemap_image_path, max_width=6.5*inch)
            except Exception as e:
                print(f"Error adding cross-environment treemap: {e}")
                report_generator.add_paragraph("Error: Unable to add cross-environment traffic visualization.")
        else:
            print("No cross-environment traffic data available for visualization.")

        # Add ATT&CK framework analysis
        report_generator.add_section("MITRE ATT&CK Analysis")
        report_generator.add_explanation(
            "This section provides an analysis of potential tactics, techniques, and procedures (TTPs) "
            "based on the MITRE ATT&CK framework, given the observed network traffic patterns.",
            icon_path="examples/info_icon.png",
            icon_position='left'
        )

        # Get ATT&CK recommendations
        attack_recommendations = attack_advisor.get_attack_recommendations(traffic_summary)
        report_generator.add_markdown(attack_recommendations)

    # Generate the report
    report_generator.save(output_file)

if __name__ == "__main__":
    output_file = "test_traffic_report.pdf"
    generate_test_traffic_report(output_file)
    print(f"Test traffic report generated: {output_file}")
