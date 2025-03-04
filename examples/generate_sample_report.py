from src.data_fetcher import DataFetcher
from src.data_processors import WorkloadProcessor, WorkloadServiceProcessor, TrafficProcessor
from src.report_generator import ReportGenerator
from src.graph_generator import GraphGenerator
from src.ai_advisor import AIAdvisor, TrafficAIAdvisor, TrafficGraphAdvisor  # Import the new AIAdvisor and TrafficAIAdvisor classes
import pandas as pd
from reportlab.lib import colors
import os
from reportlab.lib.enums import TA_LEFT
from src.ai_advisor import AnthropicModel, OpenAIModel, OllamaModel

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

def main():
    # Initialize DataFetcher
    data_fetcher = DataFetcher()
    
    # Initialize AI Model (choose one)
    # model = AnthropicModel(api_key=os.getenv('ANTHROPIC_API_KEY'), model="claude-3-5-sonnet-20240620")
    # or
    model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4-turbo-preview")
    # or
    # model = OllamaModel(model="llama2")
    
    # Initialize AI Advisor with chosen model
    ai_advisor = AIAdvisor(model)

    # Fetch workload data
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

    # Generate report with custom color scheme, header, footer, and logo
    report_generator = ReportGenerator(
        output_file="workload_service_report.pdf",
        color_scheme=illumio_color_scheme(),
        header_text="Confidential - Internal Use Only",
        footer_text="Generated on 2023-04-15",
        logo_path="examples/illumio-logo.png"
    )

    # Create report
    report_generator.add_title("Illumio Workload, Service, and Traffic Report")
    
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
        traffic_summary = "No traffic data available."
    else:
        # Initialize traffic processor
        traffic_processor = TrafficProcessor(traffic_flows, label_href_map, value_href_map)

        # Generate traffic summary for AI advisor
        traffic_summary = traffic_processor.summarize_traffic_for_ai_advisor()

        # Initialize TrafficAIAdvisor and TrafficGraphAdvisor
        traffic_ai_advisor = TrafficAIAdvisor(model)
        traffic_graph_advisor = TrafficGraphAdvisor(model)

        # Get AI advice for traffic
        traffic_ai_advice = traffic_ai_advisor.get_traffic_advice(traffic_summary)

        # Generate traffic graph
        try:
            traffic_graph = traffic_graph_advisor.generate_traffic_graph(traffic_summary)
            if traffic_graph is not None:
                report_generator.add_section("Traffic Graph")
                report_generator.add_explanation(
                    "This graph visualizes the network connections, potential issues, and recommendations.",
                    icon_path="examples/info_icon.png",
                    icon_position='left'
                )
                report_generator.add_plotly_figure(traffic_graph)
            else:
                print("Traffic graph generation returned None. Skipping graph section.")
                print("Traffic summary:")
                print(traffic_summary)
        except Exception as e:
            print(f"Error generating or adding traffic graph: {e}")
            print("Traffic summary:")
            print(traffic_summary)
            print("Skipping graph section.")

    ai_input_data = {
        'workload_summary': workload_processor.get_os_summary().to_string(index=False),
        'open_ports_summary': open_ports_data.head(20).to_string(index=False),
        'enforcement_summary': enforcement_summary.to_string(index=False),
        'traffic_summary': traffic_summary
    }
    
    # Get AI advice
    ai_advice = ai_advisor.get_security_advice(ai_input_data)

    # Add AI advice to the report
    report_generator.add_section("AI Advisor Output")
    report_generator.add_explanation(
        "This section provides AI-generated security and microsegmentation recommendations based on the analyzed data.",
        icon_path="examples/info_icon.png",
        icon_position='left'
    )
    report_generator.add_markdown(ai_advice)

    if not traffic_flows:
        print("No traffic data available. Skipping traffic section.")
    else:
        # Initialize traffic processor
        traffic_processor = TrafficProcessor(traffic_flows, label_href_map, value_href_map)

        # Generate traffic summary
        report_generator.add_section("Traffic Summary")
        report_generator.add_explanation(
            "This section provides an overview of network traffic in the environment.",
            icon_path="examples/info_icon.png",
            icon_position='left'
        )

        try:
            top_src_ips = traffic_processor.get_top_src_ips()
            if not top_src_ips.empty:
                report_generator.add_table(top_src_ips)
            else:
                print("No top source IPs data available.")
        except Exception as e:
            print(f"Error getting top source IPs: {str(e)}")

        try:
            top_dst_ips = traffic_processor.get_top_dst_ips()
            if not top_dst_ips.empty:
                report_generator.add_table(top_dst_ips)
            else:
                print("No top destination IPs data available.")
        except Exception as e:
            print(f"Error getting top destination IPs: {str(e)}")

        try:
            top_ports = traffic_processor.get_top_ports(20)
            if not top_ports.empty:
                report_generator.add_table(top_ports)
            else:
                print("No top ports data available.")
        except Exception as e:
            print(f"Error getting top ports: {str(e)}")

        try:
            protocol_summary = traffic_processor.get_protocol_summary()
            if not protocol_summary.empty:
                report_generator.add_table(protocol_summary)
            else:
                print("No protocol summary data available.")
        except Exception as e:
            print(f"Error getting protocol summary: {str(e)}")

        # Generate traffic graphs
        top_ports = traffic_processor.get_top_ports(20)
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

    # Save report
    report_generator.save("workload_service_report.pdf")

    print("Workload and Service report generated: workload_service_report.pdf")

if __name__ == "__main__":
    main()