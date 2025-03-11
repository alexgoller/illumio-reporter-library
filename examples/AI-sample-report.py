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
    model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o")
    # or
    # model = OllamaModel(model="llama2")
    
    # Initialize AI Advisor with chosen model
    ai_advisor = AIAdvisor(model)

    # Fetch workload data
    workloads = data_fetcher.fetch_workload_data()
    detailed_workloads = data_fetcher.fetch_workload_services()
    # Initialize processors
    workload_processor = WorkloadProcessor(workloads)
    service_processor = WorkloadServiceProcessor(detailed_workloads)

    # Generate OS summary graph
    os_summary_data = workload_processor.get_os_summary_for_graph()

    # Generate enforcement mode summary graph
    enforcement_mode_data = workload_processor.get_enforcement_mode_for_graph()

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


    # Initialize GraphGenerator
    graph_generator = GraphGenerator()

    report_generator = ReportGenerator(
        output_file="sample_reports/AI-sample-report.pdf",
        color_scheme=illumio_color_scheme(),
        header_text="AI Security Report",
        footer_text="Generated March 4, 2025",
        logo_path="examples/illumio-logo.png"
    )

    # Create report
    report_generator.add_title("AI Security Findings")
    
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

    open_ports_data = service_processor.get_open_ports_summary()
    enforcement_summary = workload_processor.get_enforcement_mode_summary().reset_index()

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

    # Save report
    report_generator.save("sample_reports/AI-sample-report.pdf")

    print("AI-sample-report.pdf generated")

if __name__ == "__main__":
    main()