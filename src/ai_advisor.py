import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from .ai_models import AnthropicModel, BaseAIModel, OllamaModel
import os
import json
from typing import Dict, Any
import cairosvg
import tempfile
import shutil
import re

from .ai_models.base_model import BaseAIModel
from .ai_models.anthropic_model import AnthropicModel
from .ai_models.ollama_model import OllamaModel
from .ai_models.openai_model import OpenAIModel

class AIAdvisor:
    def __init__(self, model: BaseAIModel):
        self.model = model

    def get_security_advice(self, report_data: Dict[str, Any]) -> str:
        prompt = self._generate_prompt(report_data)
        try:
            return self.model.generate_response(prompt)
        except Exception as e:
            print(f"Error calling AI model: {str(e)}")
            return "Unable to generate AI advice due to an error."

    def _generate_prompt(self, report_data: Dict[str, Any]) -> str:
        return f"""Based on the following network data, provide security and microsegmentation recommendations:

Workload Summary:
{report_data['workload_summary']}

Top Open Ports:
{report_data['open_ports_summary']}

Enforcement Mode Summary:
{report_data['enforcement_summary']}

Please analyze this data and provide:
1. Potential security risks
2. Microsegmentation recommendations
3. Best practices for improving overall network security
4. Recommendations for optimizing enforcement modes
5. Recommendations for traffic analysis and security
Break the response into sections with clear headings.

Return markdown formatted text."""

    def _generate_traffic_prompt(self, report_data: Dict[str, Any]) -> str:
        return f"""Based on the following traffic data, provide traffic analysis and security recommendations:

The traffic summary is formatted as a list of strings, each representing a connection with source and destination information, protocol, and port.
This is traffic information from my network. It references application groups like AD (prod) which means my production Active Directory. reading this output, can you give security recommendations or show issues with the traffic, anomalies, etc?

Traffic Summary:
{report_data['traffic_summary']}

Please analyze this data and provide:
1. Potential security risks
2. Microsegmentation recommendations
3. Best practices for improving overall network security
4. Recommendations for traffic analysis and security

Break the response into sections with clear headings.

Return markdown formatted text."""


class TrafficAIAdvisor:
    def __init__(self, model: BaseAIModel):
        self.model = model

    def get_traffic_advice(self, traffic_summary):
        prompt = f"""This is traffic information from my network.
        It references application groups like AD (prod) which means my production Active Directory. Reading this output, can you give security recommendations or show issues with the traffic, anomalies, etc?

        Output format is markdown. Please use markdown formatting and provide the sections with clear headings, then go into a list type style. 
        Prefix each section with a very short paragraph why the recommendation is here and what it is.

        Traffic Summary:
        {traffic_summary}

        Please provide:

        1. Security recommendations based on this traffic data
        2. Potential issues or anomalies in the traffic patterns
        3. Any insights on the network structure or application dependencies
        4. Suggestions for improving network segmentation or microsegmentation
        5. Suggestions on improving general cyber hygiene by looking at traffic
        flows that are _NOT_ existing today and making assumptions, that means
        backup should access everything, monitoring should, everyone should 
        use the identified NTP services.
        """

        response = self.model.generate_response(prompt)
        print(response)
        return response

class TrafficGraphAdvisor:
    def __init__(self, model: BaseAIModel):
        self.model = model

    def generate_traffic_graph(self, traffic_summary):
        prompt = f"""Based on the following traffic summary, create a Mermaid graph representation of the network traffic. The graph should include nodes (application groups) and edges (connections between groups).

        Traffic Summary:
        {traffic_summary}

        Please include:
        1. Nodes representing major application groups and their environments (e.g., AD Prod, Exchange Prod)
        2. Edges representing connections between these groups, including the type of traffic (e.g., high volume, direct access)
        3. Any potential security issues or anomalies (e.g., cross-environment access, potential lateral movement)
        4. highlight privileged access without jump hosts
        5. highlight other potential risks with the current traffic data

        Be sure to indicate potential risk and security issues by looking at the traffic summary, especially risks that can be mitigated by network policies and microsegmentation.
        Never use ( ) in the mermaid code.
        Respond only with the Mermaid graph code, without any surrounding backticks or explanations. Start directly with 'graph TD' or 'flowchart TD'.

Use [] for placing app group (VDI Users, AD Prod). No style or classes are allowed.
() are not allowed anywhere in the code. Create a concise graph showing the main traffic flows and potential security issues.
Arrows always go from left to right, arrows from right to left are not allowed.

graph TD
    A[VDI Users] -->|High volume| B[AD Prod]
    A -->|RDP| C[Exchange Prod]
    A -->|Direct access| D[Payment Prod]
    E[Laptop Users] -->|Direct access| D
    F[Monitoring Prod] -->|High volume| G[POS Staging]
    F -->|High volume| H[POS PCI]
    I[Ordering Prod] -->|Direct access| H
    J[Cart Prod] -->|Direct DB access| K[Shared DB Prod]
    L[Ordering Dev] -->|Cross-env DB access| M[Asset Mgmt Prod]
    A <-->|Potential lateral movement| E
        """

        response = self.model.generate_response(prompt)
        return response
        

    def clean_mermaid_graph(self, graph):
        # Remove any text before 'graph TD' or 'flowchart TD'
        graph = re.sub(r'^.*?(graph TD|flowchart TD)', r'\1', graph, flags=re.DOTALL)
        
        # Ensure the graph starts with 'graph TD' or 'flowchart TD'
        if not graph.strip().startswith(('graph TD', 'flowchart TD')):
            graph = 'graph TD\n' + graph
        
        # Remove any text after the last node or edge definition
        graph = re.sub(r'\n\s*\n.*$', '', graph, flags=re.DOTALL)
        
        # Clean up node names and connections
        lines = graph.split('\n')
        cleaned_lines = []
        for line in lines:
            if '-->' in line or '---' in line:
                parts = re.split(r'(-->|---)', line)
                cleaned_parts = []
                for part in parts:
                    cleaned_part = re.sub(r'[^\w\s\-\[\]]', '', part)
                    cleaned_parts.append(cleaned_part)
                cleaned_lines.append(''.join(cleaned_parts))
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def generate_mermaid_image(self, traffic_summary, output_path):
        mermaid_code = self.generate_traffic_graph(traffic_summary)
        
        # Print the Mermaid code for debugging
        print("Generated Mermaid Code:")
        print(mermaid_code)
        print("End of Mermaid Code")
        
        if mermaid_code:
            try:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.mmd') as temp_mmd:
                    temp_mmd.write(mermaid_code)
                    temp_mmd_path = temp_mmd.name

                os.system(f'mmdc -i {temp_mmd_path} -o {output_path} --width 2048')

                if os.path.exists(output_path):
                    print(f"Mermaid graph saved to {output_path}")
                    return output_path
                else:
                    print(f"Failed to generate Mermaid image: {output_path} not found")
                    return None
            except Exception as e:
                print(f"Error generating Mermaid image: {str(e)}")
                return None
            finally:
                if os.path.exists(temp_mmd_path):
                    os.unlink(temp_mmd_path)
        else:
            print("Failed to generate Mermaid graph")
            return None

class ATTACKAdvisor:
    def __init__(self, model: BaseAIModel):
        self.model = model

    def get_attack_recommendations(self, traffic_summary):
        prompt = f"""Analyze the following network traffic summary and provide recommendations based on the MITRE ATT&CK framework. Identify potential tactics, techniques, and procedures (TTPs) that an attacker might use given this network configuration.

        Traffic Summary:
        {traffic_summary}

        Please provide:

        1. Identified ATT&CK Tactics: List the relevant ATT&CK tactics that could be applicable based on the observed traffic patterns.

        2. Potential Techniques: For each identified tactic, list potential ATT&CK techniques that an attacker might employ, given the network configuration.

        3. Detection Strategies: Suggest detection strategies for each identified technique, focusing on how to monitor and alert on suspicious activities related to these techniques.

        4. Mitigation Recommendations: Provide specific mitigation recommendations for each identified technique, with a focus on how to use microsegmentation and network policies to reduce the attack surface.

        5. Overall Security Posture Improvement: Based on the ATT&CK analysis, suggest overall improvements to the security posture of the network, including any gaps in the current configuration that should be addressed.

        Format your response in markdown, with clear headings for each section. Use bullet points or numbered lists where appropriate for clarity.

        Remember to consider:
        - Potential lateral movement paths
        - Exposed services or ports that could be exploited
        - Cross-environment access that might indicate security boundaries being bypassed
        - Any anomalies in the traffic patterns that could indicate potential security risks
        - mention the traffic tuple where you see potential risk and annotate findings with the app and env tuple from the traffic summary

        Provide concise, actionable recommendations that align with ATT&CK framework best practices and focus on improving the network's security through effective microsegmentation and policy enforcement.
        """

        response = self.model.generate_response(prompt)
        return response