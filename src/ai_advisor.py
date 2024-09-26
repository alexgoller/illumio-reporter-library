import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
import json
from typing import Dict, Any
import cairosvg
import tempfile
import shutil

class AIAdvisor:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def get_security_advice(self, report_data: Dict[str, Any]) -> str:
        prompt = self._generate_prompt(report_data)
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=8192,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
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
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)

    def get_traffic_advice(self, traffic_summary):
        print(traffic_summary)
        prompt = f"""{HUMAN_PROMPT}This is traffic information from my network.
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

{AI_PROMPT}"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=8192,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

class TrafficGraphAdvisor:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)

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

Respond only with the Mermaid graph code, without any surrounding backticks or explanations. Start directly with 'graph TD' or 'flowchart TD'.


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

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        try:
            mermaid_code = response.content[0].text.strip()
            mermaid_code = mermaid_code.replace("```mermaid", "").replace("```", "").strip()
            print("Generated Mermaid code:")
            print(mermaid_code)
            return mermaid_code
        except Exception as e:
            print(f"Error generating Mermaid graph: {str(e)}")
            return None

    def generate_mermaid_image(self, traffic_summary, output_path):
        mermaid_code = self.generate_traffic_graph(traffic_summary)
        if mermaid_code:
            try:
                # Create a temporary directory
                temp_dir = tempfile.mkdtemp()
                temp_mmd_path = os.path.join(temp_dir, 'graph.mmd')
                temp_png_path = os.path.join(temp_dir, 'graph.png')

                # Write Mermaid code to temporary file
                with open(temp_mmd_path, 'w') as temp_mmd:
                    temp_mmd.write(mermaid_code)

                # Use mermaid-cli to convert Mermaid to PNG
                os.system(f'mmdc -i {temp_mmd_path} -o {temp_png_path} -w 2048 -H 1024')

                # Copy the generated PNG to the output path
                shutil.copy(temp_png_path, output_path)

                print(f"Mermaid graph saved to {output_path}")
                return output_path
            except Exception as e:
                print(f"Error generating Mermaid image: {str(e)}")
                return None
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print("Failed to generate Mermaid graph")
            return None