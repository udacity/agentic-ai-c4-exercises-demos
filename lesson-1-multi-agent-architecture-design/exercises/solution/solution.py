import matplotlib.pyplot as plt
import networkx as nx
import numpy as np




def create_diagram(title, nodes, edges, node_labels=None, node_types=None, edge_labels=None):
    graph = nx.DiGraph()
    
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    
    if node_labels is None:
        node_labels = {node: node for node in nodes}
    if node_types is None:
        node_types = {node: 'agent' for node in nodes}
    if edge_labels is None:
        edge_labels = {}

    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    pos = {}
    horizontal_spacing = 1.0
    vertical_spacing = 0.7
    
    pos["Visitor Input"] = (0, 0)
    pos["Language Identification"] = (horizontal_spacing, 0)
    pos["Arrernte Language Specialist"] = (2*horizontal_spacing, vertical_spacing)
    pos["Pitjantjatjara Language Specialist"] = (2*horizontal_spacing, -vertical_spacing)
    pos["Knowledge Base Lookup"] = (3*horizontal_spacing, 0)
    
    node_colors = []
    node_shapes = []
    node_sizes = []
    
    for node in nodes:
        node_type = node_types.get(node, 'agent')
        
        if node_type == 'agent':
            node_colors.append("#6495ED")
            node_shapes.append('o')
            node_sizes.append(2800)
        elif node_type == 'tool':
            node_colors.append("#FFD700")
            node_shapes.append('s')
            node_sizes.append(2600)
        elif node_type == 'user':
            node_colors.append("#FF6347")
            node_shapes.append('d')
            node_sizes.append(2400)
        elif node_type == 'data':
            node_colors.append("#90EE90")
            node_shapes.append('h')
            node_sizes.append(2500)
        else:
            node_colors.append("#C0C0C0")
            node_shapes.append('p')
            node_sizes.append(2300)

    for i, node in enumerate(nodes):
        nx.draw_networkx_nodes(graph, pos, 
                             nodelist=[node],
                             node_color=[node_colors[i]], 
                             node_shape=node_shapes[i],
                             node_size=node_sizes[i],
                             edgecolors='black', 
                             linewidths=1.5, 
                             alpha=0.9)
    
    nx.draw_networkx_edges(graph, pos, 
                         edge_color="black", 
                         arrowsize=25,
                         width=2.0, 
                         alpha=0.9, 
                         arrowstyle='-|>', 
                         connectionstyle="arc3,rad=0.1")

    for node, (x, y) in pos.items():
        plt.text(x, y, node_labels[node], 
                fontsize=11, 
                ha='center', 
                va='center',
                fontweight='bold',
                bbox=dict(facecolor='white', 
                         alpha=0.8, 
                         edgecolor='lightgray', 
                         boxstyle='round,pad=0.5'))
                         
    if edge_labels:
        for edge, label in edge_labels.items():
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2 + 0.15
            plt.text(x, y, label, 
                    fontsize=12, 
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='darkblue',
                    bbox=dict(facecolor='white', 
                             alpha=0.9,
                             edgecolor='blue',
                             boxstyle='round,pad=0.3'))

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#6495ED", markersize=15, label='Agent'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#FFD700", markersize=15, label='Tool/Resource'),
        plt.Line2D([0], [0], marker='d', color='w', markerfacecolor="#FF6347", markersize=15, label='User Interface'),
        plt.Line2D([0], [0], marker='h', color='w', markerfacecolor="#90EE90", markersize=15, label='Data Component')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


""" 
Suggested Extensions, Pick One or Mix
1. Add a "Translation Verification Agent"
This agent checks if the translation preserves meaning/cultural context.

Comes after the language specialist and before the response goes to the user.

2. Add a "Multimodal Media Tool"
A tool that retrieves images/audio relevant to the query (e.g., a sound clip of a traditional song or image of a local artifact).

Connects to both the language specialist and the knowledge base.

3. Add a "Feedback Collector"
Captures user feedback after receiving information.

Helps improve future responses and tracks which areas need more detail or clarification.

4. Add a "Cultural Sensitivity Checker" Tool
Ensures that knowledge being returned is appropriate to share (some Indigenous knowledge is restricted or sacred).


"""

def extended_uluru_solution():
    nodes = [
        # Original nodes
        "Visitor Input",
        "Language Identification",
        "Arrernte Language Specialist",
        "Pitjantjatjara Language Specialist",
        "Knowledge Base Lookup",

        # New nodes
        "Translation Verification Agent",
        "Multimodal Media Tool",
        "Feedback Collector",
        "Cultural Sensitivity Checker"
    ]

    edges = [
        # Original edges
        ("Visitor Input", "Language Identification"),
        ("Language Identification", "Arrernte Language Specialist"),
        ("Language Identification", "Pitjantjatjara Language Specialist"),
        ("Arrernte Language Specialist", "Knowledge Base Lookup"),
        ("Pitjantjatjara Language Specialist", "Knowledge Base Lookup"),
        ("Knowledge Base Lookup", "Arrernte Language Specialist"),
        ("Knowledge Base Lookup", "Pitjantjatjara Language Specialist"),
        ("Arrernte Language Specialist", "Language Identification"),
        ("Pitjantjatjara Language Specialist", "Language Identification"),
        ("Language Identification", "Visitor Input"),

        # New edges
        ("Arrernte Language Specialist", "Translation Verification Agent"),
        ("Pitjantjatjara Language Specialist", "Translation Verification Agent"),
        ("Translation Verification Agent", "Language Identification"),

        ("Knowledge Base Lookup", "Multimodal Media Tool"),
        ("Multimodal Media Tool", "Arrernte Language Specialist"),
        ("Multimodal Media Tool", "Pitjantjatjara Language Specialist"),

        ("Knowledge Base Lookup", "Cultural Sensitivity Checker"),
        ("Cultural Sensitivity Checker", "Arrernte Language Specialist"),
        ("Cultural Sensitivity Checker", "Pitjantjatjara Language Specialist"),

        ("Visitor Input", "Feedback Collector")
    ]

    node_types = {
        "Visitor Input": "user",
        "Language Identification": "tool",
        "Arrernte Language Specialist": "agent",
        "Pitjantjatjara Language Specialist": "agent",
        "Knowledge Base Lookup": "tool",
        "Translation Verification Agent": "agent",
        "Multimodal Media Tool": "tool",
        "Feedback Collector": "tool",
        "Cultural Sensitivity Checker": "tool"
    }

    edge_labels = {
        ("Language Identification", "Visitor Input"): "Final Response",
        ("Visitor Input", "Feedback Collector"): "User Feedback",
        ("Translation Verification Agent", "Language Identification"): "Verified Translation",
        ("Multimodal Media Tool", "Arrernte Language Specialist"): "Media Snippets",
        ("Multimodal Media Tool", "Pitjantjatjara Language Specialist"): "Media Snippets",
        ("Cultural Sensitivity Checker", "Arrernte Language Specialist"): "Filtered Info",
        ("Cultural Sensitivity Checker", "Pitjantjatjara Language Specialist"): "Filtered Info"
    }

    create_diagram(
        "Uluru Cultural Center: Extended Multi-Agent System",
        nodes,
        edges,
        None,
        node_types,
        edge_labels
    )

if __name__ == "__main__":
    extended_uluru_solution()