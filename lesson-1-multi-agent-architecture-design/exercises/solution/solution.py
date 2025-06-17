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

    plt.figure(figsize=(14, 10))
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    pos = {}
    horizontal_spacing = 1.5
    vertical_spacing = 1.0
    
    # Define positions in a more balanced layout
    pos["Visitor Input"] = (0, 0)
    pos["Language Identification"] = (2*horizontal_spacing, 0)
    pos["Arrernte Language Specialist"] = (4*horizontal_spacing, 2*vertical_spacing)
    pos["Pitjantjatjara Language Specialist"] = (4*horizontal_spacing, -2*vertical_spacing)
    pos["Knowledge Base Lookup"] = (6*horizontal_spacing, 0)
    
    # Define positions for extended nodes in a balanced layout
    if "Translation Verification Agent" in nodes:
        pos["Translation Verification Agent"] = (3*horizontal_spacing, 3*vertical_spacing)
    if "Cultural Sensitivity Checker" in nodes:
        pos["Cultural Sensitivity Checker"] = (6*horizontal_spacing, 3*vertical_spacing)
    if "Multimodal Media Tool" in nodes:
        pos["Multimodal Media Tool"] = (7*horizontal_spacing, -2*vertical_spacing)
    if "Feedback Collector" in nodes:
        pos["Feedback Collector"] = (0, -2.5*vertical_spacing)
    
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

    # Draw nodes first
    for i, node in enumerate(nodes):
        nx.draw_networkx_nodes(graph, pos, 
                             nodelist=[node],
                             node_color=[node_colors[i]], 
                             node_shape=node_shapes[i],
                             node_size=node_sizes[i],
                             edgecolors='black', 
                             linewidths=1.5, 
                             alpha=0.9)
    
    # Draw text labels on nodes
    for node, (x, y) in pos.items():
        plt.text(x, y, node_labels[node], 
                fontsize=11, 
                ha='center', 
                va='center',
                fontweight='bold',
                bbox=dict(facecolor='white', 
                         alpha=0.8, 
                         edgecolor='lightgray', 
                         boxstyle='round,pad=0.5'),
                zorder=5)
    
    # Draw edges on top using matplotlib patches for proper layering
    from matplotlib.patches import FancyArrowPatch
    
    for edge in edges:
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]
        
        # Calculate the direction vector
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Normalize the direction vector
        if distance > 0:
            dx_norm = dx / distance
            dy_norm = dy / distance
        else:
            dx_norm = dy_norm = 0
        
        # Simple offset from box edges
        offset = 0.4
        
        # Calculate start and end points outside the boxes
        start_adjusted = (start_pos[0] + offset * dx_norm, start_pos[1] + offset * dy_norm)
        end_adjusted = (end_pos[0] - offset * dx_norm, end_pos[1] - offset * dy_norm)
        
        # Determine curve direction to ensure concave shape (curve toward center of diagram)
        # Calculate center of diagram
        center_x = sum(pos[node][0] for node in pos) / len(pos)
        center_y = sum(pos[node][1] for node in pos) / len(pos)
        
        # Calculate midpoint of arrow
        mid_x = (start_adjusted[0] + end_adjusted[0]) / 2
        mid_y = (start_adjusted[1] + end_adjusted[1]) / 2
        
        # Determine if midpoint is above/below or left/right of center to pick curve direction
        if abs(mid_x - center_x) > abs(mid_y - center_y):
            # Horizontal dominant - curve toward vertical center
            curve_rad = 0.15 if mid_y > center_y else -0.15
        else:
            # Vertical dominant - curve toward horizontal center  
            curve_rad = 0.15 if mid_x > center_x else -0.15
        
        arrow = FancyArrowPatch(start_adjusted, end_adjusted,
                               connectionstyle=f"arc3,rad={curve_rad}",
                               arrowstyle='-|>',
                               mutation_scale=20,
                               linewidth=2.0,
                               color='black',
                               alpha=0.9,
                               zorder=10)
        plt.gca().add_patch(arrow)
                         
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

    create_diagram(
        "Uluru Cultural Center: Extended Multi-Agent System",
        nodes,
        edges,
        None,
        node_types,
    )

if __name__ == "__main__":
    extended_uluru_solution()