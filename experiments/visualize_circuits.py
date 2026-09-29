import argparse
import pickle
import networkx as nx
import matplotlib.pyplot as plt

def load_edges(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def format_node(node_name, node_idx):
    if isinstance(node_idx, slice):
        idx_str = "[:]"
    elif isinstance(node_idx, tuple) and len(node_idx) >= 3:
        idx_str = f"[:, :, {node_idx[2]}]"
    else:
        idx_str = str(node_idx)
    return f"{node_name}\n{idx_str}"

def visualize_circuits(fp16_path, quant_path, output_path):
    fp16_edges = load_edges(fp16_path)
    quant_edges = load_edges(quant_path)
    
    # ACDC edges are typically saved as (child_name, child_index, parent_name, parent_index)
    # The actual causal direction is parent -> child, so we draw from parent to child.
    
    G = nx.DiGraph()
    
    all_edges = fp16_edges.union(quant_edges)
    
    edge_colors = []
    
    for edge in all_edges:
        child_name, child_idx, parent_name, parent_idx = edge
        
        u = format_node(parent_name, parent_idx)
        v = format_node(child_name, child_idx)
        
        G.add_edge(u, v)
        
        if edge in fp16_edges and edge in quant_edges:
            G[u][v]['color'] = 'green' # Shared
            G[u][v]['label'] = 'Shared'
        elif edge in fp16_edges:
            G[u][v]['color'] = 'blue' # FP16 only
            G[u][v]['label'] = 'FP16 Only'
        else:
            G[u][v]['color'] = 'red' # Quant only
            G[u][v]['label'] = 'Quant Only'

    colors = [G[u][v]['color'] for u, v in G.edges()]
    
    plt.figure(figsize=(24, 16))
    
    # Try using graphviz layout (dot) for hierarchical structures if pygraphviz is installed
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception as e:
        print("Graphviz layout failed, using spring layout.", e)
        pos = nx.spring_layout(G, k=0.5, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightgray', alpha=0.9, edgecolors='black')
    
    nx.draw_networkx_edges(G, pos, edge_color=colors, arrowstyle='-|>', arrowsize=20, width=2.0)
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", font_weight="bold")
    
    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Shared Edges'),
        Line2D([0], [0], color='blue', lw=4, label='FP16 Only'),
        Line2D([0], [0], color='red', lw=4, label='Quant Only')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize='x-large')
    
    plt.title("Causal Circuit Differences (FP16 vs Quantized)", fontsize=24)
    plt.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-edges", type=str, required=True, help="Path to fp16 edge pickle")
    parser.add_argument("--quant-edges", type=str, required=True, help="Path to quant edge pickle")
    parser.add_argument("--output", type=str, default="circuit_overlap.png", help="Path to save the visualization")
    
    args = parser.parse_args()
    visualize_circuits(args.fp16_edges, args.quant_edges, args.output)
