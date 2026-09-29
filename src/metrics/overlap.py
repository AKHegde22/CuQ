def compute_jaccard_similarity(edges_a, edges_b):
    """
    Computes Jaccard similarity between two sets of edges.
    
    Args:
        edges_a (set): Set of edges in circuit A.
        edges_b (set): Set of edges in circuit B.
        
    Returns:
        float: Jaccard similarity.
    """
    intersection = edges_a.intersection(edges_b)
    union = edges_a.union(edges_b)
    if not union:
        return 1.0
    return len(intersection) / len(union)

def compute_precision_recall(edges_gt, edges_pred):
    """
    Computes precision and recall treating edges_gt as ground truth.
    
    Args:
        edges_gt (set): Set of edges in the ground truth circuit (e.g. fp16).
        edges_pred (set): Set of edges in the predicted circuit (e.g. quantized).
        
    Returns:
        tuple: (precision, recall)
    """
    intersection = edges_gt.intersection(edges_pred)
    precision = len(intersection) / len(edges_pred) if edges_pred else 1.0
    recall = len(intersection) / len(edges_gt) if edges_gt else 1.0
    return precision, recall
