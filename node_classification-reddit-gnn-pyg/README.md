# Reddit Node Classification with Graph Neural Networks

This project investigates node classification using graph neural networks on the Reddit2 dataset from PyTorch Geometric (PyG). The study compares different neural network architectures and evaluates their performance on classifying Reddit posts into their respective communities.

## Research Questions

The study aims to answer several key research questions:

1. How do traditional Feed-forward Neural Networks perform on node classification tasks with large datasets?
2. Can Graph Convolutional Neural Networks outperform Feed-forward Neural Networks, and at what computational cost?
3. How does Graph SAGE Neural Network performance compare to previous models?
4. What advantages do Graph Attention Neural Networks offer in this context?

## Framework Overview

### PyTorch
- Open-source machine learning library by Facebook's AI Research lab (FAIR)
- Designed for deep learning and tensor computations
- Features:
  - GPU acceleration support
  - Dynamic computation graphs
  - Pythonic API for intuitive model development
  - Rich ecosystem of tools and libraries
  - More flexible than alternatives like TensorFlow

### PyTorch Geometric (PyG)
- Extension library for PyTorch
- Specialized for deep learning on graphs and irregular structures
- Provides tools for:
  - Graph Neural Networks (GNNs) implementation
  - Node classification
  - Link prediction
  - Graph classification

## Dataset Overview

The Reddit2 dataset from PyG consists of:

### Graph Structure
- 232,965 nodes (Reddit posts)
- 23,213,838 edges (post similarities)
- 602 features per node (post content embeddings)
- 41 classes (subreddit communities)
- Average node degree: 99.65
- Undirected graph
- No self-loops
- Some isolated nodes

### Data Splits
- Training: 153,932 nodes (66%)
- Validation: 23,699 nodes (10%)
- Test: 55,334 nodes (24%)

### Dataset Characteristics
Based on analysis of a 50,000-node sample:
- Average degree: 22.31
- Graph density: 0.00044626 (very sparse)
- 2041 connected components
- Demonstrates power law degree distribution
- Clear community structure visible in network visualization

## Models Evaluated

### 1. Feed-forward Neural Network (FeedFwdNN)
- Traditional neural network architecture
- Characteristics:
  - Unidirectional information flow
  - No cycles or loops
  - Input as feature vectors
- Limitations:
  - Cannot utilize graph structure
  - Only trained on node features
  - Poor performance on graph-structured data
- Implementation:
  - Uses PyTorch's nn.Module
  - Configurable through nn.Sequential or nn.ModuleList

### 2. Graph Convolutional Neural Network (GraphConvNN)
- Adapts convolutions to graph data
- Key features:
  - Generalizes traditional CNNs to graphs
  - Learns node representations using both features and structure
  - Aggregates neighbor features
  - Uses graph convolution operations
- Implementation:
  - Uses PyG's GCNConv class
  - Stacks multiple graph convolutional layers
  - Includes ReLU activation functions
- Best suited for:
  - Graphs with meaningful local structure
  - Data exhibiting homophily

### 3. Graph SAGE Neural Network (GraphSageNN)
- Designed for large-scale graphs
- Architecture:
  - Inductive learning approach
  - Neighborhood sampling mechanism
  - Multiple Graph SAGE layers
- Key operations:
  - Sampling: Fixed-size neighborhood sampling
  - Aggregating: Feature aggregation from sampled neighbors
  - Updating: Combination of aggregated and node features
- Best configuration:
  - Hidden layers: [1024, 512, 256]
  - Dropout: 0.50
  - Learning rate: 1e-4
  - Weight decay: 5e-4
- Performance:
  - Final validation error: 0.1509
  - Test accuracy/F1/recall/precision: 0.8598
  - Smooth convergence pattern
  - Minimal fluctuations

### 4. Graph Attention Neural Network (GraphAttNN)
- Utilizes attention mechanisms
- Architecture:
  - Self-attention for neighbor importance
  - Multiple attention heads
  - Flexible weight assignment
- Best configuration:
  - Hidden layer: [128]
  - Dropout: 0.25
  - Attention heads: 4
  - Learning rate: 1e-2
  - Weight decay: 5e-4
- Performance:
  - Final validation error: 0.2291
  - Test accuracy/F1/recall/precision: 0.7660
  - Resource intensive but powerful
  - Some convergence fluctuations

## Implementation Details

### Data Loading
- Uses PyG's NeighborLoader
- Sampling configuration: [20, 15, 10]
  - 20 neighbors in 1st hop
  - 15 neighbors in 2nd hop
  - 10 neighbors in 3rd hop
- Batch sizes match respective set sizes
- Full epoch per iteration

### Hardware Setup
- M1 Max MacBook Pro
  - 64GB RAM
  - 10-core CPU
- Docker container configuration:
  - 50GB shared memory
  - 50GB RAM
  - 8 CPU cores
- Development environment:
  - VS Code IDE
  - Custom Docker image based on jupyter/datascience-notebook
  - PyTorch and PyG libraries

## Experimental Results

### Phase I: Dataset Exploration
- Comprehensive analysis of graph properties
- Visualization of node distributions
- Community structure analysis
- Degree distribution studies
- T-SNE projections of node features

### Phase II: Model Selection
- Multiple experiments with varying parameters:
  - Learning rates: [1e-2, 1e-3, 1e-4, 1e-5]
  - Optimizer algorithms: [adam, sgd]
  - Weight decay: [0, 5e-4]
  - Dropout probabilities: [0, 0.25, 0.5]
  - Hidden layer dimensions: [[128], [1024, 512, 256]]
  - Training epochs: [100, 250, 500, 1000]

### Phase III: Final Model Training
- Extended training of top performers
- Detailed metric collection
- Visualization of model progression
- Comprehensive evaluation on test set

## Key Findings

### Model Performance
- GraphSageNN achieved best overall performance
  - Superior validation error (0.1509)
  - Excellent test metrics (0.8598)
  - Smooth convergence characteristics
- GraphAttNN showed impressive efficiency
  - Strong performance with minimal architecture
  - Resource intensive but effective
  - Validation error of 0.2291 with single hidden layer

### Architectural Insights
- Traditional FeedFwdNN proved inadequate for graph data
- GraphConvNN showed moderate performance
- Network depth and width significantly impact performance
- Resource requirements vary dramatically between models

## Conclusions

The study demonstrates the superiority of graph-based neural networks for node classification tasks on the Reddit2 dataset. Key takeaways:

1. GraphSageNN provides the best balance of performance and stability
2. GraphAttNN offers impressive results with minimal architecture
3. Model choice depends on computational resources and accuracy requirements
4. Graph structure information is crucial for effective classification

The results suggest that both GraphSageNN and GraphAttNN are viable choices for large-scale node classification tasks, with the final selection depending on specific use case requirements and available computational resources.

## References

1. PyTorch Documentation: https://pytorch.org
2. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io
3. Papers:
   - Graph Convolutional Networks: Kipf & Welling (2017)
   - GraphSAGE: Hamilton et al. (2017)
   - Graph Attention Networks: Veličković et al. (2018)
