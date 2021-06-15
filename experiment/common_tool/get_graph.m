function L = get_graph(X, k)

manifold.k = k;
manifold.Metric = 'Euclidean';
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'Binary';
manifold.bNormalizeGraph = 0;
L = full(laplacian(X, manifold));
