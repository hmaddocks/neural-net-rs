use ndarray::Array2;
use std::fmt;

/// A unique identifier for a node in the computation graph arena.
///
/// [`TensorId`]s are returned by [`Graph::leaf`] and every op method on [`Graph`].
/// They remain valid until [`Graph::clear_computation_graph`] truncates the arena past
/// their index. Parameter leaves should be created first so their ids stay stable across
/// training steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

/// Callback invoked during reverse-mode autodiff to propagate gradients to children.
pub(crate) type BackwardFn = Box<dyn Fn(&mut Graph, TensorId)>;

/// A node in the tensor computation graph.
///
/// Nodes are stored in a [`Graph`] arena. Leaf nodes hold parameters or inputs; op nodes
/// store their forward value, accumulated gradient, child links, and an optional backward
/// callback invoked by [`Graph::backward`]. Inspect values through [`Graph::data`],
/// [`Graph::grad`], and [`Graph::children`].
pub struct Node {
    /// Forward-pass value for this node.
    data: Array2<f64>,
    /// Accumulated gradient from reverse-mode autodiff.
    grad: Array2<f64>,
    /// Child nodes that contributed to this node's value.
    children: Vec<TensorId>,
    /// Optional backward rule for this node.
    backward: Option<BackwardFn>,
}

impl Node {
    fn leaf(data: Array2<f64>) -> Self {
        let grad = Array2::zeros(data.dim());
        Self {
            data,
            grad,
            children: Vec::new(),
            backward: None,
        }
    }

    fn with_children(
        data: Array2<f64>,
        children: Vec<TensorId>,
        backward: Option<BackwardFn>,
    ) -> Self {
        let grad = Array2::zeros(data.dim());
        Self {
            data,
            grad,
            children,
            backward,
        }
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("children", &self.children)
            .field("has_backward", &self.backward.is_some())
            .finish()
    }
}

/// Arena-backed storage for tensor computation graph nodes.
///
/// This mirrors the scalar `ValueArena` pattern from the GPT crate, but stores dense
/// [`Array2<f64>`] values and gradients instead of scalars.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    /// Creates an empty graph arena.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Creates a leaf node (typically a parameter or input) and returns its id.
    pub fn leaf(&mut self, data: Array2<f64>) -> TensorId {
        self.push_node(Node::leaf(data))
    }

    /// Creates an operation node with children and an optional backward rule.
    pub(crate) fn op(
        &mut self,
        data: Array2<f64>,
        children: Vec<TensorId>,
        backward: Option<BackwardFn>,
    ) -> TensorId {
        self.push_node(Node::with_children(data, children, backward))
    }

    fn push_node(&mut self, node: Node) -> TensorId {
        let id = TensorId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Returns a shared reference to the node at `id`.
    pub fn node(&self, id: TensorId) -> &Node {
        &self.nodes[id.0]
    }

    /// Returns a mutable reference to the node at `id`.
    pub fn node_mut(&mut self, id: TensorId) -> &mut Node {
        &mut self.nodes[id.0]
    }

    /// Returns the forward value of `id`.
    pub fn data(&self, id: TensorId) -> &Array2<f64> {
        &self.nodes[id.0].data
    }

    /// Returns the accumulated gradient of `id`.
    pub fn grad(&self, id: TensorId) -> &Array2<f64> {
        &self.nodes[id.0].grad
    }

    /// Returns the child ids of `id`.
    pub fn children(&self, id: TensorId) -> &[TensorId] {
        &self.nodes[id.0].children
    }

    /// Performs reverse-mode autodiff from `output`, seeding its gradient with `seed`.
    ///
    /// Use this when the upstream loss defines an output delta directly (as in
    /// neural-net-rs's `(targets - outputs)` convention) instead of deriving it from
    /// a scalar loss.
    pub fn backward_with_seed(&mut self, output: TensorId, seed: Array2<f64>) {
        assert_eq!(
            seed.raw_dim(),
            self.data(output).raw_dim(),
            "seed gradient shape must match output shape"
        );

        let mut topo = Vec::new();
        let mut visited = vec![false; self.nodes.len()];

        Self::build_topo(self, output, &mut visited, &mut topo);
        self.set_grad(output, seed);

        for &id in topo.iter().rev() {
            let backward = self.nodes[id.0].backward.take();
            if let Some(backward) = backward {
                backward(self, id);
                self.nodes[id.0].backward = Some(backward);
            }
        }
    }

    /// Performs reverse-mode autodiff from `output`.
    ///
    /// This builds a topological ordering of the graph, seeds the output gradient with
    /// ones, then walks nodes in reverse calling each stored backward rule.
    pub fn backward(&mut self, output: TensorId) {
        let seed = Array2::ones(self.data(output).raw_dim());
        self.backward_with_seed(output, seed);
    }

    fn build_topo(graph: &Graph, id: TensorId, visited: &mut [bool], topo: &mut Vec<TensorId>) {
        if visited[id.0] {
            return;
        }

        visited[id.0] = true;
        for &child in graph.children(id) {
            Self::build_topo(graph, child, visited, topo);
        }
        topo.push(id);
    }

    /// Sets the forward value of `id`.
    pub fn set_data(&mut self, id: TensorId, data: Array2<f64>) {
        self.nodes[id.0].data = data;
    }

    /// Sets the gradient of `id`.
    pub fn set_grad(&mut self, id: TensorId, grad: Array2<f64>) {
        self.nodes[id.0].grad = grad;
    }

    /// Adds `grad` into the accumulated gradient of `id`.
    pub fn add_grad(&mut self, id: TensorId, grad: &Array2<f64>) {
        self.nodes[id.0].grad += grad;
    }

    /// Zeros every node's gradient.
    pub fn zero_grad(&mut self) {
        self.nodes.iter_mut().for_each(|node| {
            node.grad.fill(0.0);
        });
    }

    /// Truncates the arena, keeping only the first `num_params_to_keep` nodes.
    ///
    /// This clears intermediate computation nodes while preserving parameter leaves,
    /// preventing unbounded memory growth during training.
    pub fn clear_computation_graph(&mut self, num_params_to_keep: usize) {
        self.nodes.truncate(num_params_to_keep);
    }

    /// Returns the number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::Graph;

    #[test]
    fn leaf_stores_data_and_zero_grad() {
        let mut graph = Graph::new();
        let id = graph.leaf(array![[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(graph.data(id).shape(), &[2, 2]);
        assert_relative_eq!(graph.data(id).sum(), 10.0);
        assert_relative_eq!(graph.grad(id).sum(), 0.0);
    }

    #[test]
    fn op_node_records_children() {
        let mut graph = Graph::new();
        let left = graph.leaf(array![[1.0]]);
        let right = graph.leaf(array![[2.0]]);
        let sum = graph.op(array![[3.0]], vec![left, right], None);

        assert_eq!(graph.children(sum), &[left, right]);
    }

    #[test]
    fn add_grad_accumulates() {
        let mut graph = Graph::new();
        let id = graph.leaf(array![[1.0, 2.0]]);

        graph.add_grad(id, &array![[0.5, 0.5]]);
        graph.add_grad(id, &array![[1.5, 1.5]]);

        assert_relative_eq!(graph.grad(id).sum(), 4.0);
    }

    #[test]
    fn backward_with_seed_propagates_custom_output_delta() {
        let mut graph = Graph::new();
        let w = graph.leaf(array![[1.0], [2.0]]);
        let x = graph.leaf(array![[3.0, 4.0], [5.0, 6.0]]);
        let y = graph.matmul(x, w);
        let seed = array![[0.5], [1.5]];
        graph.backward_with_seed(y, seed);

        assert_relative_eq!(graph.grad(w)[(0, 0)], 9.0);
        assert_relative_eq!(graph.grad(w)[(1, 0)], 11.0);
    }

    #[test]
    fn zero_grad_clears_all_nodes() {
        let mut graph = Graph::new();
        let a = graph.leaf(array![[1.0]]);
        let b = graph.leaf(array![[2.0]]);

        graph.set_grad(a, array![[3.0]]);
        graph.set_grad(b, array![[4.0]]);
        graph.zero_grad();

        assert_relative_eq!(graph.grad(a).sum(), 0.0);
        assert_relative_eq!(graph.grad(b).sum(), 0.0);
    }

    #[test]
    fn op_node_stores_backward_callback() {
        let mut graph = Graph::new();
        let leaf = graph.leaf(array![[2.0]]);
        let output = graph.op(array![[4.0]], vec![leaf], Some(Box::new(|_graph, _id| {})));

        assert_eq!(graph.children(output), &[leaf]);
    }

    #[test]
    fn backward_propagates_through_elementwise_mul() {
        let mut graph = Graph::new();
        let a = graph.leaf(array![[2.0]]);
        let b = graph.leaf(array![[3.0]]);
        let product = graph.op(
            array![[6.0]],
            vec![a, b],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                let b_data = graph.data(b).clone();
                let a_data = graph.data(a).clone();
                graph.add_grad(a, &(&grad_out * &b_data));
                graph.add_grad(b, &(&grad_out * &a_data));
            })),
        );

        graph.backward(product);

        assert_relative_eq!(graph.grad(a).sum(), 3.0);
        assert_relative_eq!(graph.grad(b).sum(), 2.0);
        assert_relative_eq!(graph.grad(product).sum(), 1.0);
    }

    #[test]
    fn backward_propagates_through_chained_ops() {
        let mut graph = Graph::new();
        let a = graph.leaf(array![[2.0]]);
        let b = graph.leaf(array![[3.0]]);
        let product = graph.op(
            array![[6.0]],
            vec![a, b],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                graph.add_grad(a, &(&grad_out * graph.data(b)));
                graph.add_grad(b, &(&grad_out * graph.data(a)));
            })),
        );
        let sum = graph.op(
            array![[8.0]],
            vec![product, a],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                graph.add_grad(product, &grad_out);
                graph.add_grad(a, &grad_out);
            })),
        );

        graph.backward(sum);

        // d(sum)/d(a) = d(sum)/d(product) * d(product)/d(a) + d(sum)/d(a) = 1 * 3 + 1 = 4
        assert_relative_eq!(graph.grad(a).sum(), 4.0);
        // d(sum)/d(b) = d(sum)/d(product) * d(product)/d(b) = 1 * 2 = 2
        assert_relative_eq!(graph.grad(b).sum(), 2.0);
    }

    #[test]
    fn clear_computation_graph_keeps_parameters() {
        let mut graph = Graph::new();
        let param_a = graph.leaf(array![[5.0]]);
        let param_b = graph.leaf(array![[10.0]]);
        let _intermediate = graph.op(array![[50.0]], vec![param_a, param_b], None);
        let _output = graph.op(array![[55.0]], vec![param_a], None);

        assert_eq!(graph.len(), 4);
        graph.clear_computation_graph(2);
        assert_eq!(graph.len(), 2);
        assert_relative_eq!(graph.data(param_a).sum(), 5.0);
        assert_relative_eq!(graph.data(param_b).sum(), 10.0);

        let new_node = graph.op(array![[15.0]], vec![param_a, param_b], None);
        assert_relative_eq!(graph.data(new_node).sum(), 15.0);
        assert_eq!(graph.len(), 3);
    }
}
