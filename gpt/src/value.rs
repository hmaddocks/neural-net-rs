//! Autograd engine for automatic differentiation
//!
//! This module implements a minimal automatic differentiation framework using an arena-based
//! approach for efficient memory management. Instead of reference counting (Rc<RefCell<>>),
//! all Value nodes are stored in a Vec and referenced by indices (ValueId).

/// A unique identifier for a Value node in the arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

/// A node in the computation graph
#[derive(Debug, Clone)]
pub struct Value {
    /// The scalar value computed during the forward pass
    pub data: f64,
    /// The gradient computed during the backward pass
    pub grad: f64,
    /// Indices of children nodes in the computation graph
    children: Vec<ValueId>,
    /// Local derivatives with respect to each child (for chain rule)
    local_grads: Vec<f64>,
}

impl Value {
    /// Create a new leaf Value with the given data
    pub fn new(data: f64) -> Self {
        Self {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        }
    }

    /// Create a new Value with children and local gradients
    fn with_children(data: f64, children: Vec<ValueId>, local_grads: Vec<f64>) -> Self {
        Self {
            data,
            grad: 0.0,
            children,
            local_grads,
        }
    }
}

/// Arena that owns all Value nodes
pub struct ValueArena {
    values: Vec<Value>,
}

impl ValueArena {
    /// Create a new empty arena
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Create a new Value node and return its ID
    pub fn create(&mut self, data: f64) -> ValueId {
        let id = ValueId(self.values.len());
        self.values.push(Value::new(data));
        id
    }

    /// Create a Value node with children and return its ID
    fn create_with_children(
        &mut self,
        data: f64,
        children: Vec<ValueId>,
        local_grads: Vec<f64>,
    ) -> ValueId {
        let id = ValueId(self.values.len());
        self.values
            .push(Value::with_children(data, children, local_grads));
        id
    }

    /// Get a reference to a Value by ID
    #[inline]
    pub fn get(&self, id: ValueId) -> &Value {
        &self.values[id.0]
    }

    /// Get a mutable reference to a Value by ID
    #[inline]
    pub fn get_mut(&mut self, id: ValueId) -> &mut Value {
        &mut self.values[id.0]
    }

    /// Get the data value of a node
    #[inline]
    pub fn data(&self, id: ValueId) -> f64 {
        self.values[id.0].data
    }

    /// Get the gradient of a node
    #[inline]
    pub fn grad(&self, id: ValueId) -> f64 {
        self.values[id.0].grad
    }

    /// Set the data value of a node
    #[inline]
    pub fn set_data(&mut self, id: ValueId, data: f64) {
        self.values[id.0].data = data;
    }

    /// Set the gradient of a node
    #[inline]
    pub fn set_grad(&mut self, id: ValueId, grad: f64) {
        self.values[id.0].grad = grad;
    }

    /// Add to the gradient of a node
    #[inline]
    pub fn add_grad(&mut self, id: ValueId, grad: f64) {
        self.values[id.0].grad += grad;
    }

    /// Zero all gradients in the arena
    pub fn zero_grad(&mut self) {
        self.values.iter_mut().map(|v| v.grad = 0.0).collect()
    }

    /// Clear the computation graph while preserving model parameters
    ///
    /// This truncates the arena to keep only the first `num_params_to_keep` values,
    /// which should be the model parameters. This is essential for preventing
    /// unbounded memory growth during training.
    ///
    /// # Arguments
    /// * `num_params_to_keep` - Number of parameter values to preserve at the start of the arena
    ///
    /// # Example
    /// ```
    /// use gpt::value::ValueArena;
    ///
    /// let mut arena = ValueArena::new();
    /// // Create model parameters
    /// let param1 = arena.create(1.0);
    /// let param2 = arena.create(2.0);
    /// let num_params = 2;
    ///
    /// // ... do forward and backward pass ...
    /// let intermediate = arena.add(param1, param2);
    ///
    /// // Clear computation graph, keeping only parameters
    /// arena.clear_computation_graph(num_params);
    /// ```
    pub fn clear_computation_graph(&mut self, num_params_to_keep: usize) {
        self.values.truncate(num_params_to_keep);
    }

    /// Addition operation: a + b
    pub fn add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let data = self.data(a) + self.data(b);
        self.create_with_children(data, vec![a, b], vec![1.0, 1.0])
    }

    /// Addition with scalar: a + c
    pub fn add_scalar(&mut self, a: ValueId, c: f64) -> ValueId {
        let data = self.data(a) + c;
        self.create_with_children(data, vec![a], vec![1.0])
    }

    /// Multiplication operation: a * b
    pub fn mul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let a_data = self.data(a);
        let b_data = self.data(b);
        let data = a_data * b_data;
        self.create_with_children(data, vec![a, b], vec![b_data, a_data])
    }

    /// Multiplication with scalar: a * c
    pub fn mul_scalar(&mut self, a: ValueId, c: f64) -> ValueId {
        let data = self.data(a) * c;
        self.create_with_children(data, vec![a], vec![c])
    }

    /// Power operation: a^n
    pub fn pow(&mut self, a: ValueId, n: f64) -> ValueId {
        let a_data = self.data(a);
        let data = a_data.powf(n);
        let local_grad = n * a_data.powf(n - 1.0);
        self.create_with_children(data, vec![a], vec![local_grad])
    }

    /// Natural logarithm: ln(a)
    pub fn log(&mut self, a: ValueId) -> ValueId {
        let a_data = self.data(a);
        let data = a_data.ln();
        let local_grad = 1.0 / a_data;
        self.create_with_children(data, vec![a], vec![local_grad])
    }

    /// Exponential: e^a
    pub fn exp(&mut self, a: ValueId) -> ValueId {
        let a_data = self.data(a);
        let data = a_data.exp();
        let local_grad = data;
        self.create_with_children(data, vec![a], vec![local_grad])
    }

    /// ReLU activation: max(0, a)
    pub fn relu(&mut self, a: ValueId) -> ValueId {
        let a_data = self.data(a);
        let data = a_data.max(0.0);
        let local_grad = if a_data > 0.0 { 1.0 } else { 0.0 };
        self.create_with_children(data, vec![a], vec![local_grad])
    }

    /// Negation: -a
    pub fn neg(&mut self, a: ValueId) -> ValueId {
        self.mul_scalar(a, -1.0)
    }

    /// Subtraction: a - b
    pub fn sub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let neg_b = self.neg(b);
        self.add(a, neg_b)
    }

    /// Subtraction with scalar: a - c
    pub fn sub_scalar(&mut self, a: ValueId, c: f64) -> ValueId {
        self.add_scalar(a, -c)
    }

    /// Division: a / b
    pub fn div(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let b_inv = self.pow(b, -1.0);
        self.mul(a, b_inv)
    }

    /// Division with scalar: a / c
    pub fn div_scalar(&mut self, a: ValueId, c: f64) -> ValueId {
        self.mul_scalar(a, 1.0 / c)
    }

    /// Perform backpropagation from the given node
    ///
    /// This implements reverse-mode automatic differentiation by:
    /// 1. Building a topological ordering of the computation graph
    /// 2. Setting the gradient of the output node to 1.0
    /// 3. Traversing backwards, applying the chain rule at each node
    pub fn backward(&mut self, output: ValueId) {
        // Build topological ordering using DFS
        let mut topo = Vec::new();
        let mut visited = vec![false; self.values.len()];

        fn build_topo(
            arena: &ValueArena,
            v: ValueId,
            visited: &mut Vec<bool>,
            topo: &mut Vec<ValueId>,
        ) {
            if !visited[v.0] {
                visited[v.0] = true;
                let children = arena.values[v.0].children.clone();
                for child in children {
                    build_topo(arena, child, visited, topo);
                }
                topo.push(v);
            }
        }

        build_topo(self, output, &mut visited, &mut topo);

        // Initialize output gradient to 1.0
        self.values[output.0].grad = 1.0;

        // Propagate gradients backward through the graph
        for &v_id in topo.iter().rev() {
            let v = &self.values[v_id.0];
            let v_grad = v.grad;
            let children = v.children.clone();
            let local_grads = v.local_grads.clone();

            children
                .iter()
                .zip(local_grads)
                .for_each(|(child, local_grad)| {
                    self.values[child.0].grad += local_grad * v_grad;
                });
        }
    }

    /// Get the number of values in the arena
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the arena is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Default for ValueArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut arena = ValueArena::new();
        let a = arena.create(2.0);
        let b = arena.create(3.0);
        let c = arena.add(a, b);
        assert_eq!(arena.data(c), 5.0);

        let d = arena.mul(a, b);
        assert_eq!(arena.data(d), 6.0);
    }

    #[test]
    fn test_backward_simple() {
        let mut arena = ValueArena::new();
        let a = arena.create(2.0);
        let b = arena.create(3.0);
        let c = arena.mul(a, b); // c = a * b = 6

        arena.backward(c);

        // dc/da = b = 3.0
        assert_eq!(arena.grad(a), 3.0);
        // dc/db = a = 2.0
        assert_eq!(arena.grad(b), 2.0);
        // dc/dc = 1.0
        assert_eq!(arena.grad(c), 1.0);
    }

    #[test]
    fn test_backward_chain() {
        let mut arena = ValueArena::new();
        let a = arena.create(2.0);
        let b = arena.create(3.0);
        let c = arena.mul(a, b); // c = a * b = 6
        let d = arena.add(c, a); // d = c + a = 8

        arena.backward(d);

        // dd/da = dd/dc * dc/da + dd/da = 1 * 3 + 1 = 4
        assert_eq!(arena.grad(a), 4.0);
        // dd/db = dd/dc * dc/db = 1 * 2 = 2
        assert_eq!(arena.grad(b), 2.0);
    }

    #[test]
    fn test_exp_log() {
        let mut arena = ValueArena::new();
        let a = arena.create(2.0);
        let b = arena.exp(a);
        let c = arena.log(b);

        assert!((arena.data(c) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu() {
        let mut arena = ValueArena::new();
        let a = arena.create(-5.0);
        let b = arena.relu(a);
        assert_eq!(arena.data(b), 0.0);

        let c = arena.create(5.0);
        let d = arena.relu(c);
        assert_eq!(arena.data(d), 5.0);
    }

    #[test]
    fn test_pow() {
        let mut arena = ValueArena::new();
        let a = arena.create(2.0);
        let b = arena.pow(a, 3.0);
        assert_eq!(arena.data(b), 8.0);

        arena.backward(b);
        // db/da = 3 * 2^2 = 12
        assert_eq!(arena.grad(a), 12.0);
    }

    #[test]
    fn test_zero_grad() {
        let mut arena = ValueArena::new();
        let a = arena.create(2.0);
        let b = arena.create(3.0);
        let c = arena.mul(a, b);

        arena.backward(c);
        assert_eq!(arena.grad(a), 3.0);

        arena.zero_grad();
        assert_eq!(arena.grad(a), 0.0);
        assert_eq!(arena.grad(b), 0.0);
        assert_eq!(arena.grad(c), 0.0);
    }

    #[test]
    fn test_clear_computation_graph() {
        let mut arena = ValueArena::new();

        // Create "parameters" (first 2 values)
        let param1 = arena.create(5.0);
        let param2 = arena.create(10.0);

        // Create computation graph
        let a = arena.mul(param1, param2); // 50.0
        let b = arena.add(a, param1); // 55.0
        let _c = arena.mul(b, param2); // 550.0

        // Should have 5 values total
        assert_eq!(arena.len(), 5);

        // Clear computation graph, keeping only the first 2 (parameters)
        arena.clear_computation_graph(2);

        // Should now have only 2 values
        assert_eq!(arena.len(), 2);

        // Parameters should still be accessible with correct values
        assert_eq!(arena.data(param1), 5.0);
        assert_eq!(arena.data(param2), 10.0);

        // Can continue using the arena
        let new_computation = arena.add(param1, param2);
        assert_eq!(arena.data(new_computation), 15.0);
        assert_eq!(arena.len(), 3);
    }
}
