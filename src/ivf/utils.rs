use std::collections::BinaryHeap;

use crate::ivf::Neighbor;

pub struct TopKNeighbors {
    heap: BinaryHeap<Neighbor>,
    k: usize,
}

impl TopKNeighbors {
    pub fn new(k: usize) -> Self {
        Self { heap: BinaryHeap::with_capacity(k + 1), k }
    }

    pub fn extend(&mut self, neighbors: impl IntoIterator<Item = Neighbor>) {
        self.heap.extend(neighbors);
        while self.heap.len() > self.k {
            self.heap.pop();
        }
    }

    pub fn into_vec(self) -> Vec<Neighbor> {
        self.heap.into_vec()
    }
}
