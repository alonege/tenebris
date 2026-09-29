pub struct TensorId {
    id: usize,
}

impl TensorId {
    pub fn new(id: usize) -> Self {
        Self { id }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }
}
