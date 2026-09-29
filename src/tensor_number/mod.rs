pub trait TensorNumber: Default + Copy + Send + Sync {}
impl TensorNumber for f32 {}
impl TensorNumber for f64 {}
impl TensorNumber for f16 {}
impl TensorNumber for i8 {}
impl TensorNumber for u8 {}
impl TensorNumber for i16 {}
impl TensorNumber for u16 {}
impl TensorNumber for i32 {}
impl TensorNumber for u32 {}
impl TensorNumber for i64 {}
impl TensorNumber for u64 {}
