use std::cell::RefCell;

use rand::{SeedableRng, rngs::SmallRng};

pub fn clean_type_name<T>() -> String {
    let full_name = std::any::type_name::<T>();
    full_name
        .rsplit("::")
        .next()
        .unwrap_or(full_name)
        .to_string()
}

#[macro_export]
macro_rules! profile_layer {
    ($type:ty, $pass:expr, $block:expr) => {{
        #[cfg(feature = "profile-layers")]
        let _start = std::time::Instant::now();

        let res = $block;

        #[cfg(feature = "profile-layers")]
        {
            let name = $crate::utils::clean_type_name::<$type>(); // lub ścieżka gdzie to zdefiniowałeś
            println!("{:<25} | {:<8} | {:?}", name, $pass, _start.elapsed());
        }

        res
    }};
}

thread_local! {
    pub static GLOBAL_RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_rng(&mut rand::rng()));
}

pub fn manual_seed(seed: u64) {
    GLOBAL_RNG.with(|rng| {
        *rng.borrow_mut() = SmallRng::seed_from_u64(seed);
    });
}

pub fn with_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut SmallRng) -> R,
{
    GLOBAL_RNG.with(|rng| f(&mut *rng.borrow_mut()))
}
