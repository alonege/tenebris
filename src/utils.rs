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
