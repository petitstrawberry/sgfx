//! Verify discovery and dlopen negotiation without opening a GPU or submitting work.
use sgfx_backend_loader::{LoadedBackend, discover};

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = args.next().expect("driver directory");
    let gpu_backend = args.next().expect("GPU backend identifier");
    assert!(
        args.next().is_none(),
        "usage: probe_driver DIRECTORY GPU_BACKEND"
    );
    let manifest = discover(&[directory], &gpu_backend, None).expect("discover driver");
    let driver = LoadedBackend::load(&manifest).expect("load and negotiate ABI v2");
    assert!(driver.driver.is_some(), "missing programmable driver API");
    assert_eq!(driver.gpu_backend, gpu_backend);
    assert!(std::sync::Arc::ptr_eq(
        &driver,
        &LoadedBackend::load(&manifest).unwrap()
    ));
    println!(
        "PASS: {} ({}) {}",
        driver.name, driver.gpu_backend, driver.library
    );
}
