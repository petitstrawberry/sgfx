//! Manifest-based backend discovery and one-time dynamic loading. Loading and
//! symbol resolution never occur on the rendering path. Drivers remain resident
//! for process lifetime, including while native GPU workers are still running.

#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
#[cfg(all(
    not(feature = "std"),
    feature = "legacy-scarlet-std",
    target_os = "scarlet"
))]
extern crate scarlet_std as std;
#[cfg(not(any(
    feature = "std",
    all(feature = "legacy-scarlet-std", target_os = "scarlet")
)))]
compile_error!("SGFX loader requires std or legacy-scarlet-std");
use alloc::{
    ffi::CString,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::ffi::{CStr, c_char, c_int, c_void};
use sgfx_backend_abi as abi;
use std::sync::Mutex;
mod platform;

#[derive(Debug)]
pub struct Error(pub String);
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}
impl core::error::Error for Error {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub name: String,
    pub gpu_backend: String,
    pub library: String,
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() < 32
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

impl Manifest {
    pub fn parse(path: &str, source: &str) -> Result<Self, Error> {
        let (mut name, mut gpu, mut library, mut version) = (None, None, None, None);
        for line in source
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| Error(format!("invalid driver manifest: {}", path)))?;
            let slot = match key.trim() {
                "name" => &mut name,
                "gpu_backend" => &mut gpu,
                "library" => &mut library,
                "abi" => &mut version,
                _ => return Err(Error(format!("unknown driver manifest key: {key}"))),
            };
            if slot.replace(value.trim()).is_some() {
                return Err(Error(format!("duplicate driver manifest key: {key}")));
            }
        }
        let name = name
            .filter(|s| identifier(s))
            .ok_or_else(|| Error("invalid or missing driver name".into()))?;
        let gpu = gpu
            .filter(|s| identifier(s))
            .ok_or_else(|| Error("invalid or missing GPU backend ID".into()))?;
        let library = library
            .filter(|s| !s.is_empty() && !s.contains(['/', '\\', '\0']) && *s != "." && *s != "..")
            .ok_or_else(|| Error("library must be a filename alongside its manifest".into()))?;
        if version != Some("2") {
            return Err(Error("incompatible SGFX driver ABI".into()));
        }
        Ok(Self {
            name: name.into(),
            gpu_backend: gpu.into(),
            library: format!(
                "{}/{}",
                path.rsplit_once('/')
                    .map(|(parent, _)| parent)
                    .unwrap_or("."),
                library
            ),
        })
    }
}

pub fn discover(
    directories: &[String],
    gpu_backend: &str,
    preference: Option<&str>,
) -> Result<Manifest, Error> {
    let mut matches = Vec::new();
    let mut invalid_manifests = Vec::new();
    for directory in directories {
        for path in platform::entries(directory)? {
            if !path.ends_with(".sgfx-driver") {
                continue;
            }
            let source = platform::read(&path)?;
            let manifest = match Manifest::parse(&path, &source) {
                Ok(manifest) => manifest,
                Err(error) => {
                    // Installing a newer ABI or an unrelated broken manifest
                    // must not disable the compatible driver for this GPU.
                    invalid_manifests.push(format!("{}: {error}", path));
                    continue;
                }
            };
            if manifest.gpu_backend == gpu_backend && preference.is_none_or(|p| p == manifest.name)
            {
                matches.push(manifest);
            }
        }
    }
    matches.sort_by(|a, b| a.name.cmp(&b.name).then(a.library.cmp(&b.library)));
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => Err(Error(format!(
            "no installed SGFX driver for GPU {gpu_backend}{}{}",
            preference
                .map(|s| format!(" (requested {s})"))
                .unwrap_or_default(),
            if invalid_manifests.is_empty() {
                String::new()
            } else {
                format!(
                    "; skipped incompatible manifests: {}",
                    invalid_manifests.join("; ")
                )
            }
        ))),
        _ => Err(Error(format!(
            "multiple SGFX drivers match GPU {gpu_backend}; select one with SGFX_BACKEND"
        ))),
    }
}

/// Cached function table. The library is pinned; objects call this table without
/// taking the loader mutex or looking up a symbol again.
pub struct LoadedBackend {
    pub api: abi::BackendApi,
    pub driver: Option<abi::DriverApi>,
    pub ycbcr: Option<abi::YcbcrApi>,
    pub name: String,
    pub gpu_backend: String,
    /// Absolute path passed to the loader for this driver.
    pub library: String,
}
static LOADED: Mutex<Vec<(String, Arc<LoadedBackend>)>> = Mutex::new(Vec::new());

#[cfg(any(unix, target_os = "scarlet"))]
#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

#[cfg(any(unix, target_os = "scarlet"))]
fn dynamic_error() -> Error {
    let ptr = unsafe { dlerror() };
    if ptr.is_null() {
        Error("dynamic loader operation failed".into())
    } else {
        Error(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

#[cfg(any(unix, target_os = "scarlet"))]
fn negotiate_ycbcr(get: abi::GetYcbcrApi) -> Result<abi::YcbcrApi, Error> {
    let mut table = core::mem::MaybeUninit::<abi::YcbcrApi>::uninit();
    let status = unsafe {
        get(
            abi::VERSION,
            core::mem::size_of::<abi::YcbcrApi>(),
            table.as_mut_ptr(),
        )
    };
    if status != abi::OK {
        return Err(Error(format!(
            "driver rejected YCbCr ABI negotiation: {status}"
        )));
    }
    // A successful entry point initializes the complete extension table.
    let table = unsafe { table.assume_init() };
    if table.version != abi::VERSION
        || (table.size as usize) < core::mem::size_of::<abi::YcbcrApi>()
    {
        return Err(Error("incompatible YCbCr driver table".into()));
    }
    Ok(table)
}

#[cfg(any(unix, target_os = "scarlet"))]
unsafe fn load_ycbcr(library: *mut c_void) -> Result<Option<abi::YcbcrApi>, Error> {
    let address = unsafe { dlsym(library, abi::YCBCR_ENTRY.as_ptr().cast()) };
    if address.is_null() {
        // Optional absence must not leave a stale dynamic loader error.
        let _ = unsafe { dlerror() };
        Ok(None)
    } else {
        let get: abi::GetYcbcrApi = unsafe { core::mem::transmute(address) };
        negotiate_ycbcr(get).map(Some)
    }
}

impl LoadedBackend {
    pub fn load(manifest: &Manifest) -> Result<Arc<Self>, Error> {
        let path = platform::absolute(&manifest.library)?;
        #[cfg(feature = "std")]
        let mut loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(not(feature = "std"))]
        let mut loaded = LOADED.lock();
        if let Some((_, backend)) = loaded.iter().find(|(p, _)| p == &path) {
            if backend.name != manifest.name || backend.gpu_backend != manifest.gpu_backend {
                return Err(Error("driver manifest differs from loaded library".into()));
            }
            return Ok(backend.clone());
        }
        let backend = Self::load_uncached(&path, manifest)?;
        let backend = Arc::new(backend);
        loaded.push((path, backend.clone()));
        Ok(backend)
    }

    #[cfg(any(unix, target_os = "scarlet"))]
    fn load_uncached(path: &str, manifest: &Manifest) -> Result<Self, Error> {
        let library_path = path.to_string();
        let path = CString::new(path).map_err(|_| Error("NUL in driver path".into()))?;
        #[cfg(target_os = "scarlet")]
        let flags = 0x102; // scarlet-ld currently requires NOW | GLOBAL.
        #[cfg(all(unix, not(target_os = "scarlet")))]
        let flags = 2; // NOW | LOCAL on the host, with the same process-lifetime pin.
        let library = unsafe { dlopen(path.as_ptr(), flags) };
        if library.is_null() {
            return Err(dynamic_error());
        }
        // Never dlclose: even a failed ABI negotiation can have run constructors.
        let address = unsafe { dlsym(library, abi::ENTRY.as_ptr().cast()) };
        if address.is_null() {
            return Err(dynamic_error());
        }
        let entry: abi::GetApi = unsafe { core::mem::transmute(address) };
        let mut table = core::mem::MaybeUninit::<abi::BackendApi>::uninit();
        #[cfg(all(feature = "std", target_arch = "aarch64"))]
        let cpu_features = if std::arch::is_aarch64_feature_detected!("lse") {
            abi::CPU_AARCH64_LSE
        } else {
            0
        };
        #[cfg(not(all(feature = "std", target_arch = "aarch64")))]
        let cpu_features = 0;
        let host = abi::HostInfo {
            size: core::mem::size_of::<abi::HostInfo>() as u32,
            reserved: 0,
            cpu_features,
        };
        let status = unsafe {
            entry(
                abi::VERSION,
                core::mem::size_of::<abi::BackendApi>(),
                &host,
                table.as_mut_ptr(),
            )
        };
        if status != abi::OK {
            return Err(Error(format!(
                "driver rejected SGFX ABI negotiation: {status}"
            )));
        }
        // The entry-point contract initializes the complete v2 table on success.
        let api = unsafe { table.assume_init() };
        if api.version != abi::VERSION
            || (api.size as usize) < core::mem::size_of::<abi::BackendApi>()
        {
            return Err(Error("incompatible SGFX driver table".into()));
        }
        fn text(bytes: &[u8]) -> Result<String, Error> {
            let len = bytes
                .iter()
                .position(|c| *c == 0)
                .ok_or_else(|| Error("unterminated driver identifier".into()))?;
            let value = core::str::from_utf8(&bytes[..len])
                .map_err(|_| Error("invalid driver identifier".into()))?;
            if !identifier(value) {
                return Err(Error("invalid driver identifier".into()));
            }
            Ok(value.into())
        }
        let name = text(&api.name)?;
        let gpu_backend = text(&api.gpu_backend)?;
        if name != manifest.name || gpu_backend != manifest.gpu_backend {
            return Err(Error("driver does not match its manifest".into()));
        }
        // Optional extension lookup is cold; no dlsym occurs during rendering.
        let address = unsafe { dlsym(library, abi::DRIVER_ENTRY.as_ptr().cast()) };
        let driver = if address.is_null() {
            let _ = unsafe { dlerror() };
            None
        } else {
            let get: abi::GetDriverApi = unsafe { core::mem::transmute(address) };
            let mut table = core::mem::MaybeUninit::<abi::DriverApi>::uninit();
            if unsafe {
                get(
                    abi::VERSION,
                    core::mem::size_of::<abi::DriverApi>(),
                    table.as_mut_ptr(),
                )
            } != abi::OK
            {
                return Err(Error("driver rejected low-level ABI negotiation".into()));
            }
            let table = unsafe { table.assume_init() };
            if table.version != abi::VERSION
                || (table.size as usize) < core::mem::size_of::<abi::DriverApi>()
            {
                return Err(Error("incompatible low-level driver table".into()));
            }
            Some(table)
        };
        let ycbcr = unsafe { load_ycbcr(library) }?;
        Ok(Self {
            api,
            driver,
            ycbcr,
            name,
            gpu_backend,
            library: library_path,
        })
    }

    #[cfg(not(any(unix, target_os = "scarlet")))]
    fn load_uncached(_path: &str, _manifest: &Manifest) -> Result<Self, Error> {
        Err(Error(
            "dynamic SGFX drivers are not supported on this platform".into(),
        ))
    }
}

pub fn driver_directories() -> Vec<String> {
    #[cfg(feature = "std")]
    let value = std::env::var("SGFX_DRIVER_PATH").ok();
    #[cfg(not(feature = "std"))]
    let value = std::env::var("SGFX_DRIVER_PATH");
    match value {
        Some(value) => value
            .split(':')
            .filter(|p| !p.is_empty())
            .map(String::from)
            .collect(),
        None => vec![String::from(if cfg!(target_os = "linux") {
            "/usr/lib/sgfx"
        } else {
            "/system/lib/sgfx"
        })],
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    unsafe extern "C" fn import_ycbcr(
        _session: abi::Object,
        _image: u32,
        _handle: i32,
        _conversion: abi::YcbcrConversion,
    ) -> i32 {
        abi::UNSUPPORTED
    }

    unsafe extern "C" fn ycbcr_table<const VERSION: u32, const SIZE: u32>(
        version: u32,
        size: usize,
        out: *mut abi::YcbcrApi,
    ) -> i32 {
        if version != abi::VERSION || size != core::mem::size_of::<abi::YcbcrApi>() {
            return abi::ABI_MISMATCH;
        }
        unsafe {
            out.write(abi::YcbcrApi {
                version: VERSION,
                size: SIZE,
                import_ycbcr,
            });
        }
        abi::OK
    }

    #[cfg(any(unix, target_os = "scarlet"))]
    #[test]
    fn absent_ycbcr_extension_is_optional_and_clears_loader_error() {
        let process = unsafe { dlopen(core::ptr::null(), 2) };
        assert!(!process.is_null());
        assert!(unsafe { load_ycbcr(process) }.unwrap().is_none());
        assert!(unsafe { dlerror() }.is_null());
    }

    #[cfg(any(unix, target_os = "scarlet"))]
    #[test]
    fn ycbcr_negotiation_accepts_compatible_table() {
        let table = negotiate_ycbcr(ycbcr_table::<2, 16>).unwrap();
        assert_eq!(table.version, abi::VERSION);
        assert_eq!(table.size, core::mem::size_of::<abi::YcbcrApi>() as u32);
    }

    #[cfg(any(unix, target_os = "scarlet"))]
    #[test]
    fn ycbcr_negotiation_rejects_invalid_version_or_size() {
        for get in [
            ycbcr_table::<1, 16> as abi::GetYcbcrApi,
            ycbcr_table::<2, 15> as abi::GetYcbcrApi,
        ] {
            assert_eq!(
                negotiate_ycbcr(get).err().unwrap().0,
                "incompatible YCbCr driver table"
            );
        }
    }

    #[cfg(any(unix, target_os = "scarlet"))]
    #[test]
    fn ycbcr_negotiation_rejection_does_not_read_uninitialized_table() {
        unsafe extern "C" fn reject(_version: u32, _size: usize, _out: *mut abi::YcbcrApi) -> i32 {
            abi::ABI_MISMATCH
        }
        assert_eq!(
            negotiate_ycbcr(reject).err().unwrap().0,
            "driver rejected YCbCr ABI negotiation: 7"
        );
    }

    #[test]
    fn manifests_are_independent_of_compiled_backend_names() {
        let manifest = Manifest::parse(
            "/drivers/new.sgfx-driver",
            "abi=2\nname=future-gpu\ngpu_backend=new-gpu\nlibrary=libfuture.so\n",
        )
        .unwrap();
        assert_eq!(manifest.library, "/drivers/libfuture.so");
        for source in [
            "abi=1\nname=x\ngpu_backend=y\nlibrary=x.so",
            "abi=2\nabi=2\nname=x\ngpu_backend=y\nlibrary=x.so",
            "abi=2\nname=x\ngpu_backend=y\nlibrary=../x.so",
        ] {
            assert!(Manifest::parse("/drivers/x.sgfx-driver", source).is_err());
        }
    }

    #[test]
    fn discovery_selects_installed_drivers_and_reports_missing_or_ambiguous_matches() {
        struct Directory(std::path::PathBuf);
        impl Drop for Directory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let directory =
            std::env::temp_dir().join(format!("sgfx-driver-discovery-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let directory = Directory(directory);
        let dirs = [directory.0.to_str().unwrap().to_string()];
        let install = |name: &str, version: u32| {
            std::fs::write(
                directory.0.join(format!("{name}.sgfx-driver")),
                format!(
                    "abi={version}\nname={name}\ngpu_backend=future-gpu\nlibrary=lib{name}.so\n"
                ),
            )
            .unwrap();
        };
        assert!(discover(&dirs, "future-gpu", None).is_err());
        install("newer", abi::VERSION + 1);
        install("first", abi::VERSION);
        assert_eq!(discover(&dirs, "future-gpu", None).unwrap().name, "first");
        assert!(discover(&dirs, "different-gpu", None).is_err());
        install("second", abi::VERSION);
        assert!(discover(&dirs, "future-gpu", None).is_err());
        assert_eq!(
            discover(&dirs, "future-gpu", Some("second")).unwrap().name,
            "second"
        );
        assert!(discover(&dirs, "future-gpu", Some("missing")).is_err());
        let missing = Manifest::parse(
            directory.0.join("missing.sgfx-driver").to_str().unwrap(),
            "abi=2\nname=missing\ngpu_backend=future-gpu\nlibrary=libmissing.so",
        )
        .unwrap();
        assert!(LoadedBackend::load(&missing).is_err());
    }
}
