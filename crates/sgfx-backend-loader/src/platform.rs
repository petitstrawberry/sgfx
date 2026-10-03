//! Filesystem adapter for standard Rust and Scarlet's legacy no_std processes.
use super::*;

#[cfg(feature = "std")]
pub fn entries(directory: &str) -> Result<Vec<String>, Error> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error(format!("{directory}: {e}"))),
    };
    entries
        .map(|entry| {
            let path = entry.map_err(|e| Error(e.to_string()))?.path();
            path.to_str()
                .map(String::from)
                .ok_or_else(|| Error("non-UTF-8 driver path".into()))
        })
        .collect()
}
#[cfg(not(feature = "std"))]
pub fn entries(directory: &str) -> Result<Vec<String>, Error> {
    use std::handle::{Handle, HandleError};
    let handle = match Handle::open(directory, 0) {
        Ok(handle) => handle,
        Err(HandleError::NotFound) => return Ok(Vec::new()),
        Err(e) => return Err(Error(format!("{directory}: {e:?}"))),
    };
    let mut file =
        std::fs::File::from_handle(handle).map_err(|e| Error(format!("{directory}: {e:?}")))?;
    let mut paths = Vec::new();
    while let Some(entry) = file
        .read_dir()
        .map_err(|e| Error(format!("{directory}: {e:?}")))?
    {
        paths.push(format!(
            "{}/{}",
            directory.trim_end_matches('/'),
            entry.name_str()
        ));
    }
    Ok(paths)
}
#[cfg(feature = "std")]
pub fn read(path: &str) -> Result<String, Error> {
    std::fs::read_to_string(path).map_err(|e| Error(format!("{path}: {e}")))
}
#[cfg(not(feature = "std"))]
pub fn read(path: &str) -> Result<String, Error> {
    let mut file = std::fs::File::open(path).map_err(|e| Error(format!("{path}: {e:?}")))?;
    let mut bytes = Vec::new();
    let mut chunk = [0; 1024];
    loop {
        let count = file
            .read(&mut chunk)
            .map_err(|e| Error(format!("{path}: {e:?}")))?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    String::from_utf8(bytes).map_err(|e| Error(format!("{path}: {e}")))
}
#[cfg(feature = "std")]
pub fn absolute(path: &str) -> Result<String, Error> {
    std::fs::canonicalize(path)
        .map_err(|e| Error(format!("{path}: {e}")))?
        .to_str()
        .map(String::from)
        .ok_or_else(|| Error("non-UTF-8 driver path".into()))
}
#[cfg(not(feature = "std"))]
pub fn absolute(path: &str) -> Result<String, Error> {
    // Legacy fs has no canonicalize. Preserve components (including symlinks)
    // for scarlet-ld to resolve; never lexically collapse a symlink/../ path.
    if path.starts_with('/') {
        Ok(path.into())
    } else {
        let cwd = std::fs::get_cwd_path().map_err(|e| Error(format!("driver cwd: {e:?}")))?;
        Ok(format!("{}/{path}", cwd.trim_end_matches('/')))
    }
}
