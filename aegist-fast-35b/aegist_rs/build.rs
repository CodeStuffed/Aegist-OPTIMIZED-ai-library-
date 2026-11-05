//! Compiles kernels/aegist.cu to PTX with nvcc (embedded into the binary) and links the
//! CUDA driver API. At run time aegist.exe needs only the NVIDIA driver, not the toolkit.
use std::{env, path::PathBuf, process::Command};

/// nvcc needs MSVC's cl.exe on Windows even for PTX; find it with vswhere when it is not on PATH.
fn msvc_bin() -> Option<PathBuf> {
    let vswhere = PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    let out = Command::new(vswhere).args(["-latest", "-products", "*", "-property", "installationPath"]).output().ok()?;
    let root = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim()).join(r"VC\Tools\MSVC");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&root).ok()?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    versions.sort();
    let bin = versions.pop()?.join(r"bin\Hostx64\x64");
    bin.join("cl.exe").exists().then_some(bin)
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/aegist.cu");
    println!("cargo:rerun-if-changed=kernels/hybrid.cu");
    println!("cargo:rerun-if-changed=kernels/next.cu");
    println!("cargo:rerun-if-changed=kernels/max.cu");
    println!("cargo:rerun-if-changed=kernels/kimi.cu");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("aegist.ptx");
    let mut nvcc = Command::new("nvcc");
    // compute_75 PTX: the driver JIT-compiles it for the GPU it finds (Turing and newer).
    nvcc.args(["--ptx", "-arch=compute_75", "-O3", "kernels/aegist.cu", "-o"]).arg(&out);
    if cfg!(windows) {
        if let Some(bin) = msvc_bin() {
            nvcc.arg("-ccbin").arg(bin);
        }
    }
    let status = nvcc.status().expect("nvcc not found: install the CUDA toolkit to build aegist");
    assert!(status.success(), "nvcc failed to compile kernels/aegist.cu");
    // The driver (nvcuda.dll) is linked in src/cuda.rs with raw-dylib: no cuda.lib needed.
}
