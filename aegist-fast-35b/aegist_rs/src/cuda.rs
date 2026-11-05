//! The few CUDA driver API calls Aegist needs, with RAII device buffers.
//! Linked straight to the driver DLL (raw-dylib), so no CUDA toolkit is needed to link or run.
//!
//! Several GPUs: each `Gpu` is one device's primary context. Every buffer and kernel remembers the
//! context it was made in and makes it current before it is used, so code that works on two GPUs
//! never has to switch contexts itself. Buffers are made in the context current when they are
//! allocated: call `Gpu::bind` first.
use std::cell::Cell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;

#[cfg_attr(windows, link(name = "nvcuda", kind = "raw-dylib"))]
#[cfg_attr(not(windows), link(name = "cuda"))]
extern "system" {
    fn cuInit(flags: u32) -> CUresult;
    fn cuDeviceGetCount(count: *mut i32) -> CUresult;
    fn cuDeviceGet(dev: *mut CUdevice, ordinal: i32) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, n: usize) -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    fn cuLaunchKernel(f: CUfunction, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, shmem: u32,
                      stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult;
    fn cuGetErrorString(err: CUresult, s: *mut *const c_char) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
}

/// $AEGIST_SYNC_LAUNCH (debugging): wait for every kernel and report its errors at the launch site.
fn sync_launches() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var_os("AEGIST_SYNC_LAUNCH").is_some())
}

pub type Result<T> = std::result::Result<T, String>;

fn check(r: CUresult, what: &str) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let mut s: *const c_char = ptr::null();
    unsafe { cuGetErrorString(r, &mut s) };
    let msg = if s.is_null() { "unknown error".into() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() };
    Err(format!("CUDA error {r} in {what}: {msg}"))
}

thread_local! {
    /// The context this thread made current last (0: none yet), so switching to the context that
    /// is already current costs nothing.
    static CURRENT: Cell<usize> = const { Cell::new(0) };
}

fn make_current(ctx: usize) -> Result<()> {
    CURRENT.with(|c| {
        if c.get() != ctx {
            check(unsafe { cuCtxSetCurrent(ctx as CUcontext) }, "cuCtxSetCurrent")?;
            c.set(ctx);
        }
        Ok(())
    })
}

fn current() -> Result<usize> {
    let mut ctx = ptr::null_mut();
    check(unsafe { cuCtxGetCurrent(&mut ctx) }, "cuCtxGetCurrent")?;
    if ctx.is_null() {
        return Err("no GPU context is current (Gpu::bind first)".into());
    }
    CURRENT.with(|c| c.set(ctx as usize));
    Ok(ctx as usize)
}

/// How many CUDA GPUs the driver sees.
pub fn device_count() -> Result<usize> {
    let mut n = 0;
    unsafe {
        check(cuInit(0), "cuInit (is an NVIDIA GPU and driver present?)")?;
        check(cuDeviceGetCount(&mut n), "cuDeviceGetCount")?;
    }
    Ok(n.max(0) as usize)
}

pub struct Gpu {
    pub name: String,
    pub ordinal: usize,
    module: CUmodule,
    ctx: usize,
}

impl Gpu {
    /// The first GPU.
    pub fn new(ptx: &str) -> Result<Gpu> {
        Gpu::on(0, ptx)
    }

    /// GPU number `ordinal` (as nvidia-smi counts them, unless CUDA_VISIBLE_DEVICES reorders),
    /// with the kernels loaded; its context is left current.
    pub fn on(ordinal: usize, ptx: &str) -> Result<Gpu> {
        unsafe {
            check(cuInit(0), "cuInit (is an NVIDIA GPU and driver present?)")?;
            let mut dev = 0;
            check(cuDeviceGet(&mut dev, ordinal as i32), &format!("cuDeviceGet({ordinal})"))?;
            let mut name = [0 as c_char; 256];
            check(cuDeviceGetName(name.as_mut_ptr(), 256, dev), "cuDeviceGetName")?;
            let mut ctx = ptr::null_mut();
            check(cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            make_current(ctx as usize)?;
            let image = CString::new(ptx).map_err(|e| e.to_string())?;
            let mut module = ptr::null_mut();
            check(cuModuleLoadData(&mut module, image.as_ptr() as *const c_void), "loading kernels")?;
            Ok(Gpu { name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(), ordinal, module, ctx: ctx as usize })
        }
    }

    /// Make this GPU current: buffers allocated next live on it.
    pub fn bind(&self) -> Result<()> {
        make_current(self.ctx)
    }

    pub fn function(&self, name: &str) -> Result<Kernel> {
        let c = CString::new(name).unwrap();
        let mut f = ptr::null_mut();
        check(unsafe { cuModuleGetFunction(&mut f, self.module, c.as_ptr()) }, name)?;
        Ok(Kernel { f, name: name.to_string(), ctx: self.ctx })
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> (usize, usize) {
        let (mut free, mut total) = (0usize, 0usize);
        if self.bind().is_ok() {
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        }
        (free, total)
    }
}

/// A kernel argument. Every value is passed by pointer to an 8-byte slot.
#[derive(Clone, Copy)]
pub enum Arg {
    Ptr(u64),
    I32(i32),
    F32(f32),
}

pub struct Kernel {
    f: CUfunction,
    name: String,
    ctx: usize,
}

impl Kernel {
    pub fn launch(&self, grid: (u32, u32), block: u32, args: &[Arg]) -> Result<()> {
        make_current(self.ctx)?;
        let mut slots: Vec<[u8; 8]> = args
            .iter()
            .map(|a| {
                let mut s = [0u8; 8];
                match *a {
                    Arg::Ptr(p) => s.copy_from_slice(&p.to_le_bytes()),
                    Arg::I32(v) => s[..4].copy_from_slice(&v.to_le_bytes()),
                    Arg::F32(v) => s[..4].copy_from_slice(&v.to_le_bytes()),
                }
                s
            })
            .collect();
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s.as_mut_ptr() as *mut c_void).collect();
        let r = unsafe {
            cuLaunchKernel(self.f, grid.0, grid.1, 1, block, 1, 1, 0, ptr::null_mut(), ptrs.as_mut_ptr(), ptr::null_mut())
        };
        check(r, &self.name)?;
        if sync_launches() {
            check(unsafe { cuCtxSynchronize() }, &format!("{} (grid {grid:?}, block {block})", self.name))?;
        }
        Ok(())
    }
}

/// Device memory on the GPU that was current when it was allocated, freed on drop.
pub struct Buf {
    pub ptr: u64,
    pub bytes: usize,
    ctx: usize,
}

impl Buf {
    pub fn alloc(bytes: usize) -> Result<Buf> {
        let ctx = current()?;
        let mut p = 0;
        check(unsafe { cuMemAlloc_v2(&mut p, bytes.max(1)) }, &format!("allocating {:.1} MB", bytes as f64 / 1e6))?;
        Ok(Buf { ptr: p, bytes, ctx })
    }

    /// `bytes` at `ptr` inside a buffer owned elsewhere: never freed by itself (layer streaming
    /// points a layer's weights into a shared slot this way).
    pub fn view(ptr: u64, bytes: usize) -> Buf {
        Buf { ptr, bytes, ctx: 0 }
    }

    pub fn zeroed(bytes: usize) -> Result<Buf> {
        let b = Buf::alloc(bytes)?;
        check(unsafe { cuMemsetD8_v2(b.ptr, 0, bytes.max(1)) }, "memset")?;
        Ok(b)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Buf> {
        let b = Buf::alloc(data.len())?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn upload(&self, data: &[u8]) -> Result<()> {
        self.upload_at(0, data)
    }

    pub fn upload_at(&self, offset: usize, data: &[u8]) -> Result<()> {
        assert!(offset + data.len() <= self.bytes);
        // a view has no context of its own: the caller has bound its GPU
        if self.ctx != 0 {
            make_current(self.ctx)?;
        }
        check(unsafe { cuMemcpyHtoD_v2(self.ptr + offset as u64, data.as_ptr() as *const c_void, data.len()) }, "upload")
    }

    pub fn download_f32(&self, n: usize) -> Result<Vec<f32>> {
        assert!(n * 4 <= self.bytes);
        make_current(self.ctx)?;
        let mut v = vec![0f32; n];
        check(unsafe { cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.ptr, n * 4) }, "download")?;
        Ok(v)
    }

    pub fn download_i32(&self, n: usize) -> Result<Vec<i32>> {
        assert!(n * 4 <= self.bytes);
        make_current(self.ctx)?;
        let mut v = vec![0i32; n];
        check(unsafe { cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.ptr, n * 4) }, "download")?;
        Ok(v)
    }

    /// `n` bytes from `offset`.
    /// Fill `out` from this buffer at `offset`.
    pub fn download_into(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        assert!(offset + out.len() <= self.bytes);
        make_current(self.ctx)?;
        check(unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.ptr + offset as u64, out.len()) }, "download")
    }

    pub fn download_at(&self, offset: usize, n: usize) -> Result<Vec<u8>> {
        assert!(offset + n <= self.bytes);
        make_current(self.ctx)?;
        let mut v = vec![0u8; n];
        check(unsafe { cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.ptr + offset as u64, n) }, "download")?;
        Ok(v)
    }

    /// Copy the first `bytes` of `src` to the start of this one.
    pub fn copy_from(&self, src: &Buf, bytes: usize) -> Result<()> {
        self.copy_range(0, src, 0, bytes)
    }

    /// Copy `bytes` from `src` at `src_off` to this buffer at `dst_off`. Between two GPUs the
    /// bytes go through the host: GeForce cards on Windows have no peer access, and the
    /// synchronous download also waits for the kernels that wrote them.
    pub fn copy_range(&self, dst_off: usize, src: &Buf, src_off: usize, bytes: usize) -> Result<()> {
        assert!(dst_off + bytes <= self.bytes && src_off + bytes <= src.bytes);
        if bytes == 0 {
            return Ok(());
        }
        if self.ctx == src.ctx {
            make_current(self.ctx)?;
            return check(unsafe { cuMemcpyDtoD_v2(self.ptr + dst_off as u64, src.ptr + src_off as u64, bytes) }, "device copy");
        }
        make_current(src.ctx)?;
        let mut v = vec![0u8; bytes];
        check(unsafe { cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, src.ptr + src_off as u64, bytes) }, "copy between GPUs")?;
        self.upload_at(dst_off, &v)
    }

    /// Zero the first `bytes` bytes.
    pub fn zero_bytes(&self, bytes: usize) -> Result<()> {
        assert!(bytes <= self.bytes);
        make_current(self.ctx)?;
        check(unsafe { cuMemsetD8_v2(self.ptr, 0, bytes.max(1)) }, "memset")
    }

    pub fn zero(&self) -> Result<()> {
        self.zero_bytes(self.bytes)
    }

    pub fn at(&self, byte_offset: usize) -> u64 {
        self.ptr + byte_offset as u64
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.ctx != 0 && make_current(self.ctx).is_ok() {
            unsafe { cuMemFree_v2(self.ptr) };
        }
    }
}
