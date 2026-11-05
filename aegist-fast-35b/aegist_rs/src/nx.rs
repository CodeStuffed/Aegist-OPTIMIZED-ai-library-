//! Reader for the .nx brain format (see aegist/nx.py for the writer).
//!
//!   bytes 0..8   magic "AEGISTNX"
//!   bytes 8..12  u32 LE format version
//!   bytes 12..20 u64 LE header length H, then H bytes of JSON header
//!   data section from the next 64-byte boundary; every tensor 64-byte aligned
//!
//! A cores file (a learned version of a brain too big to copy) has the same layout but holds only
//! the cores, with header "base": {"path", "sha256"} naming the brain it was learned on. Opening it
//! opens that base and puts the stored cores in place of the base's; everything else (experts,
//! embedding, the path the expert reader opens) is the base's. Its header (lineage, notice) wins.
use memmap2::Mmap;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;

pub const MAGIC: &[u8; 8] = b"AEGISTNX";
/// Files written before the rename to Aegist.
pub const LEGACY_MAGIC: &[u8; 8] = b"AUGISTNX";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
    pub group: String,
    /// stored in the cores file, not the base
    pub overlay: bool,
}

pub struct NxFile {
    pub path: String,
    pub header: Value,
    pub tensors: BTreeMap<String, TensorInfo>,
    data_start: usize,
    mmap: Mmap,
    /// a cores file over this base: (its path, its data start, its map, its group hashes)
    overlay: Option<(String, usize, Mmap, Value)>,
}

fn pad(n: usize) -> usize {
    (64 - n % 64) % 64
}

impl NxFile {
    pub fn open(path: &str) -> Result<NxFile, String> {
        let f = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| format!("cannot map {path}: {e}"))?;
        if mmap.len() < 20 || (&mmap[..8] != MAGIC && &mmap[..8] != LEGACY_MAGIC) {
            return Err(format!("{path} is not an .nx file"));
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version > VERSION {
            return Err(format!("{path} uses .nx format v{version}; this Aegist reads up to v{VERSION}"));
        }
        let hlen = u64::from_le_bytes(mmap[12..20].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&mmap[20..20 + hlen]).map_err(|e| format!("bad .nx header: {e}"))?;
        let mut tensors = BTreeMap::new();
        for (name, m) in header["tensors"].as_object().ok_or("header has no tensors")? {
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype: m["dtype"].as_str().unwrap_or("").to_string(),
                    shape: m["shape"].as_array().map(|a| a.iter().map(|d| d.as_u64().unwrap() as usize).collect()).unwrap_or_default(),
                    offset: m["offset"].as_u64().unwrap() as usize,
                    nbytes: m["nbytes"].as_u64().unwrap() as usize,
                    group: m["group"].as_str().unwrap_or("base").to_string(),
                    overlay: false,
                },
            );
        }
        let data_start = 20 + hlen + pad(20 + hlen);
        if header["base"].is_object() {
            return NxFile::over_base(path, header, tensors, data_start, mmap);
        }
        Ok(NxFile { path: path.to_string(), header, tensors, data_start, mmap, overlay: None })
    }

    fn over_base(path: &str, header: Value, tensors: BTreeMap<String, TensorInfo>, data_start: usize, mmap: Mmap) -> Result<NxFile, String> {
        let base_path = header["base"]["path"].as_str().ok_or(format!("{path}: base has no path"))?.to_string();
        let mut base = NxFile::open(&base_path).map_err(|e| format!("{path} is learned cores over {base_path}: {e}"))?;
        if base.overlay.is_some() {
            return Err(format!("{path}: its base {base_path} is itself a cores file"));
        }
        if base.header["sha256"]["base"] != header["base"]["sha256"] {
            return Err(format!("{path} was learned on a different {base_path} (its weights changed since)"));
        }
        for (name, mut t) in tensors {
            if t.group != "cores" {
                return Err(format!("{path}: a cores file may only hold cores, not {name}"));
            }
            match base.tensors.get(&name) {
                Some(b) if b.dtype == t.dtype && b.shape.get(1..) == t.shape.get(1..) => {}
                _ => return Err(format!("{path}: {name} does not fit {base_path}")),
            }
            t.overlay = true;
            base.tensors.insert(name, t);
        }
        for k in ["lineage", "notice", "created", "version"] {
            if !header[k].is_null() {
                base.header[k] = header[k].clone();
            }
        }
        let hashes = header["sha256"].clone();
        base.overlay = Some((path.to_string(), data_start, mmap, hashes));
        Ok(base)
    }

    /// Names the weights in use (the base's group hashes and any cores file's), for caches of
    /// what the brain computed. None when the header has no hashes.
    pub fn fingerprint(&self) -> Option<String> {
        if !self.header["sha256"].is_object() {
            return None;
        }
        let mut h = Sha256::new();
        h.update(self.header["sha256"].to_string());
        if let Some((_, _, _, hashes)) = &self.overlay {
            h.update(hashes.to_string());
        }
        Some(h.finalize()[..12].iter().map(|b| format!("{b:02x}")).collect())
    }

    fn data(&self, t: &TensorInfo) -> &[u8] {
        match (&self.overlay, t.overlay) {
            (Some((_, start, m, _)), true) => &m[start + t.offset..start + t.offset + t.nbytes],
            _ => &self.mmap[self.data_start + t.offset..self.data_start + t.offset + t.nbytes],
        }
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors.get(name).ok_or_else(|| format!("{} has no tensor {name}", self.path))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8], String> {
        Ok(self.data(self.info(name)?))
    }

    /// Where a tensor's bytes start in the mapped base file, for callers that slice it repeatedly.
    pub fn offset_of(&self, name: &str) -> Result<usize, String> {
        let t = self.info(name)?;
        if t.overlay {
            return Err(format!("{name} is in a cores file; read it with bytes()"));
        }
        Ok(self.data_start + t.offset)
    }

    /// `len` bytes of the mapped file from absolute offset `start`.
    pub fn raw(&self, start: usize, len: usize) -> &[u8] {
        &self.mmap[start..start + len]
    }

    pub fn blob(&self, name: &str) -> Result<&[u8], String> {
        let b = &self.header["blobs"][name];
        let off = b["offset"].as_u64().ok_or(format!("no blob {name}"))? as usize;
        let n = b["nbytes"].as_u64().unwrap() as usize;
        Ok(&self.mmap[self.data_start + off..self.data_start + off + n])
    }

    /// Recompute each group's SHA-256 and compare with the header. Order: (group, name). With a
    /// cores file, the cores are checked against its hashes and the rest against the base's.
    pub fn verify(&self) -> BTreeMap<String, bool> {
        let mut order: Vec<(&String, &TensorInfo)> = self.tensors.iter().collect();
        order.sort_by(|a, b| (&a.1.group, a.0).cmp(&(&b.1.group, b.0)));
        let mut hashes: BTreeMap<String, Sha256> = ["base", "engine", "cores"].iter().map(|g| (g.to_string(), Sha256::new())).collect();
        for (_, t) in order {
            if let Some(h) = hashes.get_mut(&t.group) {
                h.update(self.data(t));
            }
        }
        hashes
            .into_iter()
            .map(|(g, h)| {
                let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
                let want = match &self.overlay {
                    Some((_, _, _, oh)) if g == "cores" => &oh[&g],
                    _ => &self.header["sha256"][&g],
                };
                (g, want.as_str() == Some(hex.as_str()))
            })
            .collect()
    }
}
