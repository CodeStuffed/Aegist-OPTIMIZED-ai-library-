//! Parallel reads of fixed-size blocks (experts) from a brain file.
//!
//! Faulting experts in through the memory map runs at about 2 GB/s on an NVMe drive that reads
//! 4-5 GB/s: each 4 KB page fault is handled one at a time, then mapped into the working set.
//! Here a few threads, each with its own file handle (reads on one handle are serialised), read
//! whole blocks with ordinary cached reads into reused buffers, while the calling thread hands
//! finished blocks to `sink` (the GPU copy). Cached reads keep the OS file cache working as a RAM
//! tier: a block read recently comes back at memory speed.
use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};

pub struct Reader {
    files: Vec<File>,
    bufs: Vec<Vec<u8>>,
}

#[cfg(windows)]
fn read_at(f: &File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, off)
}

impl Reader {
    /// `threads` handles on `path` ($AEGIST_READ_THREADS overrides; at least 1).
    pub fn open(path: &str, threads: usize) -> Result<Reader, String> {
        let threads = std::env::var("AEGIST_READ_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(threads).max(1);
        let files = (0..threads).map(|_| File::open(path).map_err(|e| format!("cannot open {path}: {e}"))).collect::<Result<_, _>>()?;
        Ok(Reader { files, bufs: vec![] })
    }

    /// Read `len` bytes at each of `jobs`' (tag, file offset) and call `sink(tag, bytes)` for each,
    /// on this thread, in completion order. Stops at the first error.
    pub fn load(&mut self, jobs: &[(usize, usize)], len: usize, mut sink: impl FnMut(usize, &[u8]) -> Result<(), String>) -> Result<(), String> {
        if jobs.is_empty() {
            return Ok(());
        }
        let w = self.files.len().min(jobs.len());
        // two buffers per reader: one being read, one waiting for the copy
        while self.bufs.len() < 2 * w {
            self.bufs.push(vec![0u8; len]);
        }
        for b in &mut self.bufs {
            b.resize(len, 0);
        }
        if w == 1 {
            let b = &mut self.bufs[0];
            for &(tag, off) in jobs {
                read_at(&self.files[0], b, off as u64).map_err(|e| format!("reading the brain at {off}: {e}"))?;
                sink(tag, b)?;
            }
            return Ok(());
        }
        let (free_tx, free_rx) = mpsc::channel::<Vec<u8>>();
        for b in self.bufs.drain(..) {
            free_tx.send(b).unwrap();
        }
        let free_rx = Mutex::new(free_rx);
        let next = AtomicUsize::new(0);
        let mut err: Option<String> = None;
        std::thread::scope(|s| {
            let (done_tx, done_rx) = mpsc::channel::<(usize, Vec<u8>, std::io::Result<()>)>();
            for f in &self.files[..w] {
                let (done_tx, free_rx, next) = (done_tx.clone(), &free_rx, &next);
                s.spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        return;
                    }
                    let Ok(mut b) = free_rx.lock().unwrap().recv() else { return };
                    let r = read_at(f, &mut b, jobs[i].1 as u64);
                    if done_tx.send((i, b, r)).is_err() {
                        return;
                    }
                });
            }
            drop(done_tx);
            // ends when every reader has stopped; each buffer goes back for the next read
            for (i, b, r) in done_rx.iter() {
                if err.is_none() {
                    let r = r.map_err(|e| format!("reading the brain at {}: {e}", jobs[i].1)).and_then(|_| sink(jobs[i].0, &b));
                    if let Err(e) = r {
                        err = Some(e);
                        next.store(jobs.len(), Ordering::Relaxed);
                    }
                }
                let _ = free_tx.send(b);
            }
        });
        drop(free_tx);
        self.bufs.extend(free_rx.into_inner().unwrap().try_iter());
        err.map_or(Ok(()), Err)
    }
}

/// Background reads that only warm the OS file cache: blocks a later `Reader::load` will
/// probably ask for (the next layer's likely experts) are read while the disk would otherwise
/// sit idle (GPU work, host syncs), so that load finds them in RAM. The bytes are thrown away;
/// what is read later is the same either way. `want` replaces the queue (blocks not started are
/// dropped); `settle` drops the blocks not started (the load reads them itself, in parallel) and
/// waits for the reads in flight. `warm` queues blocks read only while the engine is idle (no
/// `want` or `settle` for IDLE): the experts used most last time, while the user types.
pub struct Prefetcher {
    shared: std::sync::Arc<(Mutex<PreQueue>, std::sync::Condvar)>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

const IDLE: std::time::Duration = std::time::Duration::from_millis(300);

#[derive(Default)]
struct PreQueue {
    queue: std::collections::VecDeque<u64>,
    warm: std::collections::VecDeque<u64>,
    active: Option<std::time::Instant>,
    busy: usize,
    stop: bool,
    /// Warm-up paused: another brain in this process is using the disk.
    held: bool,
    /// blocks read so far
    done: u64,
}

impl Prefetcher {
    pub fn open(path: &str, threads: usize, len: usize) -> Result<Prefetcher, String> {
        let shared = std::sync::Arc::new((Mutex::new(PreQueue::default()), std::sync::Condvar::new()));
        let mut hs = vec![];
        for _ in 0..threads.max(1) {
            let f = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
            let sh = shared.clone();
            hs.push(std::thread::spawn(move || {
                let mut buf = vec![0u8; len];
                let (m, cv) = &*sh;
                loop {
                    let off = {
                        let mut q = m.lock().unwrap();
                        loop {
                            if q.stop {
                                return;
                            }
                            if let Some(o) = q.queue.pop_front() {
                                q.busy += 1;
                                break o;
                            }
                            let idle = !q.held && q.active.map_or(true, |t| t.elapsed() >= IDLE);
                            if idle {
                                if let Some(o) = q.warm.pop_front() {
                                    q.busy += 1;
                                    break o;
                                }
                            }
                            q = if q.warm.is_empty() { cv.wait(q).unwrap() } else { cv.wait_timeout(q, IDLE / 3).unwrap().0 };
                        }
                    };
                    // an error only means the later load reads it itself
                    let _ = read_at(&f, &mut buf, off);
                    let mut q = m.lock().unwrap();
                    q.busy -= 1;
                    q.done += 1;
                    cv.notify_all();
                }
            }));
        }
        Ok(Prefetcher { shared, threads: hs })
    }

    pub fn want(&self, offsets: impl IntoIterator<Item = u64>) {
        let (m, cv) = &*self.shared;
        let mut q = m.lock().unwrap();
        q.active = Some(std::time::Instant::now());
        q.queue.clear();
        q.queue.extend(offsets);
        cv.notify_all();
    }

    pub fn settle(&self) {
        let (m, cv) = &*self.shared;
        let mut q = m.lock().unwrap();
        q.active = Some(std::time::Instant::now());
        q.queue.clear();
        while q.busy > 0 {
            q = cv.wait(q).unwrap();
        }
    }

    pub fn warm(&self, offsets: impl IntoIterator<Item = u64>) {
        let (m, cv) = &*self.shared;
        let mut q = m.lock().unwrap();
        q.warm.clear();
        q.warm.extend(offsets);
        cv.notify_all();
    }

    /// Pause (true) or resume the warm-up reads; reads already asked for with `want` go on.
    pub fn hold(&self, on: bool) {
        let (m, cv) = &*self.shared;
        let mut q = m.lock().unwrap();
        q.held = on;
        q.active = Some(std::time::Instant::now());
        cv.notify_all();
    }

    pub fn done(&self) -> u64 {
        self.shared.0.lock().unwrap().done
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        let (m, cv) = &*self.shared;
        m.lock().unwrap().stop = true;
        cv.notify_all();
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_every_block_once() {
        let path = std::env::temp_dir().join(format!("aegist-reader-{}.bin", std::process::id()));
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let len = 1000;
        let jobs: Vec<(usize, usize)> = (0..250).map(|i| (i, (i * 1177) % (data.len() - len))).collect();
        for threads in [1, 3, 8] {
            let mut r = Reader::open(path.to_str().unwrap(), threads).unwrap();
            let mut seen = vec![0; jobs.len()];
            r.load(&jobs, len, |tag, b| {
                seen[tag] += 1;
                assert_eq!(b, &data[jobs[tag].1..jobs[tag].1 + len]);
                Ok(())
            })
            .unwrap();
            assert!(seen.iter().all(|&n| n == 1), "{threads} threads");
            // a failing sink stops the load and is reported
            let mut n = 0;
            let e = r.load(&jobs, len, |_, _| {
                n += 1;
                if n == 5 { Err("stop".into()) } else { Ok(()) }
            });
            assert_eq!(e, Err("stop".into()));
            assert_eq!(r.bufs.len(), 2 * threads, "buffers come back after a failed load");
        }
        std::fs::remove_file(&path).ok();
    }
}
