//! aegist.exe: the launcher and fast engine for Aegist .nx brains.
//!
//!   aegist                      launcher menu
//!   aegist chat [--mode blunt] [--think auto|on|off] [--context N] [--brain FILE]
//!   aegist ask "question" [--mode M] [--think T] [--seed N] [--temp T]   one reply, then exit
//!   aegist learn [args]         learning time (runs python -m aegist.learn)
//!   aegist study [status|lesson|exam|run] [args]   the curriculum (python -m aegist.curriculum)
//!   aegist lang design|new|show|translate|check|lessons|exam [args]   the language workbench
//!   aegist aq [--full] [--seed S] [--brain FILE]   the AQ test: checkable questions, a score out of 100
//!   aegist tricky [--no-web] [--think auto|on|off]   50 tricky questions (riddles, made-up things, exact instructions)
//!   aegist exam in.json out.json [--brain FILE] [--no-tools] [--no-second-look]   questions through chat (for aq)
//!   chat and ask also take --no-tools and --no-second-look
//!   aegist lanes BRAIN in.json out.bin --layer L   hidden lanes going into layer L (177B learning time)
//!   aegist score BRAIN in.json out.json     loss of given answers (checks learned cores on the engine)
//!   aegist status | rollback    brain versions
//!   aegist info [FILE] | verify [FILE] | about [--license]
//!   aegist register | unregister   open .nx files with Aegist (current user only)
mod chat;
mod cuda;
mod governor;
mod hybrid;
mod kimi;
mod laya;
mod max;
mod model;
mod next;
mod nx;
mod persona;
mod probe;
mod reader;
mod shield;
mod tokenizer;
mod tools;

use chat::{Chat, Event, GenConfig, Session, TurnStats};
use model::Lm;
use nx::NxFile;
use serde_json::{json, Value};
use std::io::{BufRead, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

#[cfg(windows)]
fn enable_ansi() {
    extern "system" {
        fn GetStdHandle(n: u32) -> *mut std::ffi::c_void;
        fn GetConsoleMode(h: *mut std::ffi::c_void, m: *mut u32) -> i32;
        fn SetConsoleMode(h: *mut std::ffi::c_void, m: u32) -> i32;
    }
    unsafe {
        let h = GetStdHandle(-11i32 as u32);
        let mut m = 0;
        if GetConsoleMode(h, &mut m) != 0 {
            SetConsoleMode(h, m | 0x0004); // ENABLE_VIRTUAL_TERMINAL_PROCESSING
        }
    }
}
#[cfg(not(windows))]
fn enable_ansi() {}

// --- where things live ------------------------------------------------------------

/// The Aegist folder: $AEGIST_HOME, else the nearest folder with brains/ above the current
/// directory or above aegist.exe. If that folder's aegist.json says {"home": "..."}, Aegist lives
/// there instead (so the code can stay in one place and the brains on a bigger drive).
fn home() -> PathBuf {
    if let Ok(h) = std::env::var("AEGIST_HOME") {
        return PathBuf::from(h);
    }
    let starts = [std::env::current_dir().ok(), std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf))];
    for s in starts.into_iter().flatten() {
        for d in s.ancestors() {
            if d.join("brains").is_dir() {
                return match setting(d, "home").map(PathBuf::from) {
                    Some(h) if h.join("brains").is_dir() => h,
                    _ => d.to_path_buf(),
                };
            }
        }
    }
    std::env::current_dir().unwrap_or_default()
}

/// A string setting from aegist.json in `dir`.
fn setting(dir: &Path, key: &str) -> Option<String> {
    let s = std::fs::read_to_string(dir.join("aegist.json")).ok()?;
    serde_json::from_str::<Value>(&s).ok()?[key].as_str().map(str::to_string)
}

/// The folder that holds the aegist Python package: "code" in aegist.json, else the one above
/// aegist.exe that has aegist/learn.py in it, else `home`.
fn code_dir(home: &Path) -> PathBuf {
    if let Some(c) = setting(home, "code") {
        return PathBuf::from(c);
    }
    let exe = std::env::current_exe().ok();
    exe.iter()
        .flat_map(|p| p.ancestors())
        .find(|d| d.join("aegist").join("learn.py").is_file())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| home.to_path_buf())
}

fn current_brain(home: &Path) -> PathBuf {
    let brains = home.join("brains");
    if let Ok(rel) = std::fs::read_to_string(brains.join("current.txt")) {
        let p = brains.join(rel.trim());
        if p.exists() {
            return p;
        }
    }
    // the 35B brain, under its name from before the rename if that is what is there
    ["aegist-nx-35b.nx", "aegist.nx"].iter().map(|n| brains.join(n)).find(|p| p.exists())
        .unwrap_or_else(|| brains.join("aegist-nx-35b.nx"))
}

fn python(home: &Path) -> String {
    if let Ok(p) = std::env::var("AEGIST_PYTHON") {
        return p;
    }
    setting(home, "python").unwrap_or_else(|| "python".into())
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn stamp() -> String {
    // UTC yyyymmdd-hhmmss without a date crate (civil-from-days, Hinnant)
    let t = now_secs() as i64;
    let (days, sod) = (t.div_euclid(86400), t.rem_euclid(86400));
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", sod / 3600, sod / 60 % 60, sod % 60)
}

// --- arguments ----------------------------------------------------------------------

struct Args {
    pos: Vec<String>,
    opts: Vec<(String, String)>,
    flags: Vec<String>,
}

fn parse(args: &[String], with_value: &[&str]) -> Args {
    let mut a = Args { pos: vec![], opts: vec![], flags: vec![] };
    let mut i = 0;
    while i < args.len() {
        let s = &args[i];
        if let Some(name) = s.strip_prefix("--") {
            if with_value.contains(&name) && i + 1 < args.len() {
                a.opts.push((name.into(), args[i + 1].clone()));
                i += 1;
            } else {
                a.flags.push(name.into());
            }
        } else {
            a.pos.push(s.clone());
        }
        i += 1;
    }
    a
}

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.opts.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
}

// --- loading ----------------------------------------------------------------------

fn arch(nx: &NxFile) -> String {
    nx.header["config"]["arch"].as_str().unwrap_or("dense").to_string()
}

/// The engine for the brain's architecture. The hybrid engine keeps the file open (it reads
/// experts and embedding rows from it while running); the dense one copies everything to VRAM.
fn load_engine(nx: NxFile, context: usize, progress: impl FnMut(usize, usize)) -> Result<Box<dyn Lm>, String> {
    if kimi::is_kimi(&nx.header["config"]) {
        Ok(Box::new(kimi::KimiEngine::load(nx, context, progress)?))
    } else if max::is_max(&nx.header["config"]) {
        Ok(Box::new(max::MaxEngine::load(nx, context, progress)?))
    } else if arch(&nx) == "hybrid-moe" && next::is_next(&nx.header["config"]) {
        Ok(Box::new(next::NextEngine::load(nx, context, progress)?))
    } else if arch(&nx) == "hybrid-moe" {
        Ok(Box::new(hybrid::HybridEngine::load(nx, context, progress)?))
    } else {
        Ok(Box::new(model::Engine::load(&nx, context, progress)?))
    }
}

/// `probe`: a confidence probe to mark unsure sentences with (a probe that does not fit the
/// brain is reported and left out).
/// The fast brain that answers everyday prompts beside the big one: brains/aegist-nx-35b.nx
/// ($AEGIST_FAST_BRAIN another file; $AEGIST_FAST=off none), with its own probe.
fn fast_brain(home: &Path, brain: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    if std::env::var("AEGIST_FAST").is_ok_and(|v| ["off", "0", "no", "false"].contains(&v.to_lowercase().as_str())) {
        return None;
    }
    let p = std::env::var("AEGIST_FAST_BRAIN").map(PathBuf::from).unwrap_or_else(|_| home.join("brains").join("aegist-nx-35b.nx"));
    let same = std::fs::canonicalize(&p).ok().zip(std::fs::canonicalize(brain).ok()).is_some_and(|(a, b)| a == b);
    (p.exists() && !same).then(|| {
        let probe = probe::find(home, &p);
        (p, probe)
    })
}

/// GB of VRAM the fast brain's expert slots take on its GPU ($AEGIST_FAST_GB).
const FAST_GB: f64 = 8.0;

/// `brain` with `fast` loaded first beside it (on the last GPU, with a small expert pool, so the
/// big brain sizes its pool to what is left).
fn load_brains(brain: &Path, context: usize, probe: Option<PathBuf>, fast: Option<(PathBuf, Option<PathBuf>)>) -> Result<Chat, String> {
    let fast = match fast {
        Some((fp, fprobe)) => {
            let gpu = cuda::device_count().unwrap_or(1).max(1) - 1;
            let gb = std::env::var("AEGIST_FAST_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(FAST_GB);
            let mut f = load_one(&fp, context, fprobe, Some((gpu, (gb * 1e9) as usize)))?;
            // its unsure answers go to the big brain instead of a second look of its own
            f.second_look = false;
            Some(Box::new(f))
        }
        None => None,
    };
    let mut chat = load_one(brain, context, probe, None)?;
    chat.fast = fast;
    Ok(chat)
}

/// One brain; `on`: (GPU, expert pool bytes) for a hybrid brain loaded as the fast one.
fn load_one(brain: &Path, context: usize, probe: Option<PathBuf>, on: Option<(usize, usize)>) -> Result<Chat, String> {
    let path = brain.to_string_lossy().to_string();
    let nx = NxFile::open(&path)?;
    let name = nx.header["name"].as_str().unwrap_or("").to_string();
    let tok = tokenizer::Tok::from_json(nx.blob("tokenizer.json")?)?;
    let gen = GenConfig::from_header(&nx.header["generation"], &arch(&nx));
    let dims = |k: &str| nx.header["config"][k].as_u64().unwrap_or(0) as usize;
    let (d_model, n_layers) = (dims("d_model"), dims("n_layers"));
    let t0 = Instant::now();
    eprint!("loading {} ...", brain.file_name().unwrap_or_default().to_string_lossy());
    let mut last = 0;
    let progress = |done: usize, total: usize| {
        let pct = done * 100 / total.max(1);
        if pct >= last + 10 {
            last = pct;
            eprint!(" {pct}%");
        }
    };
    let engine: Box<dyn Lm> = match on {
        Some((gpu, bytes)) if arch(&nx) == "hybrid-moe" && !next::is_next(&nx.header["config"]) => {
            Box::new(hybrid::HybridEngine::load_on(nx, context, gpu, Some(bytes), progress)?)
        }
        _ => load_engine(nx, context, progress)?,
    };
    eprintln!(
        "\r{} on {} ({:.1}s){}",
        brain.file_name().unwrap_or_default().to_string_lossy(),
        engine.describe(),
        t0.elapsed().as_secs_f64(),
        " ".repeat(20)
    );
    let mut chat = Chat::new(engine, tok, gen);
    chat.name = name;
    let argv: Vec<String> = std::env::args().collect();
    chat.tools = !argv.iter().any(|a| a == "--no-tools");
    chat.second_look = !argv.iter().any(|a| a == "--no-second-look");
    // the brain's state after the shared start of the prompt, kept on disk ($AEGIST_STATES=off: not)
    let off = std::env::var("AEGIST_STATES").is_ok_and(|v| ["off", "0", "no", "false"].contains(&v.to_lowercase().as_str()));
    if !off {
        chat.state_dir = Some(home().join("states"));
    }
    // the reading core: Laya in Rust on the CPU, so it takes nothing from the GPU
    match laya::find(&home()).map(|d| laya::Laya::open(&d)) {
        Some(Ok(r)) => chat.reader = Some(r),
        Some(Err(e)) => eprintln!("{YELLOW}reading core not used: {e}{RESET}"),
        None => {}
    }
    if let Some(p) = probe {
        match probe::Probe::load(&p, d_model, n_layers) {
            Ok(pr) => chat.set_probe(Some(pr))?,
            Err(e) => eprintln!("{YELLOW}confidence probe not used: {e}{RESET}"),
        }
    }
    Ok(chat)
}

// --- chat ---------------------------------------------------------------------------

const CHAT_HELP: &str = "commands:
  /mode normal|blunt|idea  switch mode        /think auto|on|off   thinking
  /brain auto|fast|big which brain answers
  /good                last reply was good    /bad                 last reply was bad
  /fix <better reply>  teach the right reply  /teach <text>        add a note to learn from
  /remember <fact>     remember for this chat /learn               learning time now
  /reset               new conversation       /stats               last reply's numbers
  /unsure on|off       mark likely-wrong sentences [unsure: N% chance this is wrong]
  /governor            how much it is allowed to make things up, and why
  /help                this help              /quit                leave";

enum ChatExit {
    Quit,
    Learn,
}

/// `unsure`: the reply's sentences the probe marked, so learning time can check the probe against the rating.
fn feedback(home: &Path, s: &Session, rating: &str, correction: Option<&str>, unsure: &[(String, f32)]) -> Result<(), String> {
    let n = s.turns.len();
    if n < 2 {
        return Err("no reply to rate yet".into());
    }
    let prior: Vec<[&str; 2]> = s.turns[..n - 1].iter().map(|(r, t)| [r.as_str(), t.as_str()]).collect();
    let mut rec = json!({"mode": s.mode, "turns": prior, "reply": s.turns[n - 1].1, "rating": rating,
                         "unsure": unsure.iter().map(|(t, p)| json!({"sentence": t, "p_wrong": p})).collect::<Vec<_>>()});
    if let Some(c) = correction {
        rec["correction"] = json!(c);
    }
    let dir = home.join("learn");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("feedback.jsonl")).map_err(|e| e.to_string())?;
    writeln!(f, "{rec}").map_err(|e| e.to_string())
}

fn chat_loop(home: &Path, chat: &mut Chat, session: &mut Session) -> ChatExit {
    let stdin = std::io::stdin();
    let mut last_stats: Option<chat::TurnStats> = None;
    let mut stashed_probe: Option<probe::Probe> = None;
    let marks = match &chat.probe {
        Some(p) => match p.metrics["test_auroc"].as_f64() {
            Some(a) => format!("on, probe test AUROC {a:.2}"),
            None => "on".into(),
        },
        None => "off (no probe yet)".into(),
    };
    let reader = if chat.reader.is_some() { "on" } else { "off (Laya not downloaded)" };
    println!("{DIM}Aegist ({} mode, thinking {}, unsure marks {marks}, reading core {reader}). /help for commands.{RESET}", session.mode, session.think);
    loop {
        print!("\n{BOLD}you>{RESET} ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            return ChatExit::Quit;
        }
        let text = line.trim_end_matches(['\r', '\n']).to_string();
        if text.trim().is_empty() {
            continue;
        }
        if let Some(cmd) = text.strip_prefix('/') {
            let (name, arg) = cmd.split_once(' ').map(|(a, b)| (a, b.trim())).unwrap_or((cmd, ""));
            let say = |m: &str| println!("{DIM}{m}{RESET}");
            let marked: &[(String, f32)] = last_stats.as_ref().map_or(&[], |s| s.unsure.as_slice());
            match name {
                "quit" | "exit" | "q" => return ChatExit::Quit,
                "help" | "?" => say(CHAT_HELP),
                "mode" if matches!(arg, "normal" | "blunt" | "idea") => {
                    session.mode = arg.into();
                    say(&format!("mode: {arg}"));
                }
                "think" if matches!(arg, "auto" | "on" | "off") => {
                    session.think = arg.into();
                    say(&format!("thinking: {arg}"));
                }
                "brain" if matches!(arg, "auto" | "big" | "fast") => {
                    session.brain = arg.into();
                    say(&match (&chat.fast, arg) {
                        (None, _) => "only one brain is loaded".to_string(),
                        (Some(_), "auto") => "brain: auto (everyday prompts on the fast brain; hard ones and its unsure answers on the big one)".into(),
                        (Some(f), "fast") => format!("brain: always the fast one ({})", f.name),
                        _ => format!("brain: always the big one ({})", chat.name),
                    });
                }
                "good" => say(&feedback(home, session, "good", None, marked).map(|_| "noted: good reply (learned at the next learning time)".into()).unwrap_or_else(|e| e)),
                "bad" => say(&feedback(home, session, "bad", None, marked).map(|_| "noted: bad reply. Use /fix <better reply> so learning time has something to learn from.".into()).unwrap_or_else(|e| e)),
                "fix" if !arg.is_empty() => say(&feedback(home, session, "bad", Some(arg), marked).map(|_| "noted: correction saved for learning time".into()).unwrap_or_else(|e| e)),
                "teach" if !arg.is_empty() => {
                    let dir = home.join("learn").join("inbox");
                    let path = dir.join(format!("taught-{}.txt", stamp()));
                    let r = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, arg));
                    say(&match r {
                        Ok(_) => format!("saved to {} for learning time", path.display()),
                        Err(e) => format!("could not save: {e}"),
                    });
                }
                "unsure" if matches!(arg, "on" | "off") => {
                    let r = if arg == "off" {
                        stashed_probe = chat.probe.take().or(stashed_probe.take());
                        chat.set_probe(None).map(|_| "unsure marks off".to_string())
                    } else if let Some(p) = stashed_probe.take().or(chat.probe.take()) {
                        let path = p.path.clone();
                        chat.set_probe(Some(p)).map(|_| format!("unsure marks on ({path})"))
                    } else {
                        Ok("no confidence probe yet (python -m aegist.confidence trains one)".into())
                    };
                    say(&r.unwrap_or_else(|e| e));
                }
                "governor" => {
                    say(&chat.governor.describe(chat.gen.temperature));
                    for l in chat.governor.log.iter().rev().take(5) {
                        say(l);
                    }
                }
                "remember" if !arg.is_empty() => {
                    session.memory.push(arg.into());
                    say("remembered for this conversation");
                }
                "learn" => return ChatExit::Learn,
                "reset" => {
                    session.turns.clear();
                    chat.reset();
                    say("new conversation");
                }
                "stats" => match &last_stats {
                    Some(s) => say(&format!(
                        "prompt {} tokens ({} reused), thinking {}, answer {}, {:.1}s total, prefill {:.2}s, {:.1} tok/s, {} cores used, {}, read as {} (fact {:.0}%, creative {:.0}%, other {:.0}%), {} of {} sentences marked unsure, highest chance of a mistake {:.0}%{}, {} not judged",
                        s.prompt_tokens, s.reused_tokens, s.thinking_tokens, s.answer_tokens, s.seconds, s.prefill_seconds,
                        s.tok_per_s(), s.cores_used, s.think_plan, s.kind.map_or("-", |k| k.name()), 100.0 * s.kind_p[0],
                        100.0 * s.kind_p[1], 100.0 * s.kind_p[2], s.sentences_unsure, s.sentences_scored, 100.0 * s.highest_unsure,
                        if s.cooled { ", cooled to greedy after the first unsure sentence" } else { "" }, s.sentences_unjudged
                    )),
                    None => say("no reply yet"),
                },
                _ => say("unknown command; /help lists them"),
            }
            continue;
        }
        let seed = now_secs() ^ (session.turns.len() as u64) << 32;
        let mut in_think = false;
        let res = chat.reply(session, &text, seed, |ev| {
            let mut out = std::io::stdout();
            match ev {
                Event::Shield(s) => {
                    let _ = writeln!(out, "{YELLOW}{s}{RESET}");
                }
                Event::Think(t) => {
                    if !in_think {
                        in_think = true;
                        let _ = write!(out, "{DIM}thinking: ");
                    }
                    let _ = write!(out, "{t}");
                }
                Event::Answer(t) => {
                    if in_think {
                        in_think = false;
                        let _ = write!(out, "{RESET}\n");
                    }
                    let _ = write!(out, "{t}");
                }
                Event::Unsure(p) => {
                    let _ = write!(out, " {YELLOW}[unsure: {:.0}% chance this is wrong]{RESET}", 100.0 * p);
                }
                Event::Read(k, p, how) => {
                    let _ = writeln!(out, "{DIM}[read as {} ({:.0}%): {how}]{RESET}", k.name(), 100.0 * p);
                }
                Event::UnsureSummary(t) => {
                    let _ = write!(out, "\n\n{YELLOW}{t}{RESET}");
                }
                Event::Phase(p) => {
                    let _ = write!(out, "{RESET}\n{CYAN}[{p}]{RESET}\n");
                }
                Event::Note(n) => {
                    let _ = write!(out, "\n{DIM}{n}{RESET}");
                }
                Event::Tool(t) => {
                    let _ = write!(out, "{DIM}[{t}]{RESET}\n");
                }
            }
            let _ = out.flush();
        });
        print!("{RESET}");
        match res {
            Ok(s) => {
                println!("\n{DIM}[{}{:.1} tok/s, {} cores{}]{RESET}", brain_note(&s), s.tok_per_s(), s.cores_used, unjudged_note(&s));
                last_stats = Some(s);
            }
            Err(e) => println!("\n{YELLOW}error: {e}{RESET}"),
        }
    }
}

fn run_chat(home: &Path, brain: Option<PathBuf>, mode: &str, think: &str, context: usize) -> Result<(), String> {
    let mut session = Session::new(mode);
    session.think = think.into();
    loop {
        let path = brain.clone().unwrap_or_else(|| current_brain(home));
        let fast = if brain.is_none() { fast_brain(home, &path) } else { None };
        let mut chat = load_brains(&path, context, probe::find(home, &path), fast)?;
        chat.governor = governor::Governor::load(home);
        if let Some(f) = chat.fast.as_mut() {
            f.governor = governor::Governor::load(home).detached();
        }
        chat.reset();
        match chat_loop(home, &mut chat, &mut session) {
            ChatExit::Quit => return Ok(()),
            ChatExit::Learn => {
                drop(chat); // free the GPU for learning time
                learn(home, &[])?;
                session.turns.clear();
            }
        }
    }
}

fn ask(home: &Path, a: &Args) -> Result<(), String> {
    let q = a.pos.get(1).ok_or("usage: aegist ask \"question\"")?;
    let brain = a.get("brain").map(PathBuf::from).unwrap_or_else(|| current_brain(home));
    let probe = if a.flag("no-unsure") { None } else { probe::find(home, &brain) };
    let fast = if a.get("brain").is_none() { fast_brain(home, &brain) } else { None };
    let mut chat = load_brains(&brain, a.get("context").and_then(|c| c.parse().ok()).unwrap_or(4096), probe, fast)?;
    // a temperature given here wins over the governor's (which is then not loaded or saved)
    match a.get("temp").and_then(|t| t.parse().ok()) {
        Some(t) => {
            chat.gen.temperature = t;
            chat.gen.think_temperature = t;
        }
        None => chat.governor = governor::Governor::load(home),
    }
    if let Some(f) = chat.fast.as_mut() {
        f.governor = governor::Governor::load(home).detached();
        f.gen.max_new_tokens = chat.gen.max_new_tokens;
    }
    if let Some(n) = a.get("max-new").and_then(|t| t.parse().ok()) {
        chat.gen.max_new_tokens = n;
    }
    let mut s = Session::new(a.get("mode").unwrap_or("normal"));
    s.think = a.get("think").unwrap_or("auto").into();
    s.brain = a.get("pick").unwrap_or("auto").into();
    let seed = a.get("seed").and_then(|x| x.parse().ok()).unwrap_or_else(now_secs);
    let stats = chat.reply(&mut s, q, seed, |ev| {
        match ev {
            Event::Shield(t) => eprintln!("{t}"),
            Event::Think(t) => eprint!("{DIM}{t}{RESET}"),
            Event::Answer(t) => print!("{t}"),
            Event::Unsure(p) => print!(" [unsure: {:.0}% chance this is wrong]", 100.0 * p),
            Event::Read(k, p, how) => eprintln!("{DIM}[read as {} ({:.0}%): {how}]{RESET}", k.name(), 100.0 * p),
            Event::UnsureSummary(t) => print!("\n\n{t}"),
            Event::Phase(p) => print!("\n[{p}]\n"),
            Event::Note(n) => eprintln!("\n{DIM}{n}{RESET}"),
            Event::Tool(t) => eprintln!("{DIM}[{t}]{RESET}"),
        }
        std::io::stdout().flush().ok();
    })?;
    println!();
    eprintln!(
        "{DIM}[{}prompt {} tokens, prefill {:.2}s; {} thinking + {} answer tokens at {:.1} tok/s; {} cores used{}]{RESET}",
        brain_note(&stats), stats.prompt_tokens, stats.prefill_seconds, stats.thinking_tokens, stats.answer_tokens, stats.tok_per_s(), stats.cores_used,
        unjudged_note(&stats)
    );
    Ok(())
}

/// "answered by NAME (why); " when two brains are loaded.
fn brain_note(s: &TurnStats) -> String {
    if s.brain.is_empty() { String::new() } else { format!("answered by {}; ", s.brain) }
}

/// Unmarked does not mean checked: say how many sentences the confidence check could not judge.
fn unjudged_note(s: &TurnStats) -> String {
    match s.sentences_unjudged {
        0 => String::new(),
        n => format!("; confidence check: {n} of {} sentences unlike its training, not judged", n + s.sentences_scored),
    }
}

// --- learning, versions, info -------------------------------------------------------

fn py_module(home: &Path, module: &str, extra: &[String]) -> Result<Option<i32>, String> {
    let py = python(home);
    let status = Command::new(&py)
        .args(["-m", module])
        .args(extra)
        .current_dir(home)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONPATH", code_dir(home))
        .status()
        .map_err(|e| format!("could not run {py}: {e}. Set AEGIST_PYTHON or \"python\" in aegist.json to the Python that has torch."))?;
    Ok(status.code())
}

/// The curriculum (aegist/curriculum.py) or the language workbench (aegist/conlang.py).
fn study(home: &Path, module: &str, extra: &[String]) -> Result<(), String> {
    // 2 is also argparse's usage error, so it means "rejected" only for a curriculum run
    let run = module == "aegist.curriculum" && extra.first().map(String::as_str) == Some("run");
    match py_module(home, module, extra)? {
        Some(0) => Ok(()),
        Some(2) if run => {
            println!("{DIM}(the new version did not pass the gate; nothing changed){RESET}");
            Ok(())
        }
        c => Err(format!("{module} exited with {c:?}")),
    }
}

/// The AQ test (aegist/aq.py): the brain answers checkable questions and gets a score.
fn aq(home: &Path, extra: &[String]) -> Result<(), String> {
    match py_module(home, "aegist.aq", extra)? {
        Some(0) => Ok(()),
        c => Err(format!("the AQ test exited with {c:?}")),
    }
}

fn learn(home: &Path, extra: &[String]) -> Result<(), String> {
    match py_module(home, "aegist.learn", extra)? {
        Some(0) => Ok(()),
        Some(2) => {
            println!("{DIM}(the new version did not pass the gate; nothing changed){RESET}");
            Ok(())
        }
        c => Err(format!("learning time exited with {c:?}")),
    }
}

fn info(path: &Path) -> Result<(), String> {
    let nx = NxFile::open(&path.to_string_lossy())?;
    let h = &nx.header;
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let c = &h["config"];
    let p = &h["params"];
    let silent = h["cores_silent"].as_array().map(|a| a.iter().filter(|x| x.as_bool() == Some(true)).count()).unwrap_or(0);
    println!("{BOLD}{}{RESET}  {}  ({:.2} GB, .nx v{})", h["name"].as_str().unwrap_or("?"), path.display(), size as f64 / 1e9, h["format_version"]);
    println!("created     {}", h["created"].as_str().unwrap_or("?"));
    let weights = match (h["quant"].as_str().unwrap_or("?"), h["dtype"].as_str().unwrap_or("?"), c["expert_format"].as_str()) {
        // the FP8 release: its experts stay FP8, so the brain is not all bf16
        ("none", d, Some(f)) if f != d => format!("experts {f} as released ({}), the rest {d}, nothing quantized further", h["experts"].as_str().unwrap_or("?")),
        ("none", d, _) => format!("{d}, not quantized"),
        (q, d, _) => format!("{q} ({d})"),
    };
    println!("layers {}  d_model {}  heads {}/{}  vocab {}  weights {weights}", c["n_layers"], c["d_model"], c["n_heads"], c["n_kv_heads"], c["vocab_size"]);
    let g = |k: &str| p[k].as_f64().unwrap_or(0.0) / 1e9;
    if c["arch"].as_str() == Some("hybrid-moe") {
        let n = c["n_layers"].as_u64().unwrap_or(0);
        let attn = c["layer_types"].as_array().map(|a| a.iter().filter(|t| t.as_str() == Some("attn")).count()).unwrap_or(0);
        let silent = (0..n).filter(|i| nx.info(&format!("blocks.{i}.cores.ids")).map(|t| t.shape.first().copied().unwrap_or(0) == 0).unwrap_or(true)).count();
        println!(
            "mixers      {} gated DeltaNet + {attn} gated attention; {} experts per layer, {} active, {} experts ({})",
            n as usize - attn, c["n_experts"], c["experts_active"], c["expert_hidden"], h["experts"].as_str().unwrap_or("?")
        );
        println!("parameters  {:.2}B stored: {:.2}B language core ({:.2}B active per token), {:.4}B core engine, {:.3}B in {} trained cores", g("total_stored"), g("base_total"), g("base_active_per_token"), g("engine"), g("cores_trained"), p["cores_trained_count"]);
        println!("growth      room for {:.1}B in cores ({} per row of {}, up to {} rows): {:.1}B capacity in all", g("core_capacity"), c["core_grid"][1], c["core_grid"][0], c["core_rows_max"], g("total_capacity"));
        println!("core banks  {} of {} layers still silent (no trained cores)", silent, n);
    } else {
        println!(
            "parameters  {:.2}B total: {:.2}B always on, {:.3}B in {} switchable cores ({} switched on per token per layer, {:.4}B active)",
            g("total"), g("base_always_on"), g("cores_total"), p["n_cores"], c["cores_active"], g("cores_active_per_token")
        );
        println!("core banks  {} of {} layers still silent (untrained)", silent, c["n_layers"]);
    }
    println!("lineage:");
    for e in h["lineage"].as_array().into_iter().flatten() {
        match e["event"].as_str() {
            Some("learned") => println!(
                "  {}  learned: {} files, {} feedback; held-out loss {} -> {}, drift {}",
                e["time"].as_str().unwrap_or(""), e["material"]["files"].as_array().map(|a| a.len()).unwrap_or(0),
                e["material"]["feedback"], e["held_out_loss"][0], e["held_out_loss"][1], e["probe_kl"][1]
            ),
            ev => println!("  {}  {}", e["time"].as_str().unwrap_or(""), ev.unwrap_or("?")),
        }
    }
    Ok(())
}

fn verify(path: &Path) -> Result<(), String> {
    let nx = NxFile::open(&path.to_string_lossy())?;
    let t0 = Instant::now();
    let r = nx.verify();
    for (g, ok) in &r {
        println!("{g:7} {}", if *ok { "ok" } else { "CHECKSUM MISMATCH" });
    }
    println!("{DIM}({:.1}s){RESET}", t0.elapsed().as_secs_f64());
    if r.values().all(|&v| v) { Ok(()) } else { Err("the brain file is damaged or was modified".into()) }
}

fn about(home: &Path, license: bool) -> Result<(), String> {
    let brain = current_brain(home);
    println!("{BOLD}Aegist{RESET} {} - Atech's local AI.", env!("CARGO_PKG_VERSION"));
    println!("\n{}", persona::IDENTITY);
    match NxFile::open(&brain.to_string_lossy()) {
        Ok(nx) => {
            println!("\nModel: {} ({})", nx.header["name"].as_str().unwrap_or("?"), brain.display());
            println!("\n{}", nx.header["notice"].as_str().unwrap_or(""));
            if license {
                println!("\n{}", String::from_utf8_lossy(nx.blob("LICENSE")?));
            } else {
                println!("{DIM}Full license text: aegist about --license{RESET}");
            }
        }
        Err(e) => println!("\n({e})"),
    }
    Ok(())
}

fn register(on: bool) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let reg = |args: &[&str]| -> Result<(), String> {
        let st = Command::new("reg").args(args).status().map_err(|e| e.to_string())?;
        if st.success() { Ok(()) } else { Err(format!("reg {} failed", args[0])) }
    };
    let cls = r"HKCU\Software\Classes";
    if on {
        let cmd = format!("\"{}\" chat --brain \"%1\"", exe.display());
        reg(&["add", &format!(r"{cls}\.nx"), "/ve", "/d", "Aegist.Brain", "/f"])?;
        reg(&["add", &format!(r"{cls}\Aegist.Brain"), "/ve", "/d", "Aegist brain", "/f"])?;
        reg(&["add", &format!(r"{cls}\Aegist.Brain\DefaultIcon"), "/ve", "/d", &format!("\"{}\",0", exe.display()), "/f"])?;
        reg(&["add", &format!(r"{cls}\Aegist.Brain\shell\open\command"), "/ve", "/d", &cmd, "/f"])?;
        println!(".nx files now open with {} (for this Windows user). Undo: aegist unregister", exe.display());
    } else {
        reg(&["delete", &format!(r"{cls}\.nx"), "/f"]).ok();
        reg(&["delete", &format!(r"{cls}\Aegist.Brain"), "/f"]).ok();
        println!(".nx association removed.");
    }
    Ok(())
}

// --- debugging / parity ---------------------------------------------------------------

/// in.json: {"prompt": [ids], "steps": [ids]} -> out.bin: f32 logits after the prompt and after
/// each step token, [(1 + steps) x vocab]. Used by tests/test_rust_parity.py.
fn debug_logits(a: &Args) -> Result<(), String> {
    let (brain, inp, out) = match &a.pos[1..] {
        [b, i, o] => (b, i, o),
        _ => return Err("usage: aegist debug-logits BRAIN in.json out.bin".into()),
    };
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(inp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let ids = |k: &str| -> Vec<u32> { spec[k].as_array().into_iter().flatten().map(|v| v.as_u64().unwrap() as u32).collect() };
    let nx = NxFile::open(brain)?;
    let mut e = load_engine(nx, a.get("context").and_then(|c| c.parse().ok()).unwrap_or(2048), |_, _| {})?;
    if let Some(ops) = spec["ops"].as_array() {
        // {"ops": [{"feed": [ids]} | {"snapshot": true} | {"crop": n}]}: one logits row per feed;
        // the lengths kept by each crop go to stderr as "crop n -> kept"
        // optional "taps": [layers]: each feed's last-token state after those layers goes to out.taps
        let taps: Vec<usize> = spec["taps"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|v| v as usize)).collect();
        e.set_taps(&taps)?;
        let mut tap_rows: Vec<f32> = vec![];
        let mut rows: Vec<f32> = vec![];
        for op in ops {
            if let Some(f) = op["feed"].as_array() {
                let f: Vec<u32> = f.iter().map(|v| v.as_u64().unwrap() as u32).collect();
                rows.extend(e.feed(&f)?);
                tap_rows.extend(e.taps()?);
            } else if op["snapshot"].as_bool() == Some(true) {
                e.snapshot()?;
            } else if let Some(n) = op["crop"].as_u64() {
                eprintln!("crop {n} -> {}", e.crop(n as usize));
            } else if let Some(p) = op["save"].as_str() {
                let t0 = Instant::now();
                e.save_state(Path::new(p))?;
                eprintln!("save {} tokens: {:.2}s", e.length(), t0.elapsed().as_secs_f64());
            } else if let Some(p) = op["load"].as_str() {
                let t0 = Instant::now();
                let n = e.load_state(Path::new(p))?.len();
                eprintln!("load {n} tokens: {:.2}s", t0.elapsed().as_secs_f64());
            } else {
                return Err(format!("unknown op {op}"));
            }
        }
        if !taps.is_empty() {
            let tb: Vec<u8> = tap_rows.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{out}.taps"), tb).map_err(|e| e.to_string())?;
        }
        let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        return std::fs::write(out, bytes).map_err(|e| e.to_string());
    }
    let prompt = ids("prompt");
    let t0 = Instant::now();
    let mut rows: Vec<f32> = e.feed(&prompt)?;
    let prefill = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let steps = ids("steps");
    for &t in &steps {
        rows.extend(e.feed(&[t])?);
    }
    let dt = t0.elapsed().as_secs_f64();
    let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(out, bytes).map_err(|e| e.to_string())?;
    let (hits, page_ins) = e.expert_counters();
    eprintln!(
        "{}; prefill {} tokens in {prefill:.3}s; {} rows; {} decode steps in {:.3}s ({:.1} tok/s); experts {hits} hits, {page_ins} copied in",
        e.describe(),
        prompt.len(),
        1 + steps.len(),
        steps.len(),
        dt,
        steps.len() as f64 / dt.max(1e-9)
    );
    Ok(())
}

/// Material for the confidence probe (aegist/confidence.py runs this).
/// in.json: {"prompts": [[ids]], "max_new": n, "samples": k, "temperature", "top_p", "top_k", "taps": [layers], "seed",
/// "answers": [[ids]] (optional: replay these answers, where not empty, instead of answering; no samples then)}.
/// For each prompt: a greedy answer, with each token's log-prob and the entropy of its
/// distribution (temperature 1) and the tapped hidden states after feeding it; then k sampled
/// answers from the same prompt. out.json: [{"greedy", "logprob", "entropy", "samples"}];
/// out.json.taps: f32 [every greedy token of every prompt in order][taps][d].
/// `aegist exam in.json out.json`: each question in its own new conversation, through the whole
/// chat pipeline (reading core, tools, unsure marks, second look), as the user would get it.
/// in {"prompts": [text], "think": "off"|"auto"|"on", "seed": N} -> out [{"answer", "marks",
/// "final_marks", "tools", "second_look", "kind", "seconds"}]. The governor's settings are used
/// but not changed. With "taps": [layers] each result also gets "passes" (each answer pass's text
/// and the sentences the probe looked at) and out.json.taps their hidden-state means; --resume
/// goes on after the questions out.json.partial already has.
fn exam(home: &Path, a: &Args) -> Result<(), String> {
    let (inp, outp) = match &a.pos[1..] {
        [i, o] => (i, o),
        _ => return Err("usage: aegist exam in.json out.json [--brain FILE] [--no-tools] [--no-second-look] [--resume]".into()),
    };
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(inp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let prompts: Vec<String> = spec["prompts"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect();
    let brain = a.get("brain").map(PathBuf::from).unwrap_or_else(|| current_brain(home));
    // the fast brain only when asked for ("pick": auto|fast|big), never while recording probe data
    let pick = spec["pick"].as_str().map(String::from);
    let fast = if pick.is_some() && spec["taps"].is_null() && a.get("brain").is_none() { fast_brain(home, &brain) } else { None };
    let mut chat = load_brains(&brain, a.get("context").and_then(|c| c.parse().ok()).unwrap_or(4096), probe::find(home, &brain), fast)?;
    chat.governor = governor::Governor::load(home).detached();
    if let Some(f) = chat.fast.as_mut() {
        f.governor = governor::Governor::load(home).detached();
    }
    if let Some(n) = spec["max_new"].as_u64() {
        chat.gen.max_new_tokens = n as usize;
        if let Some(f) = chat.fast.as_mut() {
            f.gen.max_new_tokens = n as usize;
        }
    }
    let think = spec["think"].as_str().unwrap_or("off").to_string();
    let seed = spec["seed"].as_u64().unwrap_or(1);
    // "taps": [layers] records every sentence the probe looks at, for probe training on chat
    // answers (aegist/confidence.py train-chat): its hidden-state means go to out.json.taps as
    // f32 rows of taps x d, and each answer pass's sentences into the results
    let record: Vec<usize> = spec["taps"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
    if !record.is_empty() {
        chat.record_sentences(record.clone())?;
    }
    let taps_path = format!("{outp}.taps");
    // --resume: keep the answers of a stopped run (out.json.partial) and go on after them
    let mut results: Vec<Value> = match std::fs::read_to_string(format!("{outp}.partial")) {
        Ok(s) if a.flag("resume") => serde_json::from_str(&s).map_err(|e| e.to_string())?,
        _ => vec![],
    };
    let mut taps_file = if record.is_empty() {
        None
    } else {
        let f = std::fs::OpenOptions::new().create(true).write(true).open(&taps_path).map_err(|e| e.to_string())?;
        // rows written for a question that was not saved yet are dropped
        let end = results.last().and_then(|r| r["taps_end"].as_u64()).unwrap_or(0);
        f.set_len(end).map_err(|e| e.to_string())?;
        Some((f, end))
    };
    if !results.is_empty() {
        eprintln!("resuming after {} answered questions", results.len());
    }
    let t0 = Instant::now();
    for (i, q) in prompts.iter().enumerate().skip(results.len()) {
        let mut s = Session::new("normal");
        s.think = think.clone();
        if let Some(p) = &pick {
            s.brain = p.clone();
        }
        let t = Instant::now();
        let mut marks = vec![];
        let stats = chat.reply(&mut s, q, seed + i as u64, |ev| {
            if let Event::Unsure(p) = ev {
                marks.push(p);
            }
        })?;
        let answer = s.turns.last().map(|t| tokenizer::strip_thinking(&t.1)).unwrap_or_default();
        let mut row = json!({
            "answer": answer, "marks": marks, "final_marks": stats.final_marks, "tools": stats.tools_used,
            "second_look": stats.second_look, "kind": stats.kind.map(|k| k.name()), "seconds": t.elapsed().as_secs_f64(),
            "prompt_tokens": stats.prompt_tokens, "prefill_seconds": stats.prefill_seconds,
            "thinking_tokens": stats.thinking_tokens, "answer_tokens": stats.answer_tokens, "think_plan": stats.think_plan,
            "brain": stats.brain,
        });
        if let Some((f, end)) = &mut taps_file {
            let mut passes = vec![];
            for (text, sents) in &stats.recorded {
                let mut bytes = Vec::with_capacity(sents.iter().map(|s| s.mean.len() * 4).sum());
                for s in sents {
                    bytes.extend(s.mean.iter().flat_map(|v| v.to_le_bytes()));
                }
                f.seek(std::io::SeekFrom::Start(*end)).map_err(|e| e.to_string())?;
                f.write_all(&bytes).map_err(|e| e.to_string())?;
                *end += bytes.len() as u64;
                passes.push(json!({"text": text, "sentences": sents.iter().map(|s| json!({
                    "text": s.text, "stats": s.stats, "p": s.p, "tokens": s.tokens, "width": s.mean.len(),
                })).collect::<Vec<_>>()}));
            }
            f.flush().map_err(|e| e.to_string())?;
            row["passes"] = Value::Array(passes);
            row["taps_end"] = json!(*end);
        }
        results.push(row);
        eprintln!("{}/{} questions ({:.0}s)", i + 1, prompts.len(), t0.elapsed().as_secs_f64());
        // the answers so far, so a run that is stopped still leaves them
        std::fs::write(format!("{outp}.partial"), serde_json::to_string(&results).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    std::fs::write(outp, serde_json::to_string(&results).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// `aegist lanes BRAIN in.json out.bin --layer L`: for each token list in {"prompts": [[ids]]},
/// every token's hidden lanes going into layer L, as bf16 [tokens][hc * d], one prompt after the
/// other. Learning time on the 177B brain trains the layers from L up on these (aegist/learn_top.py).
fn lanes(a: &Args) -> Result<(), String> {
    let (brain, inp, outp) = match &a.pos[1..] {
        [b, i, o] => (b, i, o),
        _ => return Err("usage: aegist lanes BRAIN in.json out.bin --layer L".into()),
    };
    let layer: usize = a.get("layer").and_then(|l| l.parse().ok()).ok_or("--layer L is needed")?;
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(inp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let prompts: Vec<Vec<u32>> = spec["prompts"].as_array().ok_or("no prompts")?.iter()
        .map(|p| p.as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as u32)).collect()).collect();
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
    let nx = NxFile::open(brain)?;
    if !next::is_next(&nx.header["config"]) {
        return Err("aegist lanes is for the 177B brain (hyper-connection lanes)".into());
    }
    let mut eng = next::NextEngine::load(nx, longest.max(64), |_, _| {})?;
    eprintln!("{}", eng.describe());
    let mut out = std::io::BufWriter::new(std::fs::File::create(outp).map_err(|e| e.to_string())?);
    let t0 = Instant::now();
    let write = |out: &mut std::io::BufWriter<std::fs::File>, v: Vec<f32>| -> Result<(), String> {
        let mut bytes = Vec::with_capacity(v.len() * 2);
        for x in v {
            let b = x.to_bits();
            let r = (b + 0x7fff + ((b >> 16) & 1)) >> 16; // round to nearest even
            bytes.extend_from_slice(&(r as u16).to_le_bytes());
        }
        out.write_all(&bytes).map_err(|e| e.to_string())
    };
    if a.flag("one-by-one") {
        for (i, ids) in prompts.iter().enumerate() {
            write(&mut out, eng.lanes_before(ids, layer)?)?;
            if (i + 1) % 10 == 0 || i + 1 == prompts.len() {
                eprintln!("{}/{} ({:.0}s)", i + 1, prompts.len(), t0.elapsed().as_secs_f64());
            }
        }
    } else {
        // layer by layer over groups of prompts (each layer's experts read once per group),
        // up to $AEGIST_LANES_GROUP_TOKENS tokens (default 200,000: about 6.5 GB of host memory)
        let budget = std::env::var("AEGIST_LANES_GROUP_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000usize);
        let (mut at, mut done) = (0, 0);
        while at < prompts.len() {
            let mut end = at + 1;
            let mut n = prompts[at].len();
            while end < prompts.len() && n + prompts[end].len() <= budget {
                n += prompts[end].len();
                end += 1;
            }
            let group = &prompts[at..end];
            let res = eng.lanes_before_many(group, layer, |l| {
                if l % 5 == 0 || l == layer {
                    eprintln!("prompts {}-{} of {}: layer {l}/{layer} ({:.0}s)", at + 1, end, prompts.len(), t0.elapsed().as_secs_f64());
                }
            })?;
            for v in res {
                write(&mut out, v)?;
            }
            done += group.len();
            eprintln!("{done}/{} ({:.0}s)", prompts.len(), t0.elapsed().as_secs_f64());
            at = end;
        }
    }
    out.flush().map_err(|e| e.to_string())
}

/// `aegist score BRAIN in.json out.json`: in {"items": [{"ids": [..], "start": n}]}, the engine
/// reads ids[..start] and then each later token, and out gets each item's mean loss (nats per
/// token) on ids[start..]: [{"nll": x, "tokens": n}]. Learning time uses it to check that cores
/// learned in Python do the same on the engine. On the 177B brain the items are read layer by
/// layer instead (NextEngine::score_many, each layer's experts read once for all of them), unless
/// --decode asks for the token-at-a-time reads (what $AEGIST_SKIP_MISS measurements need).
fn score(a: &Args) -> Result<(), String> {
    let (brain, inp, outp) = match &a.pos[1..] {
        [b, i, o] => (b, i, o),
        _ => return Err("usage: aegist score BRAIN in.json out.json".into()),
    };
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(inp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let items = spec["items"].as_array().ok_or("no items")?;
    let parsed: Vec<(Vec<u32>, usize)> = items.iter().map(|it| {
        let ids: Vec<u32> = it["ids"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as u32)).collect();
        (ids, it["start"].as_u64().unwrap_or(1) as usize)
    }).collect();
    let longest = parsed.iter().map(|p| p.0.len()).max().unwrap_or(0);
    let nx = NxFile::open(brain)?;
    let t0 = Instant::now();
    if next::is_next(&nx.header["config"]) && !a.flag("decode") && parsed.iter().all(|p| p.0.len() >= 2) {
        // the 177B brain: every item layer by layer, each layer's experts read once for all of them
        let mut eng = next::NextEngine::load(nx, longest.max(64) + 8, |_, _| {})?;
        eprintln!("{}", eng.describe());
        let nl = eng.layers();
        let res = eng.score_many(&parsed, |l| {
            if l % 8 == 0 || l == nl {
                eprintln!("{} items: layer {l}/{nl} ({:.0}s)", parsed.len(), t0.elapsed().as_secs_f64());
            }
        })?;
        let out: Vec<Value> = res.iter().map(|&(nll, n)| json!({"nll": if n > 0 { nll / n as f64 } else { 0.0 }, "tokens": n})).collect();
        let ((hits, reads), (skips, saved)) = (eng.expert_counters(), eng.skipped());
        eprintln!("experts: {hits} uses from VRAM, {reads} read from disk; {skips} picks skipped, {saved} reads saved; {:.1}s in expert reads, {} read ahead (guessed {:?})", eng.read_seconds(), eng.prefetched(), eng.guess_hits());
        return std::fs::write(outp, serde_json::to_string(&out).map_err(|e| e.to_string())?).map_err(|e| e.to_string());
    }
    let mut eng = load_engine(nx, longest.max(64) + 8, |_, _| {})?;
    eprintln!("{}", eng.describe());
    // $AEGIST_SCORE_WAIT seconds idle first, as chat sits while the user types (speed tests of
    // the idle warm-up)
    if let Some(w) = std::env::var("AEGIST_SCORE_WAIT").ok().and_then(|v| v.trim().parse::<f64>().ok()) {
        std::thread::sleep(std::time::Duration::from_secs_f64(w.max(0.0)));
    }
    let t_scored = std::time::Instant::now();
    let mut out = vec![];
    for (i, (ids, start)) in parsed.iter().enumerate() {
        let start = (*start).clamp(1, ids.len());
        eng.crop(0);
        let mut logits = eng.feed(&ids[..start])?;
        let mut nll = 0.0f64;
        for j in start..ids.len() {
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z: f64 = logits.iter().map(|&l| (l as f64 - m).exp()).sum();
            nll += m + z.ln() - logits[ids[j] as usize] as f64;
            if j + 1 < ids.len() {
                logits = eng.feed(&ids[j..j + 1])?;
            }
        }
        let n = ids.len() - start;
        out.push(json!({"nll": if n > 0 { nll / n as f64 } else { 0.0 }, "tokens": n}));
        if (i + 1) % 10 == 0 || i + 1 == parsed.len() {
            eprintln!("{}/{} ({:.0}s)", i + 1, parsed.len(), t0.elapsed().as_secs_f64());
        }
    }
    let ((hits, reads), (skips, saved)) = (eng.expert_counters(), eng.skipped());
    eprintln!("experts: {hits} uses from VRAM, {reads} read from disk; {skips} picks skipped, {saved} reads saved; {:.1}s in expert reads, {} read ahead (guessed {:?}); scored in {:.1}s", eng.read_seconds(), eng.prefetched(), eng.guess_hits(), t_scored.elapsed().as_secs_f64());
    std::fs::write(outp, serde_json::to_string(&out).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// `aegist q4-file [BRAIN]`: make the 177B brain's 4-bit expert file (next.rs Q4File), the 4-bit
/// copies chat keeps in VRAM, read straight from disk instead of packed from FP8 on every miss.
/// About 68 GB next to the brain; minutes, mostly reading the brain once.
fn q4_file(brain: &Path) -> Result<(), String> {
    std::env::set_var("AEGIST_EXPERT_BITS", "4");
    let nx = NxFile::open(&brain.to_string_lossy())?;
    if !next::is_next(&nx.header["config"]) {
        return Err("the 4-bit expert file is for the 177B brain".into());
    }
    let mut eng = next::NextEngine::load(nx, 64, |_, _| {})?;
    let t0 = Instant::now();
    let path = eng.write_q4_file(|l, n| eprintln!("layer {l}/{n} ({:.0}s)", t0.elapsed().as_secs_f64()))?;
    eprintln!("wrote {} in {:.0}s", path.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn probe_data(a: &Args) -> Result<(), String> {
    let (brain, inp, out) = match &a.pos[1..] {
        [b, i, o] => (b, i, o),
        _ => return Err("usage: aegist probe-data BRAIN in.json out.json".into()),
    };
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(inp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let num = |k: &str, d: f64| spec[k].as_f64().unwrap_or(d);
    let (max_new, samples) = (num("max_new", 48.0) as usize, num("samples", 4.0) as usize);
    let (temp, top_p, top_k) = (num("temperature", 1.0) as f32, num("top_p", 0.95) as f32, num("top_k", 20.0) as usize);
    let taps: Vec<usize> = spec["taps"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
    let id_lists = |k: &str| -> Vec<Vec<u32>> {
        spec[k].as_array().into_iter().flatten()
            .map(|p| p.as_array().into_iter().flatten().map(|v| v.as_u64().unwrap_or(0) as u32).collect())
            .collect()
    };
    let prompts = id_lists("prompts");
    if prompts.is_empty() {
        return Err("in.json needs \"prompts\"".into());
    }
    // optional "answers": replay these (e.g. rated chat replies) instead of answering greedily
    let forced = id_lists("answers");
    let nx = NxFile::open(brain)?;
    let tok = tokenizer::Tok::from_json(nx.blob("tokenizer.json")?)?;
    let longest = prompts.iter().enumerate().map(|(i, p)| p.len() + forced.get(i).map_or(0, |a| a.len())).max().unwrap_or(0);
    let mut e = load_engine(nx, longest + max_new + 8, |_, _| {})?;
    e.set_taps(&taps)?;
    let mut rng = chat::Rng::new(num("seed", 1.0) as u64);
    let (mut results, mut tap_bytes) = (vec![], Vec::<u8>::new());
    let t0 = Instant::now();
    // without sampled answers nothing returns to a prompt's end, so the snapshot can sit where
    // this prompt stops sharing tokens with the last one (the system prompt) and save its prefill
    let reuse = samples == 0;
    let mut prev: &[u32] = &[];
    for (pi, p) in prompts.iter().enumerate() {
        let first = if reuse {
            let shared = prev.iter().zip(p).take_while(|(a, b)| a == b).count().min(p.len() - 1);
            let kept = e.crop(shared);
            prev = p;
            if kept < shared {
                e.feed(&p[kept..shared])?;
                e.snapshot()?;
                e.feed(&p[shared..])?
            } else {
                e.feed(&p[kept..])?
            }
        } else {
            e.crop(0);
            let f = e.feed(p)?;
            e.snapshot()?;
            f
        };
        let (mut greedy, mut logprob, mut entropy) = (vec![], vec![], vec![]);
        let mut logits = first.clone();
        let replay = forced.get(pi).filter(|a| !a.is_empty());
        while greedy.len() < replay.map_or(max_new, |a| a.len()) {
            let nxt = match replay {
                Some(a) => a[greedy.len()],
                None => chat::sample(&logits, 0.0, 1.0, 1, &mut rng),
            };
            if tok.eos.contains(&nxt) {
                break;
            }
            let (lse, ent) = probe::logit_stats(&logits);
            logprob.push(logits[nxt as usize] as f64 - lse);
            entropy.push(ent);
            greedy.push(nxt);
            logits = e.feed(&[nxt])?;
            tap_bytes.extend(e.taps()?.iter().flat_map(|v| v.to_le_bytes()));
        }
        let mut sampled = vec![];
        for _ in 0..if replay.is_some() { 0 } else { samples } {
            if e.crop(p.len()) != p.len() {
                return Err("could not return to the prompt's snapshot".into());
            }
            let (mut ids, mut logits) = (vec![], first.clone());
            while ids.len() < max_new {
                let nxt = chat::sample(&logits, temp, top_p, top_k, &mut rng);
                if tok.eos.contains(&nxt) {
                    break;
                }
                ids.push(nxt);
                if ids.len() < max_new {
                    logits = e.feed(&[nxt])?;
                }
            }
            sampled.push(ids);
        }
        results.push(json!({"greedy": greedy, "logprob": logprob, "entropy": entropy, "samples": sampled}));
        eprintln!("{}/{} prompts ({:.0}s)", pi + 1, prompts.len(), t0.elapsed().as_secs_f64());
    }
    std::fs::write(format!("{out}.taps"), tap_bytes).map_err(|e| e.to_string())?;
    std::fs::write(out, Value::Array(results).to_string()).map_err(|e| e.to_string())
}

/// in.json: [strings] -> out.json: {"trusted": [[ids]], "untrusted": [[ids]], "chat": [ids for each as a user turn]}.
fn debug_tokenize(a: &Args) -> Result<(), String> {
    let (brain, inp, out) = match &a.pos[1..] {
        [b, i, o] => (b, i, o),
        _ => return Err("usage: aegist debug-tokenize BRAIN in.json out.json".into()),
    };
    let nx = NxFile::open(brain)?;
    let tok = tokenizer::Tok::from_json(nx.blob("tokenizer.json")?)?;
    let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(inp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let r = json!({
        "trusted": texts.iter().map(|t| tok.encode_trusted(t)).collect::<Vec<_>>(),
        "untrusted": texts.iter().map(|t| tok.encode_untrusted(t)).collect::<Vec<_>>(),
        "chat": texts.iter().map(|t| tok.chat_ids("sys", &[("user".into(), t.clone())], false, false)).collect::<Vec<_>>(),
        "decoded": texts.iter().map(|t| tok.decode(&tok.encode_untrusted(t))).collect::<Vec<_>>(),
    });
    std::fs::write(out, r.to_string()).map_err(|e| e.to_string())
}

/// The reading core on its own. `aegist read "prompt"` prints what the prompt asks for;
/// `aegist read --data in.json out.json` runs Laya choices for the parity test:
/// in [{"question", "options", "state"}] -> out [{"ids", "markers", "p", "ms"}].
fn read_cmd(home: &Path, a: &Args) -> Result<(), String> {
    let dir = laya::find(home).ok_or("Laya is not downloaded (put it in <home>/models/laya or set AEGIST_LAYA)")?;
    let t0 = Instant::now();
    let l = laya::Laya::open(&dir)?;
    let load = t0.elapsed().as_secs_f64();
    if a.flag("data") {
        let (inp, out) = match &a.pos[1..] {
            [i, o] => (i, o),
            _ => return Err("usage: aegist read --data in.json out.json".into()),
        };
        let items: Value = serde_json::from_str(&std::fs::read_to_string(inp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut res = vec![];
        for it in items.as_array().ok_or("in.json must be a list")? {
            let opts: Vec<String> = it["options"].as_array().into_iter().flatten().filter_map(|o| o.as_str().map(String::from)).collect();
            let (q, st) = (it["question"].as_str().unwrap_or(""), it["state"].as_str().unwrap_or(""));
            let t = Instant::now();
            let (ids, markers) = l.build("choice", q, &opts, st)?;
            let p = l.choice(q, &opts, st)?;
            res.push(json!({"ids": ids, "markers": markers, "p": p, "ms": t.elapsed().as_secs_f64() * 1000.0}));
        }
        return std::fs::write(out, Value::Array(res).to_string()).map_err(|e| e.to_string());
    }
    let text = a.pos.get(1).ok_or("usage: aegist read \"prompt\"")?;
    let t = Instant::now();
    let (kind, p) = l.read(text)?;
    println!("{}  (fact {:.0}%, creative {:.0}%, other {:.0}%; read in {:.0} ms, Laya loaded in {load:.1}s)",
             kind.name(), 100.0 * p[0], 100.0 * p[1], 100.0 * p[2], t.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

// --- launcher -------------------------------------------------------------------------

fn prompt(q: &str) -> String {
    print!("{q}");
    std::io::stdout().flush().ok();
    let mut s = String::new();
    std::io::stdin().lock().read_line(&mut s).ok();
    s.trim().to_string()
}

fn launcher(home: &Path) -> Result<(), String> {
    loop {
        let brain = current_brain(home);
        println!("\n{CYAN}{BOLD}  AEGIST{RESET}  {DIM}{}{RESET}", brain.strip_prefix(home).unwrap_or(&brain).display());
        println!("  1  chat");
        println!("  2  chat, blunt mode");
        println!("  3  chat, idea mode (wild ideas, then a sober review)");
        println!("  4  learning time");
        println!("  5  brain versions");
        println!("  6  roll back one version");
        println!("  7  brain info");
        println!("  8  about");
        println!("  9  study: the next curriculum lesson");
        println!("  a  AQ test (quick; aegist aq --full for every question)");
        println!("  q  quit");
        let r = match prompt("> ").as_str() {
            "1" => run_chat(home, None, "normal", "auto", 0),
            "2" => run_chat(home, None, "blunt", "auto", 0),
            "3" => run_chat(home, None, "idea", "auto", 0),
            "4" => learn(home, &[]),
            "5" => learn(home, &["--status".into()]),
            "6" => learn(home, &["--rollback".into()]),
            "7" => info(&brain),
            "8" => about(home, false),
            "9" => study(home, "aegist.curriculum", &["run".into()]),
            "a" => aq(home, &[]),
            "q" | "quit" | "" => return Ok(()),
            _ => Ok(()),
        };
        if let Err(e) = r {
            println!("{YELLOW}{e}{RESET}");
        }
    }
}

fn main() {
    enable_ansi();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let a = parse(&argv, &["mode", "think", "context", "brain", "seed", "temp", "max-new", "layer", "pick"]);
    let home = home();
    // what learning time and the checks read must be the brain's exact computation: no skipped
    // expert picks there unless $AEGIST_SKIP_MISS asks for them
    let exact = ["lanes", "score", "debug-logits", "probe-data"];
    if a.pos.first().is_some_and(|c| exact.contains(&c.as_str())) {
        if std::env::var("AEGIST_SKIP_MISS").is_err() {
            std::env::set_var("AEGIST_SKIP_MISS", "0");
        }
        // and the FP8 experts as they are, not the 4-bit copies chat uses
        if std::env::var("AEGIST_EXPERT_BITS").is_err() {
            std::env::set_var("AEGIST_EXPERT_BITS", "8");
        }
    }
    let brain_arg = || a.pos.get(1).map(PathBuf::from).or_else(|| a.get("brain").map(PathBuf::from)).unwrap_or_else(|| current_brain(&home));
    let r = match a.pos.first().map(String::as_str) {
        None => launcher(&home),
        Some("chat") => run_chat(
            &home,
            a.get("brain").map(PathBuf::from),
            a.get("mode").unwrap_or("normal"),
            a.get("think").unwrap_or("auto"),
            a.get("context").and_then(|c| c.parse().ok()).unwrap_or(0),
        ),
        Some("ask") => ask(&home, &a),
        Some("learn") => learn(&home, &argv[1..]),
        Some("aq") => aq(&home, &argv[1..]),
        Some("tricky") => study(&home, "aegist.tricky", &argv[1..]),
        Some("study") => study(&home, "aegist.curriculum", &if argv.len() > 1 { argv[1..].to_vec() } else { vec!["status".into()] }),
        Some("lang") => study(&home, "aegist.conlang", &if argv.len() > 1 { argv[1..].to_vec() } else { vec!["--help".into()] }),
        Some("status") => learn(&home, &["--status".into()]),
        Some("rollback") => learn(&home, &["--rollback".into()]),
        Some("info") => info(&brain_arg()),
        Some("verify") => verify(&brain_arg()),
        Some("about") => about(&home, a.flag("license")),
        Some("register") => register(true),
        Some("unregister") => register(false),
        Some("debug-logits") => debug_logits(&a),
        Some("debug-tokenize") => debug_tokenize(&a),
        Some("probe-data") => probe_data(&a),
        Some("exam") => exam(&home, &a),
        Some("lanes") => lanes(&a),
        Some("q4-file") => q4_file(&brain_arg()),
        Some("score") => score(&a),
        Some("read") => read_cmd(&home, &a),
        Some(p) if p.ends_with(".nx") => run_chat(&home, Some(PathBuf::from(p)), "normal", "auto", 0),
        Some(other) => Err(format!("unknown command {other:?}; run aegist with no arguments for the menu")),
    };
    if let Err(e) = r {
        eprintln!("{YELLOW}aegist: {e}{RESET}");
        std::process::exit(1);
    }
}
