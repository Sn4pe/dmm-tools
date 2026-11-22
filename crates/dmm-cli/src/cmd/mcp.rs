//! `mcp`: a Model Context Protocol server on stdin/stdout, so an AI agent can
//! read the meter and press its buttons as tools.
//!
//! stdout carries only protocol messages; everything the open path prints
//! (notes, help) already goes to stderr. The meter is opened on the first
//! tool call, not at start, so the client can launch the server before the
//! meter is switched on, and a failed open is retried on the next call.
//! Readings use the JSON export's shape, as `read --format json` prints them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dmm_lib::error::{Error, ErrorKind, Unsupported};
use dmm_lib::measurement::Measurement;
use dmm_lib::protocol::registry::Selection;
use dmm_lib::stats::SeriesStats;
use dmm_lib::text::TextKind;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::service::RequestContext;
use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router,
};
use serde_json::{Value, json};

use crate::cmd::command::UnknownCommand;
use crate::open::{BoxedDmm, open_any, opened_device, profile_name};

/// The most readings one `sample` call takes, and the longest it may run,
/// opening the meter included. A call blocks the agent until it returns, and
/// Claude Code moves a call still running after two minutes to the
/// background; other clients may give up sooner.
const MAX_SAMPLES: u32 = 1000;
const MAX_SAMPLE_SPAN: Duration = Duration::from_secs(50);
/// Names both limits above; a test keeps it in step with them.
const SAMPLE_DESCRIPTION: &str = "Take a series of readings and summarise them: min, max, mean and standard deviation of the last comparable series (a dial turn or a change of quantity starts a new one; auto-ranging does not). count is 1 to 1000; count * interval_ms must stay within 50 seconds, and a call that runs longer (a slow meter, a Bluetooth connect) returns early with cut_short set. Returns the stats, the last reading in full (as the read tool gives it) and one {elapsed_s, value, unit} row per reading.";
/// How often a wait between readings looks at the stop flags.
const STOP_POLL: Duration = Duration::from_millis(50);

/// How to open the meter, owned so the server can open it later.
#[derive(Clone)]
struct Target {
    selection: Selection,
    adapter: Option<String>,
    bluetooth: bool,
}

#[derive(Clone)]
struct Server {
    target: Target,
    /// The open meter, or `None` until a tool opens it or after the link
    /// was lost.
    dmm: Arc<Mutex<Option<BoxedDmm>>>,
    /// Raised when the client has gone (stdin closed): a call in progress
    /// ends, so the process exits and lets go of the cable.
    shutdown: Arc<AtomicBool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct SampleArgs {
    /// Number of readings to take
    #[serde(default = "default_count")]
    count: u32,
    /// Milliseconds from one reading to the next (0 = as fast as the meter answers)
    #[serde(default = "default_interval_ms")]
    interval_ms: u64,
}

fn default_count() -> u32 {
    10
}

fn default_interval_ms() -> u64 {
    500
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct PressArgs {
    /// Button command name, e.g. "hold", "range", "auto", "rel", "minmax".
    /// An unknown name answers with the ones the meter takes.
    command: String,
}

/// Whether the work in progress should end: the client cancelled this call
/// or went away.
type Stopped<'a> = &'a dyn Fn() -> bool;

#[tool_router]
impl Server {
    fn new(target: Target) -> Self {
        Self {
            target,
            dmm: Arc::new(Mutex::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    #[tool(
        description = "Take one reading from the multimeter: mode, value, unit, range and status flags (hold, rel, auto range, low battery, ...). value is a number in the displayed unit, \"OL\" on overload, {\"ncv_level\": n} in NCV mode, or null when the meter shows no reading."
    )]
    async fn read(&self, ctx: RequestContext<RoleServer>) -> Result<CallToolResult, ErrorData> {
        self.with_meter(ctx, |dmm, _| {
            let m = dmm.request_measurement()?;
            Ok(json!({ "meter": meter_name(dmm), "reading": reading_json(dmm, &m)? }))
        })
        .await
    }

    #[tool(description = SAMPLE_DESCRIPTION)]
    async fn sample(
        &self,
        ctx: RequestContext<RoleServer>,
        Parameters(args): Parameters<SampleArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let called = Instant::now();
        if let Err(refused) = check_sample(&args) {
            return Ok(CallToolResult::error(vec![ContentBlock::text(refused)]));
        }
        self.with_meter(ctx, move |dmm, stopped| sample(dmm, &args, stopped, called))
            .await
    }

    #[tool(
        description = "Press one of the multimeter's buttons remotely (e.g. hold, range, auto, rel, minmax). An unknown name answers with the commands this meter takes."
    )]
    async fn press(
        &self,
        ctx: RequestContext<RoleServer>,
        Parameters(args): Parameters<PressArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.with_meter(ctx, move |dmm, _| {
            match dmm.send_command(&args.command) {
                Err(Error::UnsupportedCommand(Unsupported::Command(_))) => {
                    let profile = dmm.profile();
                    return Err(UnknownCommand::new(
                        profile_name(profile),
                        args.command.clone(),
                        profile.supported_commands,
                    )
                    .into());
                }
                sent => sent?,
            }
            Ok(json!({ "meter": meter_name(dmm), "sent": args.command }))
        })
        .await
    }
}

impl Server {
    /// Run `work` on the meter in a blocking thread, opening it first if it
    /// is not open. A failure becomes a tool error the agent reads, and a
    /// lost link closes the meter so the next call opens it again.
    async fn with_meter<F>(
        &self,
        ctx: RequestContext<RoleServer>,
        work: F,
    ) -> Result<CallToolResult, ErrorData>
    where
        F: FnOnce(&mut BoxedDmm, Stopped<'_>) -> Result<Value, crate::l10n::Failure>
            + Send
            + 'static,
    {
        let slot = self.dmm.clone();
        let target = self.target.clone();
        let shutdown = self.shutdown.clone();
        let cancelled = ctx.ct.clone();
        let done = tokio::task::spawn_blocking(move || {
            let stopped = || shutdown.load(Ordering::SeqCst) || cancelled.is_cancelled();
            // A panic in an earlier call poisons the lock: the meter it left
            // half-used is reopened, once, rather than the server lost.
            let mut guard = slot.lock().unwrap_or_else(|poisoned| {
                slot.clear_poison();
                let mut guard = poisoned.into_inner();
                *guard = None;
                guard
            });
            // A call that waited behind another may have been called off
            // meanwhile: it must not press a button nobody wants pressed.
            if stopped() {
                return Err("the call was cancelled before it reached the meter".to_string());
            }
            let dmm = match guard.as_mut() {
                Some(dmm) => dmm,
                None => {
                    let opts = dmm_lib::OpenOptions {
                        adapter: target.adapter.as_deref(),
                        bluetooth: target.bluetooth,
                        stop: Some(&shutdown),
                    };
                    let opened = open_any(target.selection, opts, None)
                        .map_err(|e| failure_text(&e, target.selection))?;
                    guard.insert(opened)
                }
            };
            let result = work(dmm, &stopped);
            if result.as_ref().is_err_and(link_lost) {
                *guard = None;
            }
            // Text here: a failure can't leave the thread, and text is all
            // the agent gets of it.
            result.map_err(|e| failure_text(&e, target.selection))
        })
        .await
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(match done {
            Ok(value) => CallToolResult::structured(value),
            Err(failure) => CallToolResult::error(vec![ContentBlock::text(failure)]),
        })
    }
}

#[tool_handler]
impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("dmm-cli", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Tools for a digital multimeter connected to this computer by USB or Bluetooth. \
                 The meter measures what its dial and leads are set to: ask the user to set them \
                 before reading. If a tool reports that no meter answered, pass on to the user \
                 how the error says to switch on the meter's data link."
                    .to_string(),
            )
    }
}

/// Serve MCP on stdio until the client closes stdin.
pub(crate) fn cmd_mcp(
    selection: Selection,
    opts: dmm_lib::OpenOptions<'_>,
) -> Result<(), crate::l10n::Failure> {
    let server = Server::new(Target {
        selection,
        adapter: opts.adapter.map(str::to_owned),
        bluetooth: opts.bluetooth,
    });
    let shutdown = server.shutdown.clone();
    // Released after the runtime: a Bluetooth link's Drop runs a runtime of
    // its own to disconnect, which tokio refuses inside another.
    let meter = server.dmm.clone();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let outcome = runtime.block_on(async {
        let stdin = EndsOnEof {
            inner: tokio::io::stdin(),
            shutdown: shutdown.clone(),
        };
        let service = server
            .serve((stdin, tokio::io::stdout()))
            .await
            .map_err(|e| e.to_string())?;
        service.waiting().await.map_err(|e| e.to_string())?;
        Ok::<_, String>(())
    });
    shutdown.store(true, Ordering::SeqCst);
    drop(runtime);
    drop(meter);
    Ok(outcome?)
}

/// stdin that raises `shutdown` at end of file. On EOF rmcp waits up to 5 s
/// for calls in progress, then cancels them and drops their answers; raising
/// the flag here ends a `sample` at once, so its cut-short answer is sent.
struct EndsOnEof<R> {
    inner: R,
    shutdown: Arc<AtomicBool>,
}

impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for EndsOnEof<R> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        let polled = std::pin::Pin::new(&mut self.inner).poll_read(cx, buf);
        if let std::task::Poll::Ready(Ok(())) = polled
            && buf.filled().len() == before
            && buf.remaining() > 0
        {
            self.shutdown.store(true, Ordering::SeqCst);
        }
        polled
    }
}

fn check_sample(args: &SampleArgs) -> Result<(), String> {
    if !(1..=MAX_SAMPLES).contains(&args.count) {
        return Err(format!("count must be 1 to {MAX_SAMPLES}"));
    }
    let span = Duration::from_millis(args.interval_ms).saturating_mul(args.count - 1);
    if span > MAX_SAMPLE_SPAN {
        return Err(format!(
            "count * interval_ms must stay within {} seconds",
            MAX_SAMPLE_SPAN.as_secs()
        ));
    }
    Ok(())
}

/// Take `args.count` readings, one every `args.interval_ms` from the first.
/// A frame that fails to parse is skipped and counted, as `read` skips it;
/// the run ends early, with what it has, when stopped or past the bound.
fn sample(
    dmm: &mut BoxedDmm,
    args: &SampleArgs,
    stopped: Stopped<'_>,
    called: Instant,
) -> Result<Value, crate::l10n::Failure> {
    let interval = Duration::from_millis(args.interval_ms);
    let start = Instant::now();
    let mut stats = SeriesStats::new(false);
    let mut readings = Vec::with_capacity(args.count as usize);
    let mut first_at = None;
    let mut last = None;
    let mut changes = Vec::new();
    let mut skipped = 0u32;
    let mut last_bad_frame = None;
    let mut cut_short = false;
    for i in 0..args.count {
        if !wait_until(start + interval.saturating_mul(i), stopped)
            || called.elapsed() > MAX_SAMPLE_SPAN
        {
            cut_short = true;
            break;
        }
        let m = match dmm.request_measurement() {
            Ok(m) => m,
            Err(e) if e.kind() == ErrorKind::Protocol => {
                skipped += 1;
                last_bad_frame = Some(e);
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        if let Some(change) = stats.push(&m) {
            changes.push(change.to_string());
        }
        let full = reading_json(dmm, &m)?;
        let first = *first_at.get_or_insert(m.timestamp);
        let elapsed = m
            .timestamp
            .checked_duration_since(first)
            .unwrap_or_default();
        // One short row a reading: the full shape runs to some 450
        // characters, and a thousand of them overflow what a client takes in.
        readings.push(json!({
            "elapsed_s": (elapsed.as_secs_f64() * 1000.0).round() / 1000.0,
            "value": full["value"],
            "unit": full["unit"],
        }));
        last = Some(full);
    }
    let Some(last) = last else {
        return Err(match last_bad_frame {
            Some(e) => e.into(),
            None => crate::l10n::Failure::Text(format!(
                "no reading fit in the {} seconds a sample may take",
                MAX_SAMPLE_SPAN.as_secs()
            )),
        });
    };
    let figures = stats.stats();
    Ok(json!({
        "meter": meter_name(dmm),
        "stats": {
            "unit": stats.unit(),
            "count": figures.count,
            "min": figures.min,
            "max": figures.max,
            "mean": figures.avg(),
            "std_dev": figures.std_dev(),
        },
        "cut_short": cut_short,
        "skipped_bad_frames": skipped,
        "series_changes": changes,
        "last": last,
        "readings": readings,
    }))
}

/// Sleep until `deadline` in short slices; false when stopped first.
fn wait_until(deadline: Instant, stopped: Stopped<'_>) -> bool {
    loop {
        if stopped() {
            return false;
        }
        let now = Instant::now();
        let Some(left) = deadline
            .checked_duration_since(now)
            .filter(|d| !d.is_zero())
        else {
            return true;
        };
        std::thread::sleep(left.min(STOP_POLL));
    }
}

/// One reading in the JSON export's shape.
fn reading_json(dmm: &BoxedDmm, m: &Measurement) -> Result<Value, crate::l10n::Failure> {
    let experimental = !dmm.profile().stability.is_verified();
    let mut line = Vec::with_capacity(512);
    dmm_shared::export::write_measurement_json(
        &mut line,
        m,
        &dmm_shared::export::timestamp(m.wall_time),
        experimental,
        None,
        None,
    )?;
    Ok(serde_json::from_slice(&line)?)
}

fn meter_name(dmm: &BoxedDmm) -> &'static str {
    profile_name(dmm.profile())
}

/// The failure as the agent reads it, in the run's language. A meter that
/// did not answer gets the steps that switch on its data link, which the
/// terminal shows as help above the error.
fn failure_text(failure: &crate::l10n::Failure, selection: Selection) -> String {
    let l = crate::l10n::get();
    let steps = failure.steps(l).or_else(|| {
        match (failure.lib().map(Error::kind), opened_device(selection)) {
            (Some(ErrorKind::Timeout), Some(device)) => Some(
                l.label(TextKind::Activation, device.activation_instructions)
                    .into_owned(),
            ),
            _ => None,
        }
    });
    let text = failure.text(l);
    match steps {
        Some(steps) => format!("{text}\n{steps}"),
        None => text,
    }
}

/// Whether `failure` means the link to the meter is gone or it stopped
/// answering (switched off, cable pulled), so the next call opens it again
/// rather than talk to a dead handle. One bad frame keeps the link.
fn link_lost(failure: &crate::l10n::Failure) -> bool {
    failure
        .lib()
        .is_some_and(|e| matches!(e.kind(), ErrorKind::Transport | ErrorKind::Timeout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dmm_lib::protocol::registry;
    use dmm_lib::transport::mock::MockTransport;

    /// A UT61E+ frame showing `digits` V DC, its checksum broken when `bad`.
    fn frame(digits: [u8; 7], bad: bool) -> Vec<u8> {
        let mut payload = vec![0x02, 0x30];
        payload.extend_from_slice(&digits);
        payload.extend_from_slice(&[0, 0, 0x30, 0x30, 0x30]);
        let mut frame = vec![0xAB, 0xCD, u8::try_from(payload.len() + 2).expect("short")];
        frame.extend_from_slice(&payload);
        let sum = frame
            .iter()
            .fold(0u16, |acc, &b| acc.wrapping_add(u16::from(b)));
        frame.extend_from_slice(&(sum ^ u16::from(bad)).to_be_bytes());
        frame
    }

    fn meter(frames: Vec<Vec<u8>>) -> BoxedDmm {
        let device = registry::find_device("ut61eplus").expect("a registry device");
        dmm_lib::Dmm::new(MockTransport::new(frames), (device.new_protocol)())
            .expect("open the mock link")
            .boxed()
    }

    fn sample_of(dmm: &mut BoxedDmm, count: u32) -> Result<Value, crate::l10n::Failure> {
        let args = SampleArgs {
            count,
            interval_ms: 0,
        };
        sample(dmm, &args, &|| false, Instant::now())
    }

    /// A sample that would block past the bound, or takes no readings, is
    /// refused before the meter is touched.
    #[test]
    fn a_sample_stays_within_its_bounds() {
        let args = |count, interval_ms| SampleArgs { count, interval_ms };
        assert!(check_sample(&args(10, 500)).is_ok());
        assert!(check_sample(&args(1, u64::MAX)).is_ok());
        assert!(check_sample(&args(0, 500)).is_err());
        assert!(check_sample(&args(MAX_SAMPLES + 1, 0)).is_err());
        assert!(check_sample(&args(51, 1000)).is_ok());
        assert!(check_sample(&args(52, 1000)).is_err());
    }

    /// The agent reads the limits in the tool's description, typed there by
    /// hand: they must be the ones `check_sample` enforces.
    #[test]
    fn the_sample_description_names_its_limits() {
        assert!(SAMPLE_DESCRIPTION.contains(&format!("1 to {MAX_SAMPLES}")));
        let span = format!("within {} seconds", MAX_SAMPLE_SPAN.as_secs());
        assert!(SAMPLE_DESCRIPTION.contains(&span));
    }

    /// A frame that fails its checksum is skipped and counted; the readings
    /// around it still make the sample.
    #[test]
    fn a_bad_frame_is_skipped_not_fatal() {
        let mut dmm = meter(vec![
            frame(*b" 5.2090", false),
            frame(*b" 5.2100", true),
            frame(*b" 5.2110", false),
        ]);
        let sampled = sample_of(&mut dmm, 3).expect("a sample");
        assert_eq!(sampled["readings"].as_array().expect("readings").len(), 2);
        assert_eq!(sampled["skipped_bad_frames"], 1);
        assert_eq!(sampled["stats"]["min"], 5.209);
        assert_eq!(sampled["stats"]["max"], 5.211);
    }

    /// The largest sample stays well inside what an MCP client takes in one
    /// result (Claude Code's default is 25,000 tokens).
    #[test]
    fn the_largest_sample_fits_a_client() {
        let frames = vec![frame(*b" 5.2090", false); MAX_SAMPLES as usize];
        let mut dmm = meter(frames);
        let sampled = sample_of(&mut dmm, MAX_SAMPLES).expect("a sample");
        assert_eq!(
            sampled["readings"].as_array().expect("readings").len(),
            1000
        );
        let size = sampled.to_string().len();
        assert!(size < 50_000, "{size} characters");
    }

    /// Only bad frames is no sample at all: the agent hears why.
    #[test]
    fn only_bad_frames_is_an_error() {
        let mut dmm = meter(vec![frame(*b" 5.2090", true), frame(*b" 5.2100", true)]);
        let failure = sample_of(&mut dmm, 2).expect_err("no reading came through");
        assert_eq!(
            failure.lib().map(dmm_lib::error::Error::kind),
            Some(ErrorKind::Protocol)
        );
    }

    /// A pulled cable or a meter that stopped answering closes the handle,
    /// so the next call opens the meter again; one bad frame keeps it.
    #[test]
    fn a_lost_link_reopens_and_a_bad_frame_does_not() {
        let failure = |e: Error| crate::l10n::Failure::from(e);
        assert!(link_lost(&failure(Error::LinkLost)));
        assert!(link_lost(&failure(Error::Timeout)));
        assert!(!link_lost(&failure(Error::InvalidResponse {
            message: "checksum".to_string(),
            raw: Vec::new(),
        })));
    }

    /// A meter that does not answer gets the steps that switch on its data
    /// link, which the agent would not see otherwise.
    #[test]
    fn a_silent_meter_says_how_to_switch_it_on() {
        let device = registry::find_device("ut61eplus").expect("a registry device");
        let l = crate::l10n::get();
        let steps = l.label(TextKind::Activation, device.activation_instructions);
        let text = failure_text(&Error::Timeout.into(), Selection::Device(device));
        assert!(text.ends_with(steps.as_ref()), "got {text}");
        let other = failure_text(&Error::LinkLost.into(), Selection::Device(device));
        assert!(!other.contains(steps.as_ref()), "got {other}");
    }

    /// A wait ends at its deadline, or at once when stopped.
    #[test]
    fn a_wait_ends_when_stopped() {
        let soon = Instant::now() + Duration::from_millis(20);
        assert!(wait_until(soon, &|| false));
        assert!(Instant::now() >= soon);
        let later = Instant::now() + Duration::from_secs(60);
        assert!(!wait_until(later, &|| true));
    }
}
