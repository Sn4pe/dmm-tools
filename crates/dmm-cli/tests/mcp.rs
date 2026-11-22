//! `mcp` speaks the Model Context Protocol on stdio against the mock: a
//! client's session from the handshake to a tool error, and stdout carrying
//! nothing but protocol messages.
#![cfg(feature = "mcp")]

mod common;

use common::dmm_cli;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::Stdio;

/// Send `requests` (each with an `id`), read one answer per request, then
/// close stdin; the server must exit on its own. Answers come back by id.
fn session(requests: &[Value]) -> Vec<Value> {
    let mut child = dmm_cli()
        .args(["--device", "mock", "mcp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("run dmm-cli mcp");
    let mut stdin = child.stdin.take().expect("stdin");
    let initialize = json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"}}});
    writeln!(stdin, "{initialize}").expect("write");
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
    )
    .expect("write");
    for request in requests {
        writeln!(stdin, "{request}").expect("write");
    }
    let mut answers = vec![Value::Null; requests.len()];
    let mut lines = BufReader::new(child.stdout.take().expect("stdout")).lines();
    for _ in 0..=requests.len() {
        let line = lines.next().expect("an answer").expect("read stdout");
        let answer: Value = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("stdout carries only JSON-RPC ({e}): {line}"));
        let id = usize::try_from(answer["id"].as_u64().expect("an answer has an id"))
            .expect("a small id");
        if id > 0 {
            answers[id - 1] = answer;
        }
    }
    drop(stdin);
    assert!(child.wait().expect("wait").success(), "exits on stdin EOF");
    answers
}

fn call(id: usize, name: &str, arguments: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments}})
}

#[test]
fn an_agent_reads_samples_and_presses_buttons() {
    let answers = session(&[
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        call(2, "read", json!({})),
        call(3, "sample", json!({"count": 3, "interval_ms": 0})),
        call(4, "press", json!({"command": "hold"})),
    ]);

    let mut tools: Vec<&str> = answers[0]["result"]["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|t| t["name"].as_str().expect("name"))
        .collect();
    tools.sort_unstable();
    assert_eq!(tools, ["press", "read", "sample"]);

    let read = &answers[1]["result"]["structuredContent"];
    assert_eq!(read["meter"], "Mock UT61E+");
    assert_eq!(read["reading"]["mode"], "DC V");
    assert_eq!(read["reading"]["unit"], "V");
    assert!(read["reading"]["value"].is_f64(), "got {read}");

    let sample = &answers[2]["result"]["structuredContent"];
    assert_eq!(sample["readings"].as_array().expect("readings").len(), 3);
    assert_eq!(sample["stats"]["count"], 3);
    assert_eq!(sample["stats"]["unit"], "V");
    assert_eq!(sample["last"]["mode"], "DC V");
    assert_eq!(sample["readings"][0]["elapsed_s"], 0.0);
    let (min, max) = (&sample["stats"]["min"], &sample["stats"]["max"]);
    assert!(min.as_f64() <= max.as_f64(), "got {sample}");

    assert_eq!(answers[3]["result"]["structuredContent"]["sent"], "hold");
}

/// What the agent did wrong comes back as a tool error it can read and act
/// on, not a protocol error.
#[test]
fn a_wrong_call_is_a_tool_error_that_says_why() {
    let answers = session(&[
        call(1, "press", json!({"command": "nope"})),
        call(2, "sample", json!({"count": 0})),
    ]);
    let text = |answer: &Value| {
        assert_eq!(answer["result"]["isError"], true, "got {answer}");
        answer["result"]["content"][0]["text"]
            .as_str()
            .expect("text")
            .to_string()
    };
    let unknown = text(&answers[0]);
    assert!(
        unknown.contains("'nope'") && unknown.contains("hold"),
        "got {unknown}"
    );
    assert!(text(&answers[1]).contains("count must be 1 to"));
}

/// A client that goes away mid-call (stdin closed) ends the server at once,
/// with the readings taken so far, rather than leaving it sampling with the
/// cable held until rmcp's drain gives up.
#[test]
fn closing_stdin_ends_a_sample_in_progress() {
    let mut child = dmm_cli()
        .args(["--device", "mock", "mcp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("run dmm-cli mcp");
    let mut stdin = child.stdin.take().expect("stdin");
    let initialize = json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"}}});
    writeln!(stdin, "{initialize}").expect("write");
    let mut lines = BufReader::new(child.stdout.take().expect("stdout")).lines();
    lines
        .next()
        .expect("initialize answer")
        .expect("read stdout");
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
    )
    .expect("write");
    let long = call(1, "sample", json!({"count": 50, "interval_ms": 1000}));
    writeln!(stdin, "{long}").expect("write");
    std::thread::sleep(std::time::Duration::from_millis(300));
    let closed = std::time::Instant::now();
    drop(stdin);
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            break;
        }
        assert!(
            closed.elapsed() < std::time::Duration::from_secs(2),
            "still running after stdin closed"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let line = lines.next().expect("the sample's answer").expect("read");
    let answer: Value = serde_json::from_str(&line).expect("JSON-RPC");
    let sampled = &answer["result"]["structuredContent"];
    assert_eq!(sampled["cut_short"], true, "got {answer}");
    assert!(!sampled["readings"].as_array().expect("readings").is_empty());
}
