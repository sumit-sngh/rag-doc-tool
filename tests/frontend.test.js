/**
 * Frontend logic tests -- run with: node tests/frontend.test.js
 *
 * The frontend is a single HTML file with no build step, so there's no module to
 * import. This extracts the pure functions straight out of index.html by name
 * and exercises them, which means the tests run against the shipped source
 * rather than a copy that can drift.
 *
 * Only DOM-free logic is covered: escaping, citation markup, prose rendering,
 * the SSE frame parser, and the diagnostics trace line. Anything that touches
 * the DOM needs a browser and is out of scope here.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const HTML = path.join(__dirname, "..", "frontend", "index.html");
const NAMES = ["esc", "fmtBytes", "inline", "renderProse", "parseFrames", "traceLine"];

/* ----------------------------------------------------------------- extract */

function extract(source, name) {
  // `function name(...) { ... }` -- brace-match to the close.
  let m = new RegExp("\\nfunction " + name + "\\s*\\(").exec(source);
  if (m) {
    const open = source.indexOf("{", m.index + m[0].length - 1);
    let depth = 0;
    for (let i = open; i < source.length; i++) {
      if (source[i] === "{") depth++;
      else if (source[i] === "}" && --depth === 0) return source.slice(m.index, i + 1);
    }
    throw new Error("unbalanced braces in " + name);
  }

  // `const name = (...) => ...;` -- read to the `;` at depth zero.
  m = new RegExp("\\nconst " + name + " = ").exec(source);
  if (m) {
    let depth = 0;
    for (let i = m.index + m[0].length; i < source.length; i++) {
      const c = source[i];
      if ("([{".includes(c)) depth++;
      else if (")]}".includes(c)) depth--;
      else if (c === ";" && depth === 0) return source.slice(m.index, i + 1);
    }
  }
  throw new Error("could not find " + name + " in index.html");
}

const html = fs.readFileSync(HTML, "utf8");
const script = /<script>\n([\s\S]*)\n<\/script>/.exec(html)[1];
const context = { module: { exports: {} } };
vm.createContext(context);
vm.runInContext(NAMES.map((n) => extract(script, n)).join("\n"), context);
const F = {};
for (const n of NAMES) F[n] = vm.runInContext(n, context);

/* ------------------------------------------------------------------ runner */

let pass = 0;
const failures = [];

function eq(name, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  report(ok, name, ok ? "" : "got " + JSON.stringify(got) + ", want " + JSON.stringify(want));
}

function ok(name, cond, note) {
  report(!!cond, name, note || "");
}

function report(good, name, note) {
  if (good) {
    pass++;
    console.log("  ok   " + name);
  } else {
    failures.push(name);
    console.log("  FAIL " + name + (note ? "\n       " + note : ""));
  }
}

function section(title) {
  console.log("\n-- " + title + " " + "-".repeat(Math.max(2, 58 - title.length)));
}

/* ------------------------------------------------------- escaping / safety */

section("escaping and injection");

eq("angle brackets escaped", F.esc("<script>x</script>"), "&lt;script&gt;x&lt;/script&gt;");

const inj = F.renderProse("Look <img src=x onerror=alert(1)> here [1].");
ok("injected tag neutralised, not rendered",
  !inj.includes("<img") && inj.includes("&lt;img src=x onerror=alert(1)&gt;"), inj);

const inj2 = F.renderProse("</p><script>alert(1)</script>");
ok("closing-tag escape attempt is inert",
  !inj2.includes("<script") && !inj2.includes("</p><"), inj2);

ok("quotes escaped so attributes cannot break out",
  !F.inline('a" onmouseover="evil').includes('onmouseover="evil'));

/* --------------------------------------------------------------- citations */

section("citation markers");

const single = F.inline("Panels need cleaning [1].");
ok("single marker becomes a button",
  /<button type="button" class="cite" data-n="1"/.test(single), single);

const multi = F.inline("Both agree [2, 4].");
ok("multi-number marker splits into two chips",
  (multi.match(/class="cite"/g) || []).length === 2, multi);
ok("  ...carrying both numbers",
  multi.includes('data-n="2"') && multi.includes('data-n="4"'), multi);

ok("no marker, no button", !F.inline("Plain sentence.").includes("cite"));
ok("bracketed non-digits left alone", !F.inline("an [example] word").includes("cite"));

/* ------------------------------------------------------------------- prose */

section("prose blocks");

eq("two paragraphs", F.renderProse("One.\n\nTwo."), "<p>One.</p><p>Two.</p>");
eq("single newline becomes a break", F.renderProse("One.\nTwo."), "<p>One.<br>Two.</p>");
eq("dash list becomes a ul", F.renderProse("- a\n- b"), "<ul><li>a</li><li>b</li></ul>");
eq("bold and code", F.renderProse("**hi** `x`"), "<p><strong>hi</strong> <code>x</code></p>");
ok("paragraph then list", F.renderProse("Intro:\n\n- a\n- b").startsWith("<p>Intro:</p><ul>"));

/* --------------------------------------------------------- SSE frame parse */

section("SSE frame parsing");

const STREAM =
  'data: {"type":"meta","retrieved":[]}\n\n' +
  'data: {"type":"delta","text":"Solar "}\n\n' +
  'data: {"type":"delta","text":"panels "}\n\n' +
  'data: {"type":"delta","text":"lose 15%."}\n\n' +
  'data: {"type":"done","answer":"Solar panels lose 15%. [1]","grounded":true}\n\n' +
  "data: [DONE]\n\n";

const EXPECTED = ["meta", "delta", "delta", "delta", "done"];

const whole = F.parseFrames(STREAM);
eq("all frames parsed in one pass", whole.events.map((e) => e.type), EXPECTED);
eq("[DONE] sentinel is not an event", whole.rest, "");

// One byte at a time: every frame boundary gets straddled.
let buffer = "";
const drip = [];
for (const ch of STREAM) {
  buffer += ch;
  const r = F.parseFrames(buffer);
  buffer = r.rest;
  drip.push(...r.events);
}
eq("byte-at-a-time yields identical events", drip.map((e) => e.type), EXPECTED);
eq("deltas reassemble exactly",
  drip.filter((e) => e.type === "delta").map((e) => e.text).join(""),
  "Solar panels lose 15%.");

// Split at every offset; the event sequence must never change.
let unstableAt = null;
for (let n = 1; n < STREAM.length && unstableAt === null; n++) {
  let b = "";
  const evs = [];
  for (const part of [STREAM.slice(0, n), STREAM.slice(n)]) {
    b += part;
    const r = F.parseFrames(b);
    b = r.rest;
    evs.push(...r.events);
  }
  if (evs.map((e) => e.type).join(",") !== EXPECTED.join(",")) unstableAt = n;
}
ok("stable across all " + (STREAM.length - 1) + " split offsets",
  unstableAt === null, unstableAt !== null ? "diverged at offset " + unstableAt : "");

const partial = 'data: {"type":"delta","text":"ab';
eq("partial frame yields no events", F.parseFrames(partial).events.length, 0);
eq("  ...and is returned verbatim", F.parseFrames(partial).rest, partial);

eq("malformed JSON skipped, stream survives",
  F.parseFrames('data: {oops\n\ndata: {"type":"done"}\n\n').events.map((e) => e.type),
  ["done"]);
eq("error frame surfaces its code",
  F.parseFrames('data: {"type":"error","code":"no_documents"}\n\n').events[0].code,
  "no_documents");
eq("unicode survives", F.parseFrames('data: {"type":"delta","text":"café — 15°"}\n\n').events[0].text,
  "café — 15°");

/* -------------------------------------------------------------- trace line */

section("diagnostics trace");

const trace = F.traceLine({
  latency_ms: 812,
  diagnostics: {
    strategy: "hybrid", vector_hits: 12, keyword_hits: 7, candidates_considered: 16,
    rerank_strategy: "mmr", chunks_in_context: 4, filtered_out_by_distance: 3,
    llm_model: "llama3.1",
  },
});
ok("exposes the whole pipeline",
  /hybrid/.test(trace) && /12 vector \+ 7 keyword/.test(trace) &&
  /16 candidates/.test(trace) && /mmr/.test(trace) && /4 in context/.test(trace) &&
  /3 too distant/.test(trace) && /llama3\.1/.test(trace) && /812 ms/.test(trace),
  trace.replace(/<[^>]+>/g, "|"));

ok('rerank "none" is omitted', !/none/.test(F.traceLine({ diagnostics: { rerank_strategy: "none" } })));
eq("empty diagnostics render nothing", F.traceLine({ diagnostics: {} }), "");

/* ------------------------------------------------------------------- sizes */

section("byte formatting");

eq("human byte sizes",
  [F.fmtBytes(0), F.fmtBytes(512), F.fmtBytes(2048), F.fmtBytes(5 * 1024 * 1024)],
  ["0 B", "512 B", "2.0 KB", "5.0 MB"]);

/* ------------------------------------------------------------------ report */

console.log("");
if (failures.length) {
  console.log(failures.length + " failed, " + pass + " passed");
  failures.forEach((f) => console.log("  - " + f));
  process.exit(1);
}
console.log("all " + pass + " assertions passed");
