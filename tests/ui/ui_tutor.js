const { JSDOM } = require("jsdom");
const BASE = "http://localhost:5000";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const check = (name, cond, extra="") => { console.log((cond ? "PASS" : "FAIL") + "  " + name + (extra ? "  -> " + extra : "")); if (!cond) failures++; };

(async () => {
  const html = await (await fetch(BASE + "/")).text();
  const dom = new JSDOM(html, { url: BASE + "/", runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window;
  w.fetch = (u, o) => fetch(new URL(u, BASE), o);
  w.confirm = () => true;
  w.Element.prototype.scrollIntoView = function () {};
  const appjs = await (await fetch(BASE + "/static/app.js")).text();
  w.eval(appjs);
  await sleep(500);
  const $ = (s) => w.document.querySelector(s);
  const $$ = (s) => [...w.document.querySelectorAll(s)];

  check("mode panel shows offline extractive", /offline extractive/.test($("#mode").textContent), $("#mode").textContent.slice(0, 40));
  $("#load-sample").click(); await sleep(800);
  check("sample course listed", $$("#docs li").length === 1 && /6 pages/.test($("#docs li").textContent), $("#docs li") && $("#docs li").textContent);
  check("starter questions rendered", $$("#starters button").length === 5);

  // supported question
  $$("#starters button")[0].click(); await sleep(900);
  check("answer rendered as supported", !!$(".a:not(.unsup) .tag"), $(".a .tag") && $(".a .tag").textContent);
  const chips = $$(".chip");
  check("citation chips present with page labels", chips.length >= 1 && /^p\.\d+$/.test(chips[0].textContent), chips.map(c=>c.textContent).join(","));
  const iframeSrc = $("#viewer").getAttribute("src") || "";
  check("viewer opened at cited page (#page=3)", /\/files\/1\?v=\d+#page=3$/.test(iframeSrc), iframeSrc);
  check("evidence card with excerpt shown", $$(".card").length >= 1 && $(".card .excerpt").textContent.length > 50);
  check("excerpt highlights query terms", $$(".card .excerpt mark").length >= 1, $$(".card .excerpt mark").map(m=>m.textContent).slice(0,4).join("|"));
  const open = $("#viewer-open");
  check("open-in-new-tab link has page fragment", !open.hidden && /#page=3$/.test(open.getAttribute("href")), open.getAttribute("href"));

  // supported on a different page: precision/recall -> page 5
  $("#q").value = "What is the difference between precision and recall?"; $("#ask-form").requestSubmit(); await sleep(900);
  check("second answer opens page 5", /#page=5$/.test($("#viewer").getAttribute("src")), $("#viewer").getAttribute("src"));

  // unsupported
  $("#q").value = "What is the capital of France?"; $("#ask-form").requestSubmit(); await sleep(900);
  const turns = $$(".turn"); const last = turns[turns.length - 1];
  check("unsupported question flagged", !!last.querySelector(".a.unsup") && /Not found in your material/.test(last.textContent));
  check("unsupported shows no citation chips", last.querySelectorAll(".chip").length === 0);
  check("evidence panel cleared for unsupported", $$(".card").length === 0 && !$("#viewer").getAttribute("src"));

  // XSS: question containing HTML must be inert
  $("#q").value = "<img src=x onerror=window.__xss=1> gradient"; $("#ask-form").requestSubmit(); await sleep(900);
  check("HTML in question is not executed/injected", !w.__xss && $$(".q img").length === 0);

  // empty submit does nothing
  const before = $$(".turn").length; $("#q").value = "   "; $("#ask-form").requestSubmit(); await sleep(200);
  check("blank question ignored", $$(".turn").length === before);

  console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL UI CHECKS PASSED");
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error("HARNESS ERROR", e); process.exit(2); });
