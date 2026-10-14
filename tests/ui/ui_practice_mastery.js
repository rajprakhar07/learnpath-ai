const { JSDOM } = require("jsdom");
const BASE = "http://localhost:5000"; const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0; const check = (n, c, x="") => { console.log((c?"PASS":"FAIL")+"  "+n+(x?"  -> "+x:"")); if(!c) failures++; };
(async () => {
  const dom = new JSDOM(await (await fetch(BASE+"/")).text(), { url: BASE+"/", runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window; w.fetch = (u,o)=>fetch(new URL(u,BASE),o); w.confirm=()=>true; w.Element.prototype.scrollIntoView=function(){};
  w.eval(await (await fetch(BASE+"/static/app.js")).text()); await sleep(400);
  const $=(s)=>w.document.querySelector(s), $$=(s)=>[...w.document.querySelectorAll(s)];
  $("#load-sample").click(); await sleep(700);
  // Mastery tab before any practice: honest empty state
  $$(".tab")[2].click(); await sleep(500);
  check("mastery tab shows no estimates before practice", /nothing is estimated/.test($("#mastery-body").textContent) && $$(".mrow .mval.na").length === 6, $$(".mrow .mval").map(x=>x.textContent).join(","));
  // Practice
  $$(".tab")[1].click(); await sleep(900);
  check("question rendered with 4 options", $$(".opt").length === 4 && $(".qstem").textContent.includes("_____"), $(".qstem") && $(".qstem").textContent.slice(0,60));
  check("topic/answer not shown before answering", !/Overfitting|Evaluating|Neural|Loss Functions/.test($("#practice-body").textContent) );
  $$(".opt")[0].click(); await sleep(700);
  check("feedback rendered", !!$(".fb") && $$(".opt:disabled").length === 4);
  check("exactly one option marked right", $$(".opt.right").length === 1);
  check("feedback has source chip with page", /^p\.\d$/.test($(".fb .chip").textContent), $(".fb .chip").textContent);
  check("viewer opened at the source page", /#page=\d$/.test($("#viewer").getAttribute("src")||""), $("#viewer").getAttribute("src"));
  check("mastery delta + reason shown", /mastery \d\.\d\d → \d\.\d\d/.test($(".fb .delta").textContent) && /Why this question/.test($(".fb").textContent));
  $(".btn.primary").click(); await sleep(900);
  check("next question loads", $$(".opt").length === 4 && !$(".fb"));
  // Mastery after practice reflects actual attempts
  $$(".tab")[2].click(); await sleep(500);
  check("mastery now shows computed value for answered topic", $$(".mrow .mval:not(.na)").length >= 1 && /1 recorded answer/.test($("#mastery-body").textContent), $("#mastery-body .mnote").textContent);
  console.log(failures ? failures+" FAILURE(S)" : "ALL STAGE-2 UI CHECKS PASSED"); process.exit(failures?1:0);
})().catch(e=>{console.error("HARNESS ERROR",e);process.exit(2)});
