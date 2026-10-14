const { JSDOM } = require("jsdom");
const BASE = "http://localhost:5000"; const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0; const check = (n, c, x="") => { console.log((c?"PASS":"FAIL")+"  "+n+(x?"  -> "+x:"")); if(!c) failures++; };
(async () => {
  const dom = new JSDOM(await (await fetch(BASE+"/")).text(), { url: BASE+"/", runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window; w.fetch=(u,o)=>fetch(new URL(u,BASE),o); w.confirm=()=>true; w.Element.prototype.scrollIntoView=function(){};
  w.eval(await (await fetch(BASE+"/static/app.js")).text()); await sleep(400);
  const $=(s)=>w.document.querySelector(s), $$=(s)=>[...w.document.querySelectorAll(s)];
  const tab=(n)=>$$(".tab").find(t=>t.dataset.tab===n).click();
  $("#load-sample").click(); await sleep(700);
  tab("review"); await sleep(500);
  check("review before any practice recommends practising only", $$(".rec").length===1 && /Coverage/.test($(".rec").textContent) && !/Weak concept/.test($("#review-body").textContent));
  // practise: answer first question wrong via UI (click each until feedback says Not quite), then more
  tab("practice"); await sleep(900);
  for (let i=0;i<4;i++){ $$(".opt")[0].click(); await sleep(500); if (/Not quite/.test($(".fb")?.textContent||"")) break; $(".btn.primary").click(); await sleep(800); }
  check("got at least one wrong answer via UI", /Not quite/.test($(".fb")?.textContent||""));
  tab("review"); await sleep(600);
  check("weak concept recommended with real numbers", /Weak concept/.test($("#review-body").textContent) && /Concept mastery is \d\.\d\d/.test($("#review-body").textContent));
  $$(".rec .btn.primary").find(b=>/revision/.test(b.textContent)).click(); await sleep(700);
  check("diagnosis + passages + steps shown", /1 · Diagnosis/.test($("#review-body").textContent) && $$(".pass").length>=1 && $$(".step").length>=3, $$(".pass .loc").map(x=>x.textContent).join(" | "));
  check("viewer opened on a source page", /#page=\d$/.test($("#viewer").getAttribute("src")||""), $("#viewer").getAttribute("src"));
  $$("#review-body .actions .btn.primary")[0].click(); await sleep(800);
  check("retest question shown with 4 options", $$("#review-body .opt").length===4 && /4 · A different question/.test($("#review-body").textContent));
  $$("#review-body .opt")[1].click(); await sleep(800);
  check("before/after comparison rendered", $$(".cmp div").length===3 && /5 · Before vs after/.test($("#review-body").textContent), $$(".cmp b").map(x=>x.textContent).join(" / "));
  check("caveat about single question shown", /weak evidence/.test($(".result").textContent));
  console.log(failures?failures+" FAILURE(S)":"ALL STAGE-3 UI CHECKS PASSED"); process.exit(failures?1:0);
})().catch(e=>{console.error("HARNESS ERROR",e);process.exit(2)});
