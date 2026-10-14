(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const feed = $("feed"), empty = $("empty"), form = $("ask-form"), qInput = $("q"), sendBtn = $("send");
  const docsEl = $("docs"), msgEl = $("upload-msg"), fileInput = $("file"), drop = $("drop");
  const cardsEl = $("cards"), viewer = $("viewer"), viewerBox = document.querySelector(".viewer");
  const viewerTitle = $("viewer-title"), viewerOpen = $("viewer-open"), evSub = $("ev-sub");
  let docs = [];
  let nonce = 0;

  const STOP = new Set("what which who whom when where why how does did that this these those with from into about between explain describe define tell give show please mean means used using example examples the and for are was were has have had can could should would will not you your".split());

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  async function api(path, opts) {
    const r = await fetch(path, opts);
    let body = null;
    try { body = await r.json(); } catch (_) { /* non-JSON error page */ }
    if (!r.ok) throw new Error((body && body.error) || `Request failed (${r.status})`);
    return body;
  }
  function setMsg(text, kind) { msgEl.textContent = text || ""; msgEl.className = "msg" + (kind ? " " + kind : ""); }

  /* ---------- status + documents ---------- */
  async function loadStatus() {
    try {
      const s = await api("/api/status");
      const m = $("mode");
      m.textContent = "";
      if (s.llm_configured) {
        m.append("Tutor mode: ", Object.assign(el("b", "", "LLM")), ` (${s.llm_model})`);
      } else {
        m.append("Tutor mode: ", el("b", "", "offline extractive"), " — answers are verbatim sentences from your material. Set LLM_API_KEY in .env for explanations.");
      }
    } catch (e) { $("mode").textContent = "Status unavailable"; }
  }
  async function loadDocs() {
    docs = await api("/api/documents");
    docsEl.textContent = "";
    for (const d of docs) {
      const li = el("li");
      li.append(el("div", "name", d.filename));
      const unit = { pdf: "pages", pptx: "slides", transcript: "cues" }[d.kind] || "units";
      li.append(el("div", "meta", `${d.n_units} ${unit} · ${d.n_chunks} passages` + (d.n_figures ? ` · ${d.n_figures} figure${d.n_figures === 1 ? "" : "s"}` : "") + (d.n_empty ? ` · ${d.n_empty} blank` : "")));
      const del = el("button", "del", "✕");
      del.title = "Remove document";
      del.setAttribute("aria-label", "Remove " + d.filename);
      del.onclick = async () => {
        if (!confirm(`Remove "${d.filename}" and its passages?`)) return;
        try { await api(`/api/documents/${d.id}`, { method: "DELETE" }); await loadDocs(); clearEvidence(); }
        catch (e) { setMsg(e.message, "err"); }
      };
      li.append(del);
      docsEl.append(li);
    }
    renderStarters();
  }
  function renderStarters() {
    const box = $("starters");
    box.textContent = "";
    if (!docs.some((d) => d.filename === "sample_course.pdf")) return;
    for (const q of ["What is gradient descent?", "What happens if the learning rate is too large?",
                     "What is the difference between precision and recall?", "How does L1 regularization differ from L2?",
                     "What is the capital of France?"]) {
      const b = el("button", "", q); b.type = "button";
      b.onclick = () => { qInput.value = q; form.requestSubmit(); };
      box.append(b);
    }
  }
  async function upload(file) {
    if (!file) return;
    setMsg(`Reading ${file.name}…`);
    const fd = new FormData(); fd.append("file", file);
    try {
      const r = await api("/api/documents", { method: "POST", body: fd });
      setMsg(r.duplicate ? "Already uploaded — using the existing copy." : `Added ${r.document.filename}: ${r.document.n_units} pages, ${r.document.n_chunks} passages.`, "ok");
      await loadDocs(); loadStatus();
    } catch (e) { setMsg(e.message, "err"); }
    finally { fileInput.value = ""; }
  }
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener("change", () => upload(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
  $("load-all").addEventListener("click", async () => {
    setMsg("Loading sample files…");
    try {
      const r = await api("/api/documents/sample", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ names: ["sample_course.pdf", "sample_deck.pptx", "sample_lecture.vtt", "sample_figure.pdf"] }) });
      setMsg(`Loaded ${r.documents.length} sample files (PDF, PPTX, transcript, PDF with figure).`, "ok");
      await loadDocs();
    } catch (e) { setMsg(e.message, "err"); }
  });
  $("load-sample").addEventListener("click", async () => {
    setMsg("Loading sample course…");
    try {
      const r = await api("/api/documents/sample", { method: "POST" });
      setMsg(r.duplicate ? "Sample course is already loaded." : "Sample course loaded.", "ok");
      await loadDocs();
    } catch (e) { setMsg(e.message, "err"); }
  });

  /* ---------- evidence panel ---------- */
  function clearEvidence() {
    cardsEl.textContent = ""; evSub.textContent = "Citations appear here";
    viewer.removeAttribute("src"); viewerBox.classList.remove("has"); viewerOpen.hidden = true; viewerTitle.textContent = "Source viewer";
  }
  function viewerUrls(c) {
    nonce += 1;
    if (c.loc_kind === "page") {
      const page = Math.round(c.loc_start);
      // The ?v= nonce forces a reload so a changed #page= fragment is honoured by built-in PDF viewers.
      return { src: `/files/${c.doc_id}?v=${nonce}#page=${page}`, open: `/files/${c.doc_id}#page=${page}`, title: `${c.doc_name} — page ${page}` };
    }
    if (c.loc_kind === "slide") {
      const u = `/view/${c.doc_id}?slide=${Math.round(c.loc_start)}`;
      return { src: u + `&v=${nonce}#hl`, open: u + "#hl", title: `${c.doc_name} — slide ${Math.round(c.loc_start)}` };
    }
    if (c.loc_kind === "time") {
      const u = `/view/${c.doc_id}?t=${c.loc_start}&te=${c.loc_end}`;
      return { src: u + `&v=${nonce}#hl`, open: u + "#hl", title: `${c.doc_name} — ${c.label || ""}` };
    }
    return null;
  }
  function openInViewer(c) {
    const u = viewerUrls(c);
    if (!u) return;
    viewer.src = u.src;
    viewerBox.classList.add("has");
    viewerTitle.textContent = u.title;
    viewerOpen.href = u.open; viewerOpen.hidden = false;
  }
  function highlight(container, text, question) {
    const words = [...new Set((question.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length > 3 && !STOP.has(w)))];
    if (!words.length) { container.textContent = text; return; }
    const stems = words.map((w) => w.replace(/(ing|ed|es|s|ation|ization)$/, "")).filter((w) => w.length > 2);
    const re = new RegExp("\\b(" + stems.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")[a-z]*", "gi");
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) container.append(document.createTextNode(text.slice(last, m.index)));
      const mk = el("mark", "", m[0]); container.append(mk);
      last = m.index + m[0].length;
      if (m[0].length === 0) re.lastIndex++;
    }
    if (last < text.length) container.append(document.createTextNode(text.slice(last)));
  }
  function showCards(citations, question, label) {
    cardsEl.textContent = "";
    evSub.textContent = label;
    for (const c of citations) {
      const card = el("div", "card"); card.dataset.n = c.n;
      const top = el("div", "card-top");
      top.append(el("span", "n", "S" + c.n), el("span", "doc", `${c.doc_name} · ${c.label}`));
      const b = el("button", "", `Open ${c.label}`); b.type = "button";
      b.onclick = () => { activate(c.n); openInViewer(c); };
      top.append(b);
      const ex = el("div", "excerpt"); highlight(ex, c.excerpt, question);
      card.append(top, ex);
      for (const f of c.figures || []) {
        const fig = el("figure", "fig");
        const img = document.createElement("img");
        img.src = f.url; img.loading = "lazy";
        img.alt = f.alt_text || "Figure from the source (the file provides no alt text)";
        fig.append(img, el("figcaption", "", f.caption || (f.alt_text ? "Alt text: " + f.alt_text : "No caption in the source file")));
        card.append(fig);
      }
      cardsEl.append(card);
    }
  }
  function activate(n) {
    document.querySelectorAll(".card").forEach((x) => x.classList.toggle("active", x.dataset.n === String(n)));
    document.querySelectorAll(".chip").forEach((x) => x.classList.toggle("active", x.dataset.n === String(n)));
    const card = cardsEl.querySelector(`.card[data-n="${n}"]`);
    if (card) card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  /* ---------- chat ---------- */
  function renderAnswer(result) {
    const wrap = el("div", "a" + (result.status === "answered" ? "" : " unsup"));
    const byN = Object.fromEntries((result.citations || []).map((c) => [c.n, c]));
    if (result.status === "answered") {
      wrap.append(el("div", "tag", "● Answered from your material"));
      for (const line of result.answer.split("\n").filter((l) => l.trim())) {
        const p = el("p", "line");
        for (const part of line.split(/(\[S\d+\])/)) {
          const m = part.match(/^\[S(\d+)\]$/);
          if (m && byN[m[1]]) {
            const c = byN[m[1]];
            const chip = el("button", "chip", c.label); chip.type = "button"; chip.dataset.n = c.n;
            chip.title = `${c.doc_name}, ${c.label}`;
            chip.onclick = () => { showCards(result.citations, result.question, "Sources for this answer"); activate(c.n); openInViewer(c); };
            p.append(chip);
          } else if (part) p.append(document.createTextNode(part));
        }
        wrap.append(p);
      }
      const modeNote = { llm: "", extractive: "Offline mode: these are verbatim sentences from your material, not a generated explanation.",
        extractive_fallback: "The AI tutor was unreachable, so this shows verbatim sentences from your material instead." }[result.mode] || "";
      if (modeNote) wrap.append(el("div", "note", modeNote));
      if (result.llm_error) wrap.append(el("div", "note warn", "LLM error: " + result.llm_error));
      for (const w of result.warnings || []) wrap.append(el("div", "note warn", w));
      if (result.removed_invalid_citations && result.removed_invalid_citations.length)
        wrap.append(el("div", "note warn", "Removed citation markers that did not match any retrieved source."));
      showCards(result.citations, result.question, "Sources for this answer");
      if (result.citations.length) { activate(result.citations[0].n); openInViewer(result.citations[0]); }
    } else {
      wrap.append(el("div", "tag", "○ Not found in your material"));
      wrap.append(el("p", "line", "Your uploaded material doesn't answer this, so I won't guess."));
      wrap.append(el("div", "note", result.reason));
      if (result.nearest && result.nearest.length) {
        const det = el("details", "near"); det.append(el("summary", "", "Closest passages (these do not answer the question)"));
        for (const c of result.nearest) {
          const it = el("div", "near-item"); it.append(el("div", "loc", `${c.doc_name} · ${c.label}`));
          it.append(el("div", "", c.excerpt.length > 220 ? c.excerpt.slice(0, 220) + "…" : c.excerpt));
          it.onclick = () => { showCards(result.nearest, result.question, "Closest passages (not an answer)"); openInViewer(c); };
          det.append(it);
        }
        wrap.append(det);
      }
      clearEvidence();
    }
    return wrap;
  }
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = qInput.value.trim();
    if (!q) return;
    if (empty) empty.remove();
    qInput.value = ""; sendBtn.disabled = true;
    const turn = el("div", "turn");
    turn.append(el("div", "q", q));
    const pending = el("div", "a"); pending.append(el("div", "thinking", "Searching your material…"));
    turn.append(pending); feed.append(turn); feed.scrollTop = feed.scrollHeight;
    try {
      const res = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
      pending.replaceWith(renderAnswer(res));
    } catch (err) {
      const box = el("div", "a unsup"); box.append(el("div", "tag", "Error"), el("p", "line", err.message));
      pending.replaceWith(box);
    } finally { sendBtn.disabled = false; qInput.focus(); feed.scrollTop = feed.scrollHeight; }
  });


  /* ---------- tabs ---------- */
  const panes = { tutor: $("pane-tutor"), practice: $("pane-practice"), mastery: $("pane-mastery"), review: $("pane-review") };
  function showTab(name) {
    document.querySelectorAll(".tab").forEach((t) => { const on = t.dataset.tab === name; t.classList.toggle("active", on); t.setAttribute("aria-selected", on); });
    for (const [k, el_] of Object.entries(panes)) el_.hidden = k !== name;
    if (name === "practice") loadPractice();
    if (name === "mastery") loadMastery();
    if (name === "review") loadReview();
  }
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

  /* ---------- practice ---------- */
  const pBody = $("practice-body");
  const DIFF = { 1: "Easy", 2: "Medium", 3: "Hard" };
  async function loadPractice() {
    pBody.textContent = "";
    pBody.append(el("p", "muted", "Preparing a question from your material…"));
    try {
      const r = await fetch("/api/quiz/next", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const res = await r.json();
      pBody.textContent = "";
      if (res.status === "ok") return renderQuestion(res.question);
      const box = el("div", "qcard");
      box.append(el("div", "qmeta", res.status === "exhausted" ? "All caught up" : "Nothing to practise"), el("p", "qstem", res.message || "No question available."));
      if (res.status === "exhausted") {
        const b = el("button", "btn", "See mastery"); b.type = "button"; b.onclick = () => showTab("mastery"); box.append(b);
      }
      pBody.append(box);
    } catch (e) { pBody.textContent = ""; pBody.append(el("p", "msg err", "Could not load a question: " + e.message)); }
  }
  function renderQuestion(q) {
    const card = el("div", "qcard");
    card.append(el("div", "qmeta", `${DIFF[q.difficulty] || ""} · multiple choice`), el("p", "qstem", q.stem));
    const btns = q.options.map((text, i) => {
      const b = el("button", "opt"); b.type = "button";
      b.append(el("span", "k", "ABCD"[i]), document.createTextNode(text));
      return b;
    });
    const fbBox = el("div"); const actions = el("div", "actions");
    const skip = el("button", "btn", "Skip"); skip.type = "button";
    skip.onclick = async () => { await fetch("/api/quiz/skip", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question_id: q.question_id }) }); loadPractice(); };
    actions.append(skip);
    btns.forEach((b, i) => b.onclick = async () => {
      btns.forEach((x) => x.disabled = true);
      try {
        const fb = await api("/api/quiz/answer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question_id: q.question_id, choice: i }) });
        btns[fb.answer_index].classList.add("right");
        if (!fb.correct) btns[i].classList.add("wrong");
        fbBox.append(renderFeedback(fb));
        actions.textContent = "";
        const nx = el("button", "btn primary", "Next question"); nx.type = "button"; nx.onclick = loadPractice; actions.append(nx);
        if (fb.citation) {
          const c = fb.citation;
          showCards([Object.assign({ n: 1 }, c)], fb.source_quote, "Source for this question");
          activate(1); openInViewer(c);
        }
      } catch (e) { fbBox.append(el("p", "msg err", e.message)); btns.forEach((x) => x.disabled = false); }
    });
    btns.forEach((b) => card.append(b));
    card.append(actions, fbBox); pBody.append(card);
  }
  function renderFeedback(fb) {
    const box = el("div", "fb" + (fb.correct ? "" : " bad"));
    box.append(el("h3", "", fb.correct ? "Correct" : "Not quite"));
    box.append(el("p", "", fb.explanation));
    const cite = el("button", "chip", fb.citation.label); cite.type = "button"; cite.title = `${fb.citation.doc_name}, ${fb.citation.label}`;
    cite.onclick = () => openInViewer(fb.citation);
    const line = el("p"); line.append("Source: ", cite, ` ${fb.citation.doc_name}`); box.append(line);
    const tm = fb.mastery_before.topic, ta = fb.mastery_after.topic;
    box.append(el("div", "delta", `Topic “${fb.topic}” mastery ${tm.mean.toFixed(2)} → ${ta.mean.toFixed(2)} (${ta.n} answer${ta.n === 1 ? "" : "s"}, ${ta.confidence} confidence)`));
    if (fb.difficulty_note) box.append(el("div", "why", fb.difficulty_note));
    box.append(el("div", "why", "Why this question: " + fb.selection_reason));
    return box;
  }

  /* ---------- mastery ---------- */
  async function loadMastery() {
    const body = $("mastery-body"); body.textContent = "";
    try {
      const m = await api("/api/mastery");
      if (!m.topics.length) { body.append(el("p", "muted", "Upload course material to start.")); return; }
      body.append(el("div", "mh", "Topics"));
      body.append(el("p", "mnote", m.attempts ? `Computed from your ${m.attempts} recorded answer${m.attempts === 1 ? "" : "s"} (${m.correct} correct).` : "No answers recorded yet — nothing is estimated until you practise."));
      for (const t of m.topics) body.append(masteryRow(t.topic, t.assessed ? t.mean : null, t.assessed ? `${t.n} answer${t.n === 1 ? "" : "s"} · ${t.confidence} confidence` : "not assessed yet"));
      if (m.concepts.length) {
        body.append(el("div", "mh", "Concepts you've been tested on"));
        for (const c of m.concepts) body.append(masteryRow(c.concept, c.mean, `${c.topic} · ${c.n} answer${c.n === 1 ? "" : "s"}`));
      }
    } catch (e) { body.append(el("p", "msg err", e.message)); }
  }
  function masteryRow(name, mean, sub) {
    const row = el("div", "mrow"); const left = el("div"); left.append(el("div", "nm", name), el("div", "sub", sub));
    const bar = el("div", "bar" + (mean === null ? " none" : "")); const fill = el("i"); fill.style.width = mean === null ? "0%" : Math.round(mean * 100) + "%"; bar.append(fill);
    row.append(left, bar, el("div", "mval" + (mean === null ? " na" : ""), mean === null ? "—" : mean.toFixed(2)));
    return row;
  }


  /* ---------- review: Proof-of-Mastery Loop ---------- */
  const rBody = $("review-body");
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  async function loadReview() {
    rBody.textContent = "";
    try {
      const r = await api("/api/review");
      const open = r.loops.find((l) => l.status !== "complete");
      if (open) return renderLoop(open);
      rBody.append(el("div", "mh", "Recommended next steps"));
      if (!r.recommendations.length) rBody.append(el("p", "muted", "Nothing to recommend right now. Recommendations appear from your recorded answers."));
      for (const rec of r.recommendations) {
        const box = el("div", "rec");
        box.append(el("div", "tagline", rec.type === "start_loop" ? "Weak concept" : rec.type === "practice" ? "Coverage" : "In progress"), el("h3", "", rec.title), el("p", "", rec.reason));
        const b = el("button", "btn primary", rec.type === "practice" ? "Go to practice" : "Start targeted revision"); b.type = "button";
        b.onclick = async () => {
          if (rec.type === "practice") return showTab("practice");
          const res = await post("/api/loop/start", rec.concept_key ? { concept_key: rec.concept_key } : {});
          if (res.status === "ok") renderLoop(res.loop); else { rBody.textContent = ""; rBody.append(el("p", "msg err", res.message)); }
        };
        box.append(b); rBody.append(box);
      }
      const done = r.loops.filter((l) => l.status === "complete");
      if (done.length) {
        rBody.append(el("div", "mh", "Completed loops"));
        for (const l of done) {
          const box = el("div", "rec");
          box.append(el("h3", "", `“${l.concept}” — ${l.result.verdict === "improved" ? "improved" : "not yet"}`), el("p", "", l.result.summary));
          rBody.append(box);
        }
      }
    } catch (e) { rBody.append(el("p", "msg err", e.message)); }
  }
  function passageBox(p) {
    const box = el("div", "pass");
    const loc = el("span", "loc", `${p.citation.doc_name} · ${p.citation.label} ↗`);
    loc.onclick = () => openInViewer(p.citation);
    box.append(loc, el("div", "muted", p.why));
    for (const s of p.key_sentences) box.append(el("div", "", "“" + s + "”"));
    return box;
  }
  function renderLoop(L) {
    rBody.textContent = "";
    const card = el("div", "qcard");
    card.append(el("div", "qmeta", `Proof-of-mastery loop · ${L.topic}`), el("p", "qstem", `Revise “${L.concept}”`));
    card.append(el("div", "mh", "1 · Diagnosis"), el("p", "", L.diagnosis.text));
    card.append(el("div", "mh", "2 · Where your material covers it"));
    L.revision.passages.forEach((p) => card.append(passageBox(p)));
    card.append(el("div", "mh", "3 · Targeted revision"));
    L.revision.steps.forEach((t, i) => card.append(el("div", "step", `${i + 1}. ${t}`)));
    if (L.revision.passages[0]) openInViewer(L.revision.passages[0].citation);
    const area = el("div", "actions");
    if (L.status === "revising") {
      const b = el("button", "btn primary", "I've revised — give me a new question"); b.type = "button";
      b.onclick = async () => {
        const res = await post(`/api/loop/${L.id}/retest`);
        if (res.status === "ok") return renderLoop(res.loop);
        area.textContent = ""; area.append(el("p", "msg err", res.message));
      };
      area.append(b);
    }
    card.append(area);
    if (L.status === "retest_pending") {
      const q = L.retest_question;
      card.append(el("div", "mh", "4 · A different question on the same concept"), el("p", "qstem", q.stem));
      const btns = q.options.map((t, i) => { const b = el("button", "opt"); b.type = "button"; b.append(el("span", "k", "ABCD"[i]), document.createTextNode(t)); return b; });
      btns.forEach((b, i) => b.onclick = async () => {
        btns.forEach((x) => x.disabled = true);
        try { const res = await post(`/api/loop/${L.id}/answer`, { choice: i }); btns[res.feedback.answer_index].classList.add("right"); if (!res.feedback.correct) b.classList.add("wrong"); renderResult(card, res); }
        catch (e) { card.append(el("p", "msg err", e.message)); btns.forEach((x) => x.disabled = false); }
      });
      btns.forEach((b) => card.append(b));
    }
    if (L.status === "complete" && L.result) renderResult(card, { loop: L, feedback: null });
    rBody.append(card);
  }
  function renderResult(card, res) {
    const R = res.loop.result;
    card.querySelectorAll(".result").forEach((n) => n.remove());
    const box = el("div", "fb result" + (R.verdict === "improved" ? "" : " bad"));
    box.append(el("h3", "", "5 · Before vs after"), el("p", "", R.summary));
    const cmp = el("div", "cmp");
    const mk = (label, val, cls) => { const d = el("div", cls || ""); d.append(el("b", "", val), el("span", "", label)); return d; };
    cmp.append(mk(`before (${R.baseline.n} answers)`, R.baseline.mean.toFixed(2)), mk(`after (${R.after.n} answers)`, R.after.mean.toFixed(2), R.delta > 0 ? "up" : "down"), mk("change", (R.delta > 0 ? "+" : "") + R.delta.toFixed(2), R.delta > 0 ? "up" : "down"));
    box.append(cmp);
    const rt = R.retest;
    const c = el("button", "chip", rt.citation.label); c.type = "button"; c.onclick = () => openInViewer(rt.citation);
    const line = el("p"); line.append("Retest source: ", c, ` “${rt.source_quote}”`); box.append(line);
    box.append(el("div", "why", rt.different_source_sentence_than_all_earlier_questions ? "This question used a different source sentence than every earlier question on this concept." : "Warning: this reused an earlier source sentence."));
    R.caveats.forEach((t) => box.append(el("div", "why", t)));
    const again = el("button", "btn", "Back to recommendations"); again.type = "button"; again.onclick = loadReview; box.append(again);
    card.append(box);
    openInViewer(rt.citation);
  }

  loadStatus(); loadDocs().catch((e) => setMsg(e.message, "err"));
})();
