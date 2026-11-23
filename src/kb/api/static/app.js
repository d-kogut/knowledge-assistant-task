"use strict";

// Tekst z API wstawiamy wyłącznie przez textContent — nigdy jako HTML.

const EXAMPLES = [
  "Ile kosztuje Hot Desk Student i jakie ma ograniczenia?",
  "Jakie są etapy windykacji gdy członek nie płaci?",
  "Czy mogę przyjść z psem do lokalizacji na Krowodrzy?",
  "Jakie sale konferencyjne mają wideokonferencję i ile kosztują?",
  "Co się dzieje gdy zgubię kartę dostępu?",
  "Jaka jest marża operacyjna SpaceHub i z czego się składają koszty?",
  "Jakie są plany ekspansji na 2026-2027?",
  "Ile osób pracuje w dziale operacji i kto zarządza klastrem Zabłocie?",
  "Jak działa failover internetu i jaki jest SLA?",
  "Czym różni się plan Virtual Basic od Virtual Premium?",
];

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Minimalne formatowanie: **pogrubienie** -> <strong>, reszta jako tekst.
function formatted(text) {
  const parts = text.split(/\*\*(.+?)\*\*/g);
  return parts.map((part, i) => (i % 2 ? el("strong", { text: part }) : document.createTextNode(part)));
}

function sourceLocation(src) {
  return `${src.file}, linie ${src.line_start}–${src.line_end}`;
}

let providersLoaded = false;

// Lista dostawców z /status: dostępni (z kluczem) do wyboru, pozostali wyszarzeni.
function fillProviders(s) {
  const select = $("provider");
  const keep = providersLoaded ? select.value : null;
  const options = (s.providers || []).map((p) =>
    el("option", { value: p.name, ...(p.enabled ? {} : { disabled: "" }) },
      p.enabled ? `${p.label} — ${p.model}` : `${p.label} — brak ${p.key}`));
  options.push(el("option", { value: "" }, "bez modelu językowego (same fragmenty)"));
  select.replaceChildren(...options);
  const enabled = (s.providers || []).filter((p) => p.enabled).map((p) => p.name);
  select.value = keep !== null && (keep === "" || enabled.includes(keep)) ? keep : (s.default_provider || "");
  providersLoaded = true;
}

async function loadStatus() {
  try {
    const resp = await fetch("/status");
    const s = await resp.json();
    const enabled = (s.providers || []).filter((p) => p.enabled);
    const llm = enabled.length
      ? `modele: ${enabled.map((p) => p.model).join(", ")}`
      : "tryb bez modelu językowego (brak kluczy API)";
    const changed = s.last_change?.at ? ` · ostatnia zmiana: ${new Date(s.last_change.at).toLocaleTimeString("pl-PL")}` : "";
    const errors = Object.keys(s.errors || {}).length ? ` · błędy plików: ${Object.keys(s.errors).join(", ")}` : "";
    $("status").textContent = `Indeks: ${s.files ?? "?"} plików, ${s.chunks ?? "?"} fragmentów${changed} · ${llm}${errors}`;
    $("status").classList.toggle("warn", !s.llm_enabled || Boolean(errors));
    fillProviders(s);
  } catch {
    $("status").textContent = "Brak połączenia z API.";
    $("status").classList.add("warn");
  }
}

function renderAnswer(data) {
  const answer = el("div", { class: "answer" });
  for (const seg of data.segments) {
    answer.append(...formatted(seg.text));
    for (const n of seg.sources) {
      answer.append(el("button", { class: "cite", type: "button", text: `[${n}]`, title: "pokaż źródło", onclick: () => openSource(n) }));
    }
  }
  const sources = el("ol", { class: "sources" },
    data.sources.map((src) =>
      el("li", { id: `src-${src.n}`, value: src.n },
        el("details", {},
          el("summary", {}, el("span", { class: "crumb", text: src.breadcrumb }), el("span", { class: "loc", text: sourceLocation(src) })),
          src.cited_texts.map((quote) => el("blockquote", { text: quote })),
        ),
      ),
    ),
  );
  return [answer, el("h3", { text: "Źródła" }), sources];
}

function openSource(n) {
  const item = $(`src-${n}`);
  if (!item) return;
  item.querySelector("details").open = true;
  item.scrollIntoView({ behavior: "smooth", block: "nearest" });
  item.classList.remove("flash");
  void item.offsetWidth;
  item.classList.add("flash");
}

function renderFragments(data) {
  return [
    el("p", { class: "note", text: data.note }),
    el("ol", { class: "fragments" },
      data.fragments.map((f) =>
        el("li", {},
          el("div", { class: "crumb", text: f.breadcrumb }),
          el("div", { class: "loc", text: sourceLocation(f) }),
          el("pre", { text: f.body }),
        ),
      ),
    ),
  ];
}

function rankingTable(debug) {
  const columns = ["hybrid", "dense", "bm25"].filter((c) => Array.isArray(debug[c]));
  const rows = Math.max(...columns.map((c) => debug[c].length));
  const cell = (item) => (item ? el("td", { title: item.chunk_id }, el("span", { class: "file", text: item.file.slice(0, 2) }), ` ${item.section} `, el("span", { class: "score", text: item.score })) : el("td"));
  const label = { hybrid: "hybryda (RRF)", dense: "wektory (PolDense)", bm25: "słowa (BM25)" };
  return el("table", { class: "ranking" },
    el("thead", {}, el("tr", {}, el("th", { text: "#" }), columns.map((c) => el("th", { text: label[c] || c })))),
    el("tbody", {}, Array.from({ length: rows }, (_, i) => el("tr", {}, el("td", { text: i + 1 }), columns.map((c) => cell(debug[c][i]))))),
  );
}

function renderDebug(data) {
  const d = data.debug || {};
  const facts = [
    `tryb: ${d.mode}`,
    d.query_lemmas !== undefined ? `lematy pytania: „${d.query_lemmas}”` : null,
    `czasy [ms]: ${Object.entries(data.timings_ms || {}).map(([k, v]) => `${k} ${v}`).join(", ")}`,
    data.usage ? `tokeny: ${Object.entries(data.usage).map(([k, v]) => `${k} ${v}`).join(", ")}` : null,
  ].filter(Boolean);
  return el("details", { class: "debug", open: "" },
    el("summary", { text: "Wyszukiwanie krok po kroku" }),
    el("ul", {}, facts.map((f) => el("li", { text: f }))),
    rankingTable(d),
    d.llm_raw_text ? el("details", {}, el("summary", { text: "surowa odpowiedź modelu" }), el("pre", { text: d.llm_raw_text })) : null,
  );
}

function render(data) {
  const box = $("result");
  box.replaceChildren();
  if (data.status === "answered") box.append(...renderAnswer(data));
  else if (data.status === "not_found") box.append(el("div", { class: "notfound" }, el("strong", { text: "Nie ma tego w dokumentacji." }), el("p", { text: data.note || "" })));
  else box.append(...renderFragments(data));
  const meta = [data.provider && data.model ? `${data.provider}: ${data.model}` : data.model, data.timings_ms?.total ? `${(data.timings_ms.total / 1000).toFixed(1)} s` : null].filter(Boolean).join(" · ");
  if (meta) box.append(el("p", { class: "meta", text: meta }));
  if (data.debug) box.append(renderDebug(data));
  box.hidden = false;
}

async function ask(event) {
  event?.preventDefault();
  const question = $("question").value.trim();
  if (question.length < 2) return;
  const button = $("submit");
  button.disabled = true;
  button.textContent = "Szukam…";
  $("result").hidden = false;
  $("result").replaceChildren(el("p", { class: "loading", text: "Szukam w dokumentacji i przygotowuję odpowiedź…" }));
  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        mode: $("mode").value,
        debug: $("debug").checked,
        llm: $("provider").value !== "",
        provider: $("provider").value || null,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    render(data);
  } catch (err) {
    $("result").replaceChildren(el("p", { class: "error", text: `Błąd: ${err.message}` }));
  } finally {
    button.disabled = false;
    button.textContent = "Zapytaj";
  }
}

function init() {
  $("example-list").append(...EXAMPLES.map((q) => el("li", {}, el("button", { type: "button", class: "example", text: q, onclick: () => { $("question").value = q; ask(); } }))));
  $("ask-form").addEventListener("submit", ask);
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) ask(e); });
  loadStatus();
  setInterval(loadStatus, 10000);
}

document.addEventListener("DOMContentLoaded", init);
