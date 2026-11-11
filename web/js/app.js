// 화면 흐름: 시작 → ①사진(동물상) → ②기본 정보 → ③이상형·MBTI → ④자기소개 → AI 매칭 → 추천 카드(스와이프)
import { CLASSES, HOBBIES, MBTI_TYPES, REGIONS } from "./config.js";
import { DEMO_CANDIDATES } from "./candidates.js";
import * as db from "./db.js";
import { argmax, bioTags, interestTags, prefVector, reasons, topN } from "./matching.js";
import { PREVIEW_MODE, classify, confidenceIssue, detectAndCrop, initVision } from "./vision.js";

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const typeOf = p => CLASSES.find(c => c.id === p.faceType) ?? CLASSES[argmax(p.face)];

const STORE_KEY = "animal-match.v1";
const state = {
  me: {                       // 사진은 저장하지 않는다. 확률 벡터와 프로필만 보관.
    nickname: "", gender: null, age: 25, region: REGIONS[0], seek: null, ageMin: 20, ageMax: 32,
    hobbies: [], bio: "", mbti: null, prefMbti: [], prefRank: [], face: null, faceType: null,
  },
  faceImg: null,              // 이번 세션에서만 쓰는 내 얼굴 썸네일 (저장 안 함)
  matches: [], deckIndex: 0, decisions: {}, source: "", mode: "demo",
  history: [], editing: false, user: null, visionReady: null,
};

// ───────── 저장 (브라우저에만) ─────────
function persist() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ me: state.me, decisions: state.decisions })); } catch { /* 저장 불가 환경 */ }
}
function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (saved?.me?.face && saved.me.nickname && saved.me.prefRank?.length) { Object.assign(state.me, saved.me); state.decisions = saved.decisions ?? {}; return true; }
  } catch { /* 무시 */ }
  return false;
}

// ───────── 화면 전환 ─────────
let current = "s-welcome";
function go(id, { push = true } = {}) {
  if (push && current !== id) state.history.push(current);
  document.querySelectorAll(".screen").forEach(s => (s.hidden = s.id !== id));
  current = id;
  if (id === "s-basic") fillBasic();
  if (id === "s-pref") renderPref();
  if (id === "s-bio") fillBio();
  if (id !== "s-photo") stopCamera();
}
document.querySelectorAll("[data-back]").forEach(b => b.addEventListener("click", () => {
  if (state.editing) { state.editing = false; state.history = []; go("s-app", { push: false }); return; }
  go(state.history.pop() ?? "s-welcome", { push: false });
}));
function banner(msg) { $("banner").hidden = !msg; $("banner").textContent = msg || ""; }

function toast(msg) {
  const t = Object.assign(document.createElement("div"), { textContent: msg });
  Object.assign(t.style, { position: "fixed", left: "50%", bottom: "96px", transform: "translateX(-50%)", background: "rgba(20,15,25,.88)",
    color: "#fff", padding: "10px 16px", borderRadius: "999px", fontSize: "14px", zIndex: 50, maxWidth: "90vw", textAlign: "center" });
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2200);
}

// ───────── 시작 ─────────
$("btn-start").addEventListener("click", () => go("s-photo"));
$("btn-resume").addEventListener("click", () => runMatching("demo"));

// ───────── ① 사진 → 동물상 ─────────
function status(msg, retake = false) {
  $("vision-status").textContent = msg;
  $("vision-status").classList.toggle("retake", retake);
}

$("consent").addEventListener("change", e => {
  const ok = e.target.checked;
  $("btn-camera").disabled = !ok;
  $("file").disabled = !ok;
  $("lbl-upload").setAttribute("aria-disabled", String(!ok));
  if (ok && !state.visionReady) {
    state.visionReady = initVision(msg => status(msg))
      .then(({ mockModel }) => {
        if (PREVIEW_MODE) banner("🧪 체험판 · 얼굴 검출·AI 모델 없이 임시 동물상 결과가 나와요");
        else if (mockModel) banner("⚠️ 모델 학습 중 · 지금은 임시(가짜) 동물상 결과가 나와요");
      })
      .catch(err => {
        console.error(err);
        status("모델을 불러오지 못했습니다: " + err.message, true);
        state.visionReady = null;
        throw err;
      });
  }
});

let stream = null;
function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $("video").hidden = true; $("face-guide").hidden = true; $("btn-shot").hidden = true;
  $("btn-camera").hidden = PREVIEW_MODE;
}

$("btn-camera").addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } }, audio: false });
    $("video").srcObject = stream;
    await $("video").play();
    $("video").hidden = false; $("face-guide").hidden = false;
    $("cam-placeholder").hidden = true; $("face").hidden = true;
    $("btn-camera").hidden = true; $("btn-shot").hidden = false;
    status("얼굴을 점선 안에 맞추고 촬영하기를 눌러 주세요.");
  } catch (err) {
    status("카메라를 열 수 없어요 (" + err.name + "). 사진 올리기를 이용해 주세요.", true);
  }
});

$("btn-shot").addEventListener("click", () => {
  const v = $("video"), c = $("frame");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  analyze(c);
});

$("file").addEventListener("change", e => {
  const f = e.target.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => {
    const c = $("frame");
    const scale = Math.min(1, 1280 / Math.max(img.width, img.height));
    c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(img.src);
    analyze(c);
  };
  img.src = URL.createObjectURL(f);
  e.target.value = "";
});

async function analyze(frameCanvas) {
  const retakeTitle = stream ? "📷 다시 찍어 주세요" : "🖼️ 다른 사진을 올려 주세요";
  $("scan").hidden = false;
  try {
    status("AI가 얼굴을 분석하는 중…");
    await state.visionReady;
    const { face, issue } = detectAndCrop(frameCanvas, $("face"));
    if (issue) { status(`${retakeTitle} · ${issue}`, true); return; }   // 구분이 불가능한 사진 -> 재촬영 요청
    const probs = await classify(face);
    const lowConf = confidenceIssue(probs);
    if (lowConf) { status(`${retakeTitle} · ${lowConf}`, true); return; }
    await sleep(500);   // 스캔 연출
    state.me.face = probs;
    state.me.faceType = CLASSES[argmax(probs)].id;
    state.faceImg = face.toDataURL("image/jpeg", 0.85);
    stopCamera();
    $("face").hidden = false; $("cam-placeholder").hidden = true;
    $("btn-camera").textContent = "📷 다시 찍기";
    renderFaceResult(probs);
    status("");
    $("btn-photo-next").disabled = false;
  } catch (err) {
    console.error(err);
    status("분석 중 오류: " + err.message, true);
  } finally {
    $("scan").hidden = true;
  }
}

function renderFaceResult(probs) {
  const top = CLASSES[argmax(probs)];
  $("type-emoji").textContent = top.emoji;
  $("type-name").textContent = top.ko;
  $("type-desc").textContent = top.desc;
  $("prob-list").innerHTML = CLASSES
    .map((c, i) => ({ c, p: probs[i] }))
    .sort((a, b) => b.p - a.p)
    .map(({ c, p }) => `<li><span>${c.emoji} ${c.ko}</span><span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span><span>${(p * 100).toFixed(1)}%</span></li>`)
    .join("");
  $("face-result").hidden = false;
}

$("btn-photo-next").addEventListener("click", () => nextStep("s-basic"));

/** 다음 단계로. 프로필 탭에서 수정 중이면 바로 다시 매칭한다. */
function nextStep(id) {
  persist();
  if (state.editing) { state.editing = false; state.history = []; runMatching(state.mode); return; }
  go(id);
}

// ───────── ② 기본 정보 ─────────
$("region-select").innerHTML = REGIONS.map(r => `<option>${r}</option>`).join("");
const basicForm = $("basic-form");
basicForm.addEventListener("change", e => {
  if (e.target.name === "gender" && !basicForm.querySelector("input[name=seek]:checked")) {
    basicForm.querySelector(`input[name=seek][value=${e.target.value === "M" ? "F" : "M"}]`).checked = true;
  }
});
function fillBasic() {
  const m = state.me, f = basicForm.elements;
  f.nickname.value = m.nickname; f.age.value = m.age; f.region.value = m.region;
  $("age-min-range").value = m.ageMin; $("age-max-range").value = m.ageMax;
  syncAge();
  basicForm.querySelectorAll("input[name=gender]").forEach(r => (r.checked = r.value === m.gender));
  basicForm.querySelectorAll("input[name=seek]").forEach(r => (r.checked = r.value === m.seek));
}
// 선호 나이 듀얼 슬라이더: 두 손잡이를 같은 값으로 맞춰 주고, 숨은 입력(ageMin/ageMax)과 라벨, 막대 위치를 갱신
const AGE_MIN = 19, AGE_MAX = 99;
function syncAge(src) {
  const minR = $("age-min-range"), maxR = $("age-max-range");
  let a = Number(minR.value), b = Number(maxR.value);
  if (a > b) {                              // 손잡이가 서로 넘어가면 밀어낸다
    if (src === minR) { maxR.value = a; b = a; } else { minR.value = b; a = b; }
  }
  basicForm.elements.ageMin.value = a;
  basicForm.elements.ageMax.value = b;
  $("age-label").textContent = `${a} ~ ${b}세`;
  $("age-fill").style.left = ((a - AGE_MIN) / (AGE_MAX - AGE_MIN) * 100) + "%";
  $("age-fill").style.width = ((b - a) / (AGE_MAX - AGE_MIN) * 100) + "%";
}
$("age-min-range").addEventListener("input", e => syncAge(e.target));
$("age-max-range").addEventListener("input", e => syncAge(e.target));
syncAge();

$("btn-basic-next").addEventListener("click", () => {
  const f = new FormData(basicForm);
  const v = {
    nickname: String(f.get("nickname") || "").trim(), gender: f.get("gender"), age: Number(f.get("age")),
    region: f.get("region"), seek: f.get("seek"), ageMin: Number(f.get("ageMin")), ageMax: Number(f.get("ageMax")),
  };
  const err = !v.nickname ? "닉네임을 입력해 주세요."
    : !v.gender ? "성별을 골라 주세요."
    : !(v.age >= 19 && v.age <= 99) ? "나이는 19~99세로 입력해 주세요."
    : !v.seek ? "만나고 싶은 상대를 골라 주세요."
    : !(v.ageMin >= 19 && v.ageMax <= 99 && v.ageMin <= v.ageMax) ? "선호 나이 범위를 확인해 주세요." : "";
  $("basic-msg").textContent = err;
  if (err) return;
  Object.assign(state.me, v);
  nextStep("s-pref");
});

// ───────── ③ 이상형 · MBTI ─────────
const AXES = [["E", "외향"], ["I", "내향"], ["S", "감각"], ["N", "직관"], ["T", "사고"], ["F", "감정"], ["J", "판단"], ["P", "인식"]];
let mbtiDraft = ["", "", "", ""];

function renderPref() {
  const m = state.me;
  $("pref-grid").innerHTML = "";
  CLASSES.forEach((c, i) => {
    const r = m.prefRank.indexOf(i);
    const b = Object.assign(document.createElement("button"), { type: "button", className: "animal" + (r >= 0 ? " on" : "") });
    b.innerHTML = (r >= 0 ? `<span class="rank">${r + 1}</span>` : "") + `<span class="e">${c.emoji}</span>${c.ko}`;
    b.onclick = () => {
      if (r >= 0) m.prefRank.splice(r, 1);
      else if (m.prefRank.length < 3) m.prefRank.push(i);
      else toast("이상형은 최대 3개까지 고를 수 있어요");
      renderPref();
    };
    $("pref-grid").appendChild(b);
  });

  if (m.mbti) mbtiDraft = [...m.mbti];
  $("mbti-picker").innerHTML = [0, 1, 2, 3].map(ax => `<div class="mbti-axis">${AXES.slice(ax * 2, ax * 2 + 2)
    .map(([l, ko]) => `<button type="button" data-ax="${ax}" data-l="${l}" class="${mbtiDraft[ax] === l ? "on" : ""}">${l}<small>${ko}</small></button>`).join("")}</div>`).join("");
  $("mbti-picker").classList.toggle("disabled", $("mbti-unknown").checked);

  $("pref-mbti").innerHTML = MBTI_TYPES.map(t => `<button type="button" class="chip${m.prefMbti.includes(t) ? " on" : ""}" data-t="${t}">${t}</button>`).join("");
}
$("mbti-picker").addEventListener("click", e => {
  const b = e.target.closest("button[data-ax]");
  if (!b) return;
  mbtiDraft[b.dataset.ax] = b.dataset.l;
  if (mbtiDraft.every(Boolean)) state.me.mbti = mbtiDraft.join("");
  renderPref();
});
$("mbti-unknown").addEventListener("change", e => {
  if (e.target.checked) { state.me.mbti = null; mbtiDraft = ["", "", "", ""]; }
  renderPref();
});
$("pref-mbti").addEventListener("click", e => {
  const t = e.target.closest("[data-t]")?.dataset.t;
  if (!t) return;
  const list = state.me.prefMbti, i = list.indexOf(t);
  if (i >= 0) list.splice(i, 1);
  else if (list.length < 3) list.push(t);
  else toast("선호 MBTI는 최대 3개까지예요");
  renderPref();
});
$("btn-pref-next").addEventListener("click", () => {
  const unknown = $("mbti-unknown").checked;
  const err = state.me.prefRank.length === 0 ? "이상형 동물상을 1개 이상 골라 주세요."
    : !unknown && !mbtiDraft.every(Boolean) ? "MBTI 네 글자를 모두 골라 주세요. (모르면 '잘 몰라요' 체크)" : "";
  $("pref-msg").textContent = err;
  if (err) return;
  state.me.mbti = unknown ? null : mbtiDraft.join("");
  nextStep("s-bio");
});

// ───────── ④ 자기소개 · 관심사 ─────────
function renderBioTags() {
  const text = $("bio").value;
  $("bio-count").textContent = `${text.length}/300`;
  const tags = bioTags(text);
  $("bio-tags").innerHTML = tags.length
    ? tags.map(t => `<span class="tag">#${t}</span>`).join(" ") + ` <span class="muted small">을(를) 찾았어요</span>`
    : `<span class="muted small">${text.trim() ? "아직 찾은 관심사가 없어요. 취미를 조금 더 적어 볼까요?" : "자기소개에서 관심사를 찾아 드려요"}</span>`;
}
function fillBio() {
  $("bio").value = state.me.bio;
  renderBioTags();
  $("hobby-chips").innerHTML = HOBBIES.map(h => `<button type="button" class="chip${state.me.hobbies.includes(h) ? " on" : ""}" data-h="${h}">${h}</button>`).join("");
  $("btn-find-real").hidden = !(db.dbEnabled && state.user);
}
$("bio").addEventListener("input", () => { state.me.bio = $("bio").value; renderBioTags(); });
$("hobby-chips").addEventListener("click", e => {
  const b = e.target.closest("[data-h]");
  if (!b) return;
  const h = b.dataset.h, list = state.me.hobbies, i = list.indexOf(h);
  i >= 0 ? list.splice(i, 1) : list.push(h);
  b.classList.toggle("on");
});
const onFind = e => {
  if (!state.me.face) { $("bio-msg").textContent = "먼저 사진으로 동물상을 분석해 주세요."; return; }
  if (state.me.bio.trim().length < 10) { $("bio-msg").textContent = "자기소개를 10자 이상 적어 주세요."; return; }
  $("bio-msg").textContent = "";
  persist();
  state.editing = false;
  runMatching(e.currentTarget.dataset.mode);
};
$("btn-find").addEventListener("click", onFind);
$("btn-find-real").addEventListener("click", onFind);

// ───────── AI 매칭 ─────────
function profileForMatch() {
  const m = state.me;
  return { ...m, id: state.user?.id ?? "me", pref: prefVector(m.prefRank, CLASSES.length) };
}

async function runMatching(mode = "demo") {
  state.history = [];
  go("s-loading", { push: false });
  $("loading-emoji").textContent = typeOf(state.me).emoji;
  const steps = [...$("loading-steps").children];
  steps.forEach(li => li.classList.remove("on", "done"));
  const me = profileForMatch();
  let pool, source;
  try {
    const work = (async () => {
      if (mode === "real") {
        await db.saveProfile(me);
        const p = await db.fetchOtherProfiles();
        return [p, `실제 회원 ${p.length}명 중`];
      }
      return [DEMO_CANDIDATES, `데모 후보 ${DEMO_CANDIDATES.length}명 중`];
    })();
    for (const li of steps) { li.classList.add("on"); await sleep(420); li.classList.add("done"); }
    [pool, source] = await work;
  } catch (err) {
    console.error(err);
    toast("오류: " + err.message);
    go("s-bio", { push: false });
    return;
  }
  state.mode = mode;
  state.matches = topN(me, pool, 5);
  state.deckIndex = 0;
  state.source = source;
  $("match-source").textContent = `${source} Top ${state.matches.length}`;
  go("s-app", { push: false });
  switchTab("tab-discover");
  renderDeck();
  renderMatchesTab();
  renderProfileTab();
}

// ───────── 추천 카드 (스와이프) ─────────
const pct = x => Math.round(x * 100);

function cardHTML(m, rank) {
  const c = m.candidate, t = typeOf(c);
  const me = profileForMatch();
  const tags = interestTags(c);
  return `
    <div class="stamp like">LIKE</div><div class="stamp nope">NOPE</div>
    <div class="hero g-${t.id}">
      <span class="rank-pill">추천 ${rank}위</span>
      <div class="big">${t.emoji}</div>
      <div class="who"><b>${esc(c.nickname)}</b> <span>${c.age}</span><div>${esc(c.region ?? "")}${c.region ? " · " : ""}${t.ko}</div></div>
      <div class="score-ring" style="--p:${pct(m.total)}"><div>${pct(m.total)}<small>매칭점수</small></div></div>
    </div>
    <div class="body">
      <div class="badges">
        <span class="badge">${t.emoji} ${t.ko}</span>
        ${c.mbti ? `<span class="badge mbti">${esc(c.mbti)}</span>` : ""}
        ${tags.slice(0, 4).map(x => `<span class="badge">#${esc(x)}</span>`).join("")}
      </div>
      ${c.bio ? `<p class="bio">${esc(c.bio)}</p>` : ""}
      <div class="report">
        <div class="ai-tags" style="margin:0"><span class="ai-badge">AI</span><b>매칭 리포트</b></div>
        <ul>${reasons(me, m).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
      </div>
      ${breakdownHTML(m)}
    </div>`;
}

function breakdownHTML(m) {
  const row = (label, v) => `<div class="row"><span>${label}</span><span class="bar"><i style="width:${pct(v)}%"></i></span><span>${pct(v)}%</span></div>`;
  return `<div class="breakdown">
    ${row("내 이상형 적합", m.fitAB)}
    ${row("상대 이상형 적합", m.fitBA)}
    ${row("MBTI 궁합", m.mbti)}
    ${row("관심사 겹침", m.interest)}
  </div>`;
}

function renderDeck() {
  const deck = $("deck");
  deck.innerHTML = "";
  const rest = state.matches.slice(state.deckIndex, state.deckIndex + 3);
  const empty = rest.length === 0;
  $("deck-empty").hidden = !empty;
  $("deck-actions").hidden = empty;
  deck.hidden = empty;
  if (state.matches.length === 0) {
    $("deck-empty").querySelector("h3").textContent = "조건에 맞는 상대가 아직 없어요";
    $("deck-empty").querySelector("p").textContent = "선호 나이나 상대 성별 조건을 넓혀 보세요.";
    return;
  }
  $("deck-empty").querySelector("h3").textContent = `오늘의 추천 Top ${state.matches.length}을 다 봤어요`;
  $("deck-empty").querySelector("p").textContent = "좋아요 누른 상대는 매칭 탭에서 확인할 수 있어요.";
  rest.slice().reverse().forEach((m, k) => {
    const depth = rest.length - 1 - k;              // 0 = 맨 위 카드
    const card = Object.assign(document.createElement("article"), { className: "card" });
    card.innerHTML = cardHTML(m, state.deckIndex + depth + 1);
    card.style.transform = `translateY(${depth * 10}px) scale(${1 - depth * 0.04})`;
    card.style.zIndex = 10 - depth;
    if (depth === 0) attachSwipe(card);
    else card.setAttribute("aria-hidden", "true");
    deck.appendChild(card);
  });
}

function setDrag(card, dx) {
  card.style.transform = `translateX(${dx}px) rotate(${dx / 18}deg)`;
  card.querySelector(".stamp.like").style.opacity = Math.max(0, Math.min(1, dx / 100));
  card.querySelector(".stamp.nope").style.opacity = Math.max(0, Math.min(1, -dx / 100));
}

function attachSwipe(card) {
  let id = null, sx = 0, sy = 0, dx = 0, dragging = false;
  card.addEventListener("pointerdown", e => { if (e.button) return; id = e.pointerId; sx = e.clientX; sy = e.clientY; dx = 0; dragging = false; });
  card.addEventListener("pointermove", e => {
    if (e.pointerId !== id) return;
    const mx = e.clientX - sx, my = e.clientY - sy;
    if (!dragging) {
      if (Math.abs(mx) > 8 && Math.abs(mx) > Math.abs(my)) { dragging = true; card.setPointerCapture(id); card.classList.add("dragging"); }
      else return;
    }
    dx = mx;
    setDrag(card, dx);
  });
  const end = e => {
    if (e.pointerId !== id) return;
    id = null;
    if (!dragging) return;
    dragging = false;
    card.classList.remove("dragging");
    if (Math.abs(dx) > 100) decide(dx > 0 ? "like" : "pass");
    else setDrag(card, 0);
  };
  card.addEventListener("pointerup", end);
  card.addEventListener("pointercancel", end);
}

let deciding = false;
async function decide(kind) {
  const m = state.matches[state.deckIndex];
  if (!m || deciding) return;
  deciding = true;
  const card = $("deck").lastElementChild;
  card.classList.remove("dragging");
  setDrag(card, kind === "like" ? 160 : -160);
  card.style.transform = `translateX(${kind === "like" ? 150 : -150}%) rotate(${kind === "like" ? 22 : -22}deg)`;
  card.style.opacity = "0";
  const matched = kind === "like" && isMutual(m);
  state.decisions[m.candidate.id] = matched ? "match" : kind;
  persist();
  await sleep(260);
  state.deckIndex++;
  renderDeck();
  renderMatchesTab();
  deciding = false;
  if (matched) showItsAMatch(m);
  else if (kind === "like") toast(state.mode === "real" ? "좋아요를 보냈어요 💌" : "좋아요를 보냈어요 💌 상대의 답을 기다려요");
}

// 데모: 내가 상대의 이상형에 어느 정도 가까우면(상대 이상형 적합 35% 이상) 상대도 좋아요를 눌렀다고 가정한다.
// 실제 회원 모드에서는 상대의 실제 응답이 필요하므로 매칭 성사를 흉내 내지 않는다.
const isMutual = m => state.mode === "demo" && (m.fitBA >= 0.35 || m.total >= 0.6);

$("btn-like").addEventListener("click", () => decide("like"));
$("btn-pass").addEventListener("click", () => decide("pass"));
$("btn-info").addEventListener("click", () => { const m = state.matches[state.deckIndex]; if (m) showDetail(m); });
$("btn-replay").addEventListener("click", () => { state.deckIndex = 0; renderDeck(); });
$("btn-goto-matches").addEventListener("click", () => switchTab("tab-matches"));
document.addEventListener("keydown", e => {
  if (current !== "s-app" || $("tab-discover").hidden || document.querySelector("dialog[open]")) return;
  if (e.key === "ArrowRight") decide("like");
  if (e.key === "ArrowLeft") decide("pass");
});

// ───────── 매칭 성공 · 상세 ─────────
const ICEBREAKERS = {
  여행: "최근에 다녀온 여행지 중에 제일 좋았던 곳 어디예요? ✈️", 카페: "요즘 가장 자주 가는 카페 어디예요? 추천받고 싶어요 ☕",
  맛집: "인생 맛집 하나만 공유해 주실 수 있나요? 🍜", 영화: "최근에 재밌게 본 영화나 드라마 있어요? 🎬",
  음악: "요즘 플레이리스트에 제일 많이 나오는 노래 뭐예요? 🎧", 운동: "운동 루틴 궁금해요! 어떤 운동 하세요? 💪",
  게임: "요즘 하는 게임 뭐예요? 같이 하면 재밌을 것 같아요 🎮", 독서: "최근에 읽은 책 중에 추천할 만한 거 있어요? 📚",
  사진: "인생샷 건지는 비법 좀 알려 주세요 📸", 요리: "제일 자신 있는 요리가 뭐예요? 🍳",
  반려동물: "혹시 반려동물 사진 보여 주실 수 있나요? 🐾", 등산: "가 본 산 중에 제일 좋았던 곳 어디예요? ⛰️",
  술: "와인파예요, 맥주파예요? 🍷", 전시: "요즘 볼 만한 전시 있으면 같이 가요! 🖼️", 집콕: "집에서 쉴 때 주로 뭐 하세요? 🛋️",
};
function icebreaker(m) {
  const tag = m.commonTags.find(t => ICEBREAKERS[t]);
  if (tag) return ICEBREAKERS[tag];
  return `저 ${typeOf(state.me).ko}래요 ㅎㅎ ${m.candidate.nickname}님은 ${typeOf(m.candidate).ko} 결과 인정하세요? 😆`;
}

function avatarHTML(el, p, img) {
  el.className = "pair-avatar g-" + typeOf(p).id;
  el.innerHTML = img ? `<img src="${img}" alt="" style="width:100%;height:100%;border-radius:50%;object-fit:cover">` : typeOf(p).emoji;
}
function showItsAMatch(m) {
  $("match-text").textContent = `${m.candidate.nickname}님도 당신을 좋아해요! (매칭 점수 ${pct(m.total)}점)`;
  avatarHTML($("pair-me"), state.me, state.faceImg);
  avatarHTML($("pair-them"), m.candidate);
  $("icebreaker-text").textContent = icebreaker(m);
  $("match-dialog").showModal();
}

function showDetail(m) {
  const c = m.candidate, t = typeOf(c), d = state.decisions[c.id];
  $("detail-body").innerHTML = `
    <div class="d-hero g-${t.id}">${t.emoji}</div>
    <div><h3 style="margin:0">${esc(c.nickname)}, ${c.age}</h3>
      <div class="muted small">${esc(c.region ?? "")} · ${t.ko}${c.mbti ? " · " + esc(c.mbti) : ""}</div></div>
    ${c.bio ? `<p class="bio">${esc(c.bio)}</p>` : ""}
    <div class="badges">${interestTags(c).map(x => `<span class="badge">#${esc(x)}</span>`).join("")}</div>
    <div class="report"><div class="ai-tags" style="margin:0"><span class="ai-badge">AI</span><b>매칭 리포트 · ${pct(m.total)}점</b></div>
      <ul>${reasons(profileForMatch(), m).map(r => `<li>${esc(r)}</li>`).join("")}</ul></div>
    ${breakdownHTML(m)}
    ${d === "match" ? `<div class="report"><div class="ai-tags" style="margin:0"><span class="ai-badge">AI</span><b>추천 첫 마디</b></div><div>${esc(icebreaker(m))}</div></div>` : ""}
    <p class="muted small" style="margin:0">점수 = 0.6 × 동물상 상호 적합도 + 0.2 × MBTI 궁합 + 0.2 × 관심사 겹침</p>`;
  $("detail-dialog").showModal();
}

// ───────── 탭 ─────────
function switchTab(id) {
  document.querySelectorAll("#s-app .tab").forEach(t => (t.hidden = t.id !== id));
  document.querySelectorAll(".tabbar button").forEach(b => b.classList.toggle("on", b.dataset.tab === id));
  if (id === "tab-matches") $("match-dot").hidden = true;
}
document.querySelectorAll(".tabbar button").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

const STATE_LABEL = { match: "💘 매칭 성공", like: "💌 좋아요 보냄", pass: "넘김" };
function renderMatchesTab() {
  const liked = state.matches.filter(m => ["like", "match"].includes(state.decisions[m.candidate.id]));
  $("match-strip").innerHTML = liked.length
    ? liked.map(m => {
        const t = typeOf(m.candidate), mt = state.decisions[m.candidate.id] === "match";
        return `<button class="strip-item${mt ? " matched" : ""}" data-id="${esc(m.candidate.id)}"><span class="av g-${t.id}">${t.emoji}</span>${esc(m.candidate.nickname)}</button>`;
      }).join("")
    : `<div class="empty">아직 좋아요를 누른 상대가 없어요. 추천 탭에서 오른쪽으로 넘겨 보세요 →</div>`;
  if (liked.some(m => state.decisions[m.candidate.id] === "match") && $("tab-matches").hidden) $("match-dot").hidden = false;

  $("rank-list").innerHTML = state.matches.length
    ? state.matches.map((m, i) => {
        const c = m.candidate, t = typeOf(c);
        return `<li class="rank-item" data-id="${esc(c.id)}">
          <span class="no">${i + 1}</span>
          <span class="av g-${t.id}">${t.emoji}</span>
          <div><div class="name">${esc(c.nickname)} <span class="muted small">${c.age}세</span></div>
            <div class="meta">${t.ko}${c.mbti ? " · " + esc(c.mbti) : ""} · 내 이상형 ${pct(m.fitAB)}% · 상대 이상형 ${pct(m.fitBA)}%</div></div>
          <div class="pts">${pct(m.total)}<span class="state">${STATE_LABEL[state.decisions[c.id]] ?? ""}</span></div>
        </li>`;
      }).join("")
    : `<li class="muted small">조건(성별·나이)에 맞는 상대가 없어요.</li>`;
}
const openById = e => {
  const id = e.target.closest("[data-id]")?.dataset.id;
  const m = state.matches.find(x => x.candidate.id === id);
  if (m) showDetail(m);
};
$("match-strip").addEventListener("click", openById);
$("rank-list").addEventListener("click", openById);

function renderProfileTab() {
  const m = state.me, t = typeOf(m);
  const prefs = m.prefRank.map((i, r) => `${r + 1}. ${CLASSES[i].emoji} ${CLASSES[i].ko}`).join("  ");
  const seek = { M: "남자", F: "여자", any: "상관없음" }[m.seek] ?? "";
  $("me-card").innerHTML = `
    <div class="hero g-${t.id}">
      <div class="me-av">${state.faceImg ? `<img src="${state.faceImg}" alt="" style="width:100%;height:100%;border-radius:24px;object-fit:cover">` : t.emoji}</div>
      <div><b>${esc(m.nickname)}</b>, ${m.age}<div>${esc(m.region)} · ${t.emoji} ${t.ko}${m.mbti ? " · " + m.mbti : ""}</div></div>
    </div>
    <div class="info">
      <div><b>자기소개</b><div class="muted">${esc(m.bio)}</div></div>
      <div><b>관심사</b><div class="badges" style="margin-top:4px">${interestTags(m).map(x => `<span class="badge">#${esc(x)}</span>`).join("") || '<span class="muted">-</span>'}</div></div>
      <div><b>이상형 동물상</b><div class="muted">${prefs}</div></div>
      <div><b>찾는 상대</b><div class="muted">${seek} · ${m.ageMin}~${m.ageMax}세${m.prefMbti.length ? " · 선호 MBTI " + m.prefMbti.join(", ") : ""}</div></div>
    </div>`;
}
document.querySelectorAll("[data-edit]").forEach(b => b.addEventListener("click", () => {
  state.editing = true;
  go(b.dataset.edit, { push: false });
}));
let resetArmed = false;
$("btn-reset").addEventListener("click", () => {
  if (!resetArmed) {                       // confirm() 대신 한 번 더 눌러 확인
    resetArmed = true;
    $("btn-reset").textContent = "⚠️ 한 번 더 누르면 내 정보가 지워져요";
    setTimeout(() => { resetArmed = false; $("btn-reset").textContent = "🗑️ 내 정보 지우고 처음부터"; }, 4000);
    return;
  }
  try { localStorage.removeItem(STORE_KEY); } catch { /* 무시 */ }
  location.reload();
});

// ───────── 로그인 (Supabase 설정 시에만) ─────────
function renderAuth() {
  const html = !db.dbEnabled
    ? `<span>데모 모드 · 가상 후보 40명과 매칭해요</span>`
    : state.user
      ? `<span>${esc(state.user.email)}</span> <button class="btn" data-logout>로그아웃</button>`
      : `<button class="btn" data-login>로그인하고 실제 회원과 매칭</button>`;
  ["auth-area", "auth-area-2"].forEach(id => ($(id).innerHTML = html));
  document.querySelectorAll("[data-logout]").forEach(b => (b.onclick = () => db.signOut()));
  document.querySelectorAll("[data-login]").forEach(b => (b.onclick = () => { $("auth-msg").textContent = ""; $("auth-dialog").showModal(); }));
  $("btn-find-real").hidden = !(db.dbEnabled && state.user);
}

$("auth-form").addEventListener("submit", async e => {
  const action = e.submitter?.value;
  if (action === "cancel") return;
  e.preventDefault();
  const f = new FormData(e.target);
  const email = String(f.get("email")), pw = String(f.get("password"));
  try {
    if (action === "signup") {
      const { session } = await db.signUp(email, pw);
      $("auth-msg").textContent = session ? "가입 완료! 로그인되었습니다." : "가입 완료! 메일함에서 인증 후 로그인해 주세요.";
      if (session) $("auth-dialog").close();
    } else {
      await db.signIn(email, pw);
      $("auth-dialog").close();
    }
  } catch (err) {
    $("auth-msg").textContent = "오류: " + err.message;
  }
});

// ───────── 초기화 ─────────
if (PREVIEW_MODE) {                        // 체험판: 카메라 없이 사진 올리기만
  $("btn-camera").hidden = true;
  $("cam-placeholder").lastElementChild.textContent = "사진을 올려 주세요 (체험판은 카메라를 쓰지 않아요)";
}
(async () => {
  if (restore()) {
    $("btn-resume").hidden = false;
    $("btn-resume").textContent = `${typeOf(state.me).emoji} ${state.me.nickname}(으)로 이어하기`;
  }
  renderAuth();
  if (!db.dbEnabled) return;
  try {
    state.user = await db.currentUser();
    renderAuth();
    await db.onAuthChange(u => { state.user = u; renderAuth(); });
  } catch (err) {
    console.error(err);
  }
})();
