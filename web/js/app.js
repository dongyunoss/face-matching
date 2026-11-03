// 화면 흐름: 1) 얼굴 입력 → 분류  2) 프로필·이상형 입력  3) Top-5 추천
import { CLASSES, HOBBIES } from "./config.js";
import { DEMO_CANDIDATES } from "./candidates.js";
import * as db from "./db.js";
import { argmax, prefVector, topN } from "./matching.js";
import { classify, detectAndCrop, initVision } from "./vision.js";

const $ = id => document.getElementById(id);
const state = { face: null, faceType: null, hobbies: new Set(), prefRank: [], user: null, visionReady: null };

// ───────── 공통 UI ─────────
function banner(msg) { $("banner").hidden = !msg; $("banner").textContent = msg || ""; }
function show(id) { $(id).hidden = false; }

// ───────── 1. 얼굴 분석 ─────────
$("consent").addEventListener("change", e => {
  const ok = e.target.checked;
  $("btn-camera").disabled = !ok;
  $("file").disabled = !ok;
  $("lbl-upload").setAttribute("aria-disabled", String(!ok));
  if (ok && !state.visionReady) {
    state.visionReady = initVision(msg => ($("vision-status").textContent = msg))
      .then(({ mockModel }) => {
        if (mockModel) banner("⚠️ 모델 파일(web/model/face_type.onnx)이 아직 없어 임시(가짜) 결과를 보여줍니다.");
      })
      .catch(err => {
        console.error(err);
        $("vision-status").textContent = "모델을 불러오지 못했습니다: " + err.message;
        state.visionReady = null;
        throw err;
      });
  }
});

let stream = null;
$("btn-camera").addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } }, audio: false });
    $("video").srcObject = stream;
    await $("video").play();
    show("camera-box");
  } catch (err) {
    $("vision-status").textContent = "카메라를 열 수 없습니다 (" + err.name + "). 사진 올리기를 이용해 주세요.";
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
  try {
    $("vision-status").textContent = "분석 중…";
    await state.visionReady;
    const face = detectAndCrop(frameCanvas, $("face"));
    if (!face) {
      $("vision-status").textContent = "얼굴을 찾지 못했어요. 정면을 보고 밝은 곳에서 다시 시도해 주세요.";
      return;
    }
    const probs = await classify(face);
    state.face = probs;
    state.faceType = CLASSES[argmax(probs)].id;
    renderFaceResult(probs);
    $("vision-status").textContent = "";
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; $("camera-box").hidden = true; }
    show("step-profile");
  } catch (err) {
    console.error(err);
    $("vision-status").textContent = "분석 중 오류: " + err.message;
  }
}

function renderFaceResult(probs) {
  const top = CLASSES[argmax(probs)];
  $("type-name").textContent = `${top.emoji} ${top.ko}`;
  $("type-desc").textContent = top.desc;
  $("prob-list").innerHTML = CLASSES
    .map((c, i) => ({ c, p: probs[i] }))
    .sort((a, b) => b.p - a.p)
    .map(({ c, p }) => `<li><span>${c.emoji} ${c.ko}</span><span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span><span>${(p * 100).toFixed(1)}%</span></li>`)
    .join("");
  show("face-result");
}

// ───────── 2. 프로필 & 이상형 ─────────
HOBBIES.forEach(h => {
  const b = Object.assign(document.createElement("button"), { type: "button", className: "chip", textContent: h });
  b.onclick = () => { state.hobbies.has(h) ? state.hobbies.delete(h) : state.hobbies.add(h); b.classList.toggle("on"); };
  $("hobby-chips").appendChild(b);
});

function renderPrefChips() {
  $("pref-chips").innerHTML = "";
  CLASSES.forEach((c, i) => {
    const r = state.prefRank.indexOf(i);
    const b = Object.assign(document.createElement("button"), { type: "button", className: "chip" + (r >= 0 ? " on" : "") });
    b.innerHTML = (r >= 0 ? `<span class="rank">${r + 1}</span>` : "") + `${c.emoji} ${c.ko}`;
    b.onclick = () => {
      if (r >= 0) state.prefRank.splice(r, 1);
      else if (state.prefRank.length < 3) state.prefRank.push(i);
      renderPrefChips();
    };
    $("pref-chips").appendChild(b);
  });
}
renderPrefChips();

function readProfile() {
  const f = new FormData($("profile-form"));
  return {
    id: state.user?.id ?? "me",
    nickname: String(f.get("nickname")).trim(),
    gender: f.get("gender"),
    age: Number(f.get("age")),
    seek: f.get("seek"),
    ageMin: Number(f.get("ageMin")),
    ageMax: Number(f.get("ageMax")),
    hobbies: [...state.hobbies],
    face: state.face,
    pref: prefVector(state.prefRank, CLASSES.length),
    faceType: state.faceType,
  };
}

$("profile-form").addEventListener("submit", async e => {
  e.preventDefault();
  const mode = e.submitter?.dataset.mode ?? "demo";
  const me = readProfile();
  const msg = $("form-msg");
  if (!state.face) { msg.textContent = "먼저 1단계에서 얼굴 분석을 해 주세요."; return; }
  if (state.prefRank.length === 0) { msg.textContent = "이상형 동물상을 1개 이상 골라 주세요."; return; }
  if (me.ageMin > me.ageMax) { msg.textContent = "선호 나이 범위를 확인해 주세요."; return; }
  msg.textContent = "";
  try {
    let pool, source;
    if (mode === "real") {
      msg.textContent = "저장 중…";
      await db.saveProfile(me);
      pool = await db.fetchOtherProfiles();
      source = `실제 회원 ${pool.length}명 중`;
      msg.textContent = "프로필을 저장했어요.";
    } else {
      pool = DEMO_CANDIDATES;
      source = `데모 후보 ${pool.length}명 중`;
    }
    renderMatches(topN(me, pool, 5), source);
  } catch (err) {
    console.error(err);
    msg.textContent = "오류: " + err.message;
  }
});

// ───────── 3. 추천 결과 ─────────
function renderMatches(list, source) {
  $("match-source").textContent = `(${source})`;
  const ul = $("match-list");
  if (list.length === 0) {
    ul.innerHTML = `<li class="muted">조건(성별·나이)에 맞는 상대가 없어요. 선호 나이를 넓혀 보세요.</li>`;
  } else {
    ul.innerHTML = list.map(({ candidate: c, total, fitAB, fitBA, commonHobbies }) => {
      const t = CLASSES.find(x => x.id === c.faceType) ?? CLASSES[argmax(c.face)];
      const hob = commonHobbies.length ? ` · 공통 취미: ${commonHobbies.join(", ")}` : "";
      return `<li class="match">
        <div class="avatar">${t.emoji}</div>
        <div>
          <div class="name">${escapeHtml(c.nickname)} <span class="muted">· ${c.age}세 · ${t.ko}</span></div>
          <div class="detail">내 이상형 적합 ${(fitAB * 100).toFixed(0)}% · 상대 이상형 적합 ${(fitBA * 100).toFixed(0)}%${escapeHtml(hob)}</div>
        </div>
        <div class="score">${(total * 100).toFixed(0)}점</div>
      </li>`;
    }).join("");
  }
  show("step-match");
  $("step-match").scrollIntoView({ behavior: "smooth" });
}

const escapeHtml = s => String(s).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

// ───────── 로그인 (Supabase 설정 시에만) ─────────
function renderAuth() {
  const area = $("auth-area");
  if (!db.dbEnabled) { area.innerHTML = `<span class="muted small">데모 모드</span>`; return; }
  $("btn-real").hidden = !state.user;
  if (state.user) {
    area.innerHTML = `<span class="muted">${escapeHtml(state.user.email)}</span> <button id="btn-logout">로그아웃</button>`;
    $("btn-logout").onclick = () => db.signOut();
  } else {
    area.innerHTML = `<button id="btn-login" class="primary">로그인</button>`;
    $("btn-login").onclick = () => { $("auth-msg").textContent = ""; $("auth-dialog").showModal(); };
  }
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

(async () => {
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
