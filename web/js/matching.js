// 추천 · 매칭 엔진 (순수 함수 -> node 로도 테스트 가능)
//
// 용어
//   추천(recommendation): 나에게 누구를 어떤 순서로 보여 줄지 정하는 것
//   매칭(match):          두 사람이 서로 좋아요를 누른 상태
//
// 1) 후보 조건 (하나라도 어긋나면 제외)
//    - 서로의 '찾는 성별'과 '선호 나이'에 맞음   - 이미 좋아요/넘기기 한 상대가 아님
//
// 2) 항목 점수 s_k(A→B) ∈ [0,1]  — A의 기준으로 본 B
//    외모     fit = (A의 이상형 벡터 · B의 얼굴 확률 벡터) / max(A의 이상형 벡터)
//    성격     A가 선호 MBTI를 골랐을 때만: 선호 유형들 중 B와 가장 많이 같은 글자 수 / 4
//    관심사   다이스 계수 2|A∩B| / (|A|+|B|)  (태그 = 관심사 칩 ∪ 자기소개 키워드)
//    거리     같은 시·도 1.0 / 같은 권역 0.7 / 그 외 0.3
//    계산할 수 없는 항목(선호 MBTI 없음, 지역 없음 등)은 빠진다.
//
// 3) 만족도 sat(A→B) = Σ w_k·s_k / Σ w_k   (w_k = A가 정한 항목 중요도)
//
// 4) 매칭 점수 = √( sat(A→B) × sat(B→A) )   ← 기하평균: 한쪽만 좋아하면 점수가 크게 깎인다
//
// 5) 추천 순서 = 매칭 점수 내림차순, 한 번에 DECK_SIZE 명
//
// 6) 취향 학습: 좋아요 n개 → 이상형 벡터 = (1-β)·고른 이상형 + β·좋아요한 얼굴 평균, β = n/(n+LEARN_K)
import {
  CLASSES, DEFAULT_IMPORTANCE, DISTANCE_SCORE, FACTORS, INTEREST_KEYWORDS, LEARN_K, PREF_RANK_WEIGHTS, REGION_GROUPS,
} from "./config.js";

export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const argmax = v => v.reduce((bi, x, i) => (x > v[bi] ? i : bi), 0);

/** 이상형 순위(클래스 인덱스 배열, 최대 3개) -> 합이 1인 벡터 */
export function prefVector(rankedIdx, numClasses = 10) {
  const v = new Array(numClasses).fill(0);
  const ws = PREF_RANK_WEIGHTS.slice(0, rankedIdx.length);
  const total = ws.reduce((a, b) => a + b, 0);
  rankedIdx.forEach((c, r) => { v[c] = ws[r] / total; });
  return v;
}

/** 외모 적합도: B가 A의 1순위 동물상 100% 이면 1.0 */
export function fit(pref, face) {
  const mx = Math.max(...pref);
  return mx > 0 ? dot(pref, face) / mx : 0;
}

/** 취향 학습: 고른 이상형과 좋아요한 상대들의 얼굴 평균을 섞는다 */
export function learnedPref(statedPref, likedFaces = []) {
  const n = likedFaces.length;
  if (n === 0) return statedPref;
  const mean = statedPref.map((_, i) => likedFaces.reduce((s, f) => s + f[i], 0) / n);
  const beta = n / (n + LEARN_K);
  return statedPref.map((p, i) => (1 - beta) * p + beta * mean[i]);
}

// ───────── 항목 점수 ─────────
const isMbti = t => typeof t === "string" && /^[EI][SN][TF][JP]$/.test(t);

export function mbtiScore(prefList = [], target) {
  const prefs = prefList.filter(isMbti);
  if (prefs.length === 0) return null;          // 선호 MBTI를 고르지 않음 -> 이 항목은 계산에서 빠짐
  if (!isMbti(target)) return 0.5;              // 상대가 MBTI를 모르면 중립
  return Math.max(...prefs.map(p => [...p].filter((ch, i) => ch === target[i]).length / 4));
}

/** 자기소개 문장 -> 관심사 태그 배열 */
export function bioTags(bio = "") {
  const text = String(bio).toLowerCase();
  return Object.entries(INTEREST_KEYWORDS)
    .filter(([, kws]) => kws.some(k => text.includes(k.toLowerCase())))
    .map(([tag]) => tag);
}
export const interestTags = p => [...new Set([...(p.hobbies ?? []), ...bioTags(p.bio)])];

export function interestScore(a, b) {
  const A = new Set(interestTags(a)), B = new Set(interestTags(b));
  const common = [...A].filter(x => B.has(x));
  if (A.size === 0 || B.size === 0) return { score: null, common };
  return { score: (2 * common.length) / (A.size + B.size), common };
}

const groupOf = r => Object.keys(REGION_GROUPS).find(g => REGION_GROUPS[g].includes(r));
export function distanceScore(r1, r2) {
  if (!r1 || !r2) return null;
  if (r1 === r2) return DISTANCE_SCORE.same;
  const g = groupOf(r1);
  return g && g === groupOf(r2) ? DISTANCE_SCORE.group : DISTANCE_SCORE.far;
}

// ───────── 만족도 · 매칭 점수 ─────────
/** A의 기준으로 본 B의 항목 점수 (null = 계산에서 빠짐) */
export function factorScores(a, b) {
  return {
    face: fit(a.pref, b.face),
    mbti: mbtiScore(a.prefMbti, b.mbti),
    interest: interestScore(a, b).score,
    distance: distanceScore(a.region, b.region),
  };
}

/** sat(A→B): A가 정한 중요도로 가중평균. 반영할 항목이 하나도 없으면 0.5 */
export function satisfaction(a, b) {
  const parts = factorScores(a, b);
  const imp = { ...DEFAULT_IMPORTANCE, ...(a.importance ?? {}) };
  let sw = 0, s = 0;
  for (const { id } of FACTORS) {
    const w = imp[id] ?? 0;
    if (w > 0 && parts[id] !== null) { sw += w; s += w * parts[id]; }
  }
  return { score: sw > 0 ? s / sw : 0.5, parts };
}

export function matchScore(a, b) {
  const ab = satisfaction(a, b), ba = satisfaction(b, a);
  return {
    total: Math.sqrt(ab.score * ba.score),
    satAB: ab.score, satBA: ba.score,
    partsAB: ab.parts, partsBA: ba.parts,
    commonTags: interestScore(a, b).common,
  };
}

// ───────── 추천 ─────────
const wants = (p, q) => (p.seek === "any" || p.seek === q.gender) && q.age >= p.ageMin && q.age <= p.ageMax;
export const eligible = (a, b) => a.id !== b.id && wants(a, b) && wants(b, a);

/** 후보 조건을 통과한 사람 중 매칭 점수 상위 n명. exclude = 이미 좋아요/넘기기 한 id */
export function recommend(me, candidates, n = 5, exclude = new Set()) {
  return candidates
    .filter(c => !exclude.has(c.id) && eligible(me, c))
    .map(c => ({ candidate: c, ...matchScore(me, c) }))
    .sort((x, y) => y.total - x.total)
    .slice(0, n);
}

/** 추천 이유 문장 (카드의 'AI 매칭 리포트') */
export function reasons(me, m) {
  const c = m.candidate, out = [];
  const myType = CLASSES[argmax(me.face)], theirType = CLASSES[argmax(c.face)];
  const pa = m.partsAB, pb = m.partsBA;
  if (m.satAB >= 0.6 && m.satBA >= 0.6) out.push("서로의 기준에 잘 맞는 상호 추천이에요 💞");
  else if (m.satAB > m.satBA) out.push(`내 기준 만족도 ${pct(m.satAB)}% · 상대 기준 ${pct(m.satBA)}%`);
  else out.push(`상대가 찾는 사람에 내가 잘 맞아요 (상대 기준 ${pct(m.satBA)}%)`);

  if (pa.face >= 0.6 && pb.face >= 0.6) out.push(`${theirType.emoji}${myType.emoji} 서로의 이상형 동물상이에요`);
  else if (pa.face >= 0.6) out.push(`${theirType.ko}, 내가 찾던 동물상이에요`);
  else if (pb.face >= 0.6) out.push(`상대 이상형이 바로 ${myType.ko}예요`);

  if (pa.mbti === 1) out.push(`${c.mbti}, 내가 고른 선호 MBTI예요`);
  if (m.commonTags.length) out.push(`둘 다 좋아해요: ${m.commonTags.slice(0, 3).map(t => "#" + t).join(" ")}`);
  if (pa.distance === 1) out.push(`같은 지역(${c.region})이에요`);
  return out;
}
const pct = x => Math.round(x * 100);
