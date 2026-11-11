// 매칭 점수 계산 (순수 함수 -> node 로도 테스트 가능)
//
// ① 동물상 (상호)
//    fit(A→B)  = (A의 이상형 벡터 · B의 얼굴 확률 벡터) / max(A의 이상형 벡터)
//                → B가 A의 1순위 동물상 100% 이면 1.0
//    face      = (fit(A→B) + fit(B→A)) / 2        ← 양쪽 취향을 모두 반영
// ② MBTI (상호)
//    A가 선호 MBTI를 골랐으면: B의 MBTI와 가장 비슷한 선호 유형의 일치 글자 수 / 4
//    고르지 않았으면: 기본 궁합표(축별 점수 합, 0~1)
//    mbti      = (mbtiFit(A→B) + mbtiFit(B→A)) / 2
// ③ 관심사
//    태그 = 취미 칩 ∪ 자기소개에서 찾은 키워드 태그
//    interest  = 자카드 유사도 |공통 태그| / |합집합|
// 최종 점수 = 0.6 * face + 0.2 * mbti + 0.2 * interest  (0~100점으로 표시, 가중치는 config.js)
// 성별·나이 조건이 서로 맞지 않으면 후보에서 제외
import { CLASSES, INTEREST_KEYWORDS, MATCH_WEIGHTS, PREF_RANK_WEIGHTS } from "./config.js";

export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const argmax = v => v.reduce((bi, x, i) => (x > v[bi] ? i : bi), 0);

/** 이상형 순위(클래스 인덱스 배열, 최대 3개) -> 확률 벡터 */
export function prefVector(rankedIdx, numClasses = 10) {
  const v = new Array(numClasses).fill(0);
  const ws = PREF_RANK_WEIGHTS.slice(0, rankedIdx.length);
  const total = ws.reduce((a, b) => a + b, 0);
  rankedIdx.forEach((c, r) => { v[c] = ws[r] / total; });
  return v;
}

export function fit(pref, face) {
  const mx = Math.max(...pref);
  return mx > 0 ? dot(pref, face) / mx : 0;
}

// ───────── MBTI ─────────
const isMbti = t => typeof t === "string" && /^[EI][SN][TF][JP]$/.test(t);

/** 선호 유형을 고르지 않았을 때 쓰는 기본 궁합 (재미용 휴리스틱, 합 최대 1.0) */
export function mbtiBase(a, b) {
  if (!isMbti(a) || !isMbti(b)) return 0.5;
  return (a[0] !== b[0] ? 0.25 : 0.15)   // E/I: 서로 다르면 보완
       + (a[1] === b[1] ? 0.30 : 0.05)   // S/N: 같으면 대화가 잘 통함
       + (a[2] !== b[2] ? 0.20 : 0.15)   // T/F: 다르면 균형
       + (a[3] === b[3] ? 0.25 : 0.10);  // J/P: 같으면 생활 패턴이 맞음
}

export function mbtiFit(prefList = [], target) {
  if (!isMbti(target)) return 0.5;                 // 상대가 MBTI를 모르면 중립
  const prefs = prefList.filter(isMbti);
  if (prefs.length === 0) return null;             // 선호 없음 -> 기본 궁합 사용
  return Math.max(...prefs.map(p => [...p].filter((ch, i) => ch === target[i]).length / 4));
}

export function mbtiScore(a, b) {
  const ab = mbtiFit(a.prefMbti, b.mbti) ?? mbtiBase(a.mbti, b.mbti);
  const ba = mbtiFit(b.prefMbti, a.mbti) ?? mbtiBase(b.mbti, a.mbti);
  return (ab + ba) / 2;
}

// ───────── 관심사 (취미 + 자기소개 키워드) ─────────
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
  const union = new Set([...A, ...B]).size;
  return { score: union ? common.length / union : 0, common };
}

// ───────── 종합 ─────────
const wants = (p, q) => (p.seek === "any" || p.seek === q.gender) && q.age >= p.ageMin && q.age <= p.ageMax;
export const eligible = (a, b) => a.id !== b.id && wants(a, b) && wants(b, a);

export function matchScore(a, b) {
  const fitAB = fit(a.pref, b.face);
  const fitBA = fit(b.pref, a.face);
  const face = (fitAB + fitBA) / 2;
  const mbti = mbtiScore(a, b);
  const it = interestScore(a, b);
  const total = MATCH_WEIGHTS.face * face + MATCH_WEIGHTS.mbti * mbti + MATCH_WEIGHTS.interest * it.score;
  return { total, fitAB, fitBA, face, mbti, interest: it.score, commonTags: it.common };
}

export function topN(me, candidates, n = 5) {
  return candidates
    .filter(c => eligible(me, c))
    .map(c => ({ candidate: c, ...matchScore(me, c) }))
    .sort((x, y) => y.total - x.total)
    .slice(0, n);
}

/** 추천 이유 문장 (카드에 보여 줄 'AI 매칭 리포트') */
export function reasons(me, m) {
  const c = m.candidate;
  const out = [];
  const myType = CLASSES[argmax(me.face)], theirType = CLASSES[argmax(c.face)];
  if (m.fitAB >= 0.6 && m.fitBA >= 0.6) out.push(`서로가 서로의 이상형이에요! ${theirType.emoji}${myType.emoji}`);
  else if (m.fitAB >= 0.6) out.push(`${theirType.ko}, 딱 내가 찾던 얼굴상이에요`);
  else if (m.fitBA >= 0.6) out.push(`상대가 찾는 이상형이 바로 ${myType.ko}인 나예요`);
  else out.push(`동물상 상호 적합도 ${(m.face * 100).toFixed(0)}%`);

  if (isMbti(me.mbti) && isMbti(c.mbti)) {
    const mb = m.mbti >= 0.8 ? "찰떡 궁합" : m.mbti >= 0.6 ? "잘 맞는 궁합" : "서로 다른 매력";
    out.push(`${me.mbti} × ${c.mbti} · ${mb}`);
  }
  if (m.commonTags.length) out.push(`둘 다 좋아해요: ${m.commonTags.slice(0, 3).map(t => "#" + t).join(" ")}`);
  return out;
}
