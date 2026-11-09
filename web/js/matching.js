// 매칭 점수 계산 (순수 함수 -> node 로도 테스트 가능)
//
// fit(A→B)  = (A의 이상형 벡터 · B의 얼굴 확률 벡터) / max(A의 이상형 벡터)
//             → B가 A의 1순위 동물상 100% 이면 1.0
// 상호 점수 = (fit(A→B) + fit(B→A)) / 2        ← 양쪽 취향을 모두 반영
// 취미 점수 = 자카드 유사도 |공통 취미| / |합집합|
// 최종 점수 = 0.8 * 상호 점수 + 0.2 * 취미 점수   (0~100점으로 표시)
// 성별·나이 조건이 서로 맞지 않으면 후보에서 제외
import { MATCH_WEIGHTS, PREF_RANK_WEIGHTS } from "./config.js";

export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const argmax = v => v.reduce((bi, x, i) => (x > v[bi] ? i : bi), 0);

/** 이상형 순위(클래스 인덱스 배열, 최대 3개) -> 확률 벡터 */
export function prefVector(rankedIdx, numClasses = 11) {
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

export function hobbyScore(a = [], b = []) {
  const A = new Set(a), B = new Set(b);
  const common = [...A].filter(x => B.has(x));
  const union = new Set([...A, ...B]).size;
  return { score: union ? common.length / union : 0, common };
}

const wants = (p, q) => (p.seek === "any" || p.seek === q.gender) && q.age >= p.ageMin && q.age <= p.ageMax;
export const eligible = (a, b) => a.id !== b.id && wants(a, b) && wants(b, a);

export function matchScore(a, b) {
  const fitAB = fit(a.pref, b.face);
  const fitBA = fit(b.pref, a.face);
  const mutual = (fitAB + fitBA) / 2;
  const h = hobbyScore(a.hobbies, b.hobbies);
  const total = MATCH_WEIGHTS.mutual * mutual + MATCH_WEIGHTS.hobby * h.score;
  return { total, fitAB, fitBA, mutual, hobby: h.score, commonHobbies: h.common };
}

export function topN(me, candidates, n = 5) {
  return candidates
    .filter(c => eligible(me, c))
    .map(c => ({ candidate: c, ...matchScore(me, c) }))
    .sort((x, y) => y.total - x.total)
    .slice(0, n);
}
