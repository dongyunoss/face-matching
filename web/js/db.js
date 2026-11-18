// Supabase 연동 (2단계). config.js 에 URL/KEY 가 없으면 비활성화되고 데모 모드만 동작한다.
// 사진은 저장하지 않는다. 얼굴상 확률 벡터와 프로필 정보만 저장한다.
import { SUPABASE_ANON_KEY, SUPABASE_URL } from "./config.js";

export const dbEnabled = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
let sb = null;

async function client() {
  if (!dbEnabled) throw new Error("Supabase 설정이 없습니다 (web/js/config.js).");
  if (!sb) {
    const { createClient } = await import("https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.45.4/+esm");
    sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  }
  return sb;
}

export async function currentUser() {
  if (!dbEnabled) return null;
  const { data } = await (await client()).auth.getUser();
  return data.user ?? null;
}

export async function onAuthChange(cb) {
  if (!dbEnabled) return;
  (await client()).auth.onAuthStateChange((_e, session) => cb(session?.user ?? null));
}

export async function signUp(email, password) {
  const { data, error } = await (await client()).auth.signUp({ email, password });
  if (error) throw error;
  return data;
}

export async function signIn(email, password) {
  const { data, error } = await (await client()).auth.signInWithPassword({ email, password });
  if (error) throw error;
  return data;
}

export async function signOut() {
  await (await client()).auth.signOut();
}

const toRow = (p, uid) => ({
  id: uid, nickname: p.nickname, gender: p.gender, age: p.age, seek: p.seek,
  age_min: p.ageMin, age_max: p.ageMax, hobbies: p.hobbies,
  face_vec: p.face, pref_vec: p.pref, face_type: p.faceType,
  mbti: p.mbti || null, pref_mbti: p.prefMbti ?? [], bio: p.bio ?? "", region: p.region || null,
  importance: p.importance ?? null, updated_at: new Date().toISOString(),
});
const fromRow = r => ({
  id: r.id, nickname: r.nickname, gender: r.gender, age: r.age, seek: r.seek,
  ageMin: r.age_min, ageMax: r.age_max, hobbies: r.hobbies ?? [],
  face: r.face_vec, pref: r.pref_vec, faceType: r.face_type,
  mbti: r.mbti, prefMbti: r.pref_mbti ?? [], bio: r.bio ?? "", region: r.region,
  importance: r.importance ?? undefined,
});

export async function saveProfile(profile) {
  const c = await client();
  const user = await currentUser();
  if (!user) throw new Error("로그인이 필요합니다.");
  const { error } = await c.from("profiles").upsert(toRow(profile, user.id));
  if (error) throw error;
  return user.id;
}

export async function loadMyProfile() {
  const user = await currentUser();
  if (!user) return null;
  const { data, error } = await (await client()).from("profiles").select("*").eq("id", user.id).maybeSingle();
  if (error) throw error;
  return data ? fromRow(data) : null;
}

export async function fetchOtherProfiles() {
  const user = await currentUser();
  const { data, error } = await (await client()).from("profiles").select("*").neq("id", user?.id ?? "");
  if (error) throw error;
  return data.map(fromRow);
}

// ───────── 좋아요 · 매칭 ─────────
// likes(from_id, to_id) 한 줄 = from 이 to 에게 좋아요. 양방향 두 줄이 모두 있으면 매칭.

/** 좋아요 보내기. 상대도 이미 나를 좋아요 했으면 true (= 매칭 성사) */
export async function sendLike(toId) {
  const c = await client();
  const user = await currentUser();
  if (!user) throw new Error("로그인이 필요합니다.");
  const { error } = await c.from("likes").upsert({ from_id: user.id, to_id: toId }, { onConflict: "from_id,to_id", ignoreDuplicates: true });
  if (error) throw error;
  const { data, error: e2 } = await c.from("likes").select("from_id").eq("from_id", toId).eq("to_id", user.id).maybeSingle();
  if (e2) throw e2;
  return Boolean(data);
}

/** 내가 보낸 좋아요(liked)와 나에게 온 좋아요(likedMe) */
export async function fetchLikeState() {
  const c = await client();
  const user = await currentUser();
  if (!user) return { liked: new Set(), likedMe: new Set() };
  const { data, error } = await c.from("likes").select("from_id,to_id").or(`from_id.eq.${user.id},to_id.eq.${user.id}`);
  if (error) throw error;
  return {
    liked: new Set(data.filter(r => r.from_id === user.id).map(r => r.to_id)),
    likedMe: new Set(data.filter(r => r.to_id === user.id).map(r => r.from_id)),
  };
}
