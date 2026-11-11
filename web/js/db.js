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
  mbti: p.mbti || null, pref_mbti: p.prefMbti ?? [], bio: p.bio ?? "", region: p.region || null, updated_at: new Date().toISOString(),
});
const fromRow = r => ({
  id: r.id, nickname: r.nickname, gender: r.gender, age: r.age, seek: r.seek,
  ageMin: r.age_min, ageMax: r.age_max, hobbies: r.hobbies ?? [],
  face: r.face_vec, pref: r.pref_vec, faceType: r.face_type,
  mbti: r.mbti, prefMbti: r.pref_mbti ?? [], bio: r.bio ?? "", region: r.region,
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
