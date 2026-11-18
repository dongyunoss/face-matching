-- Supabase 대시보드 → SQL Editor 에 붙여넣고 Run.
-- 사진은 저장하지 않고 얼굴상 확률 벡터(face_vec)와 이상형 벡터(pref_vec)만 저장한다.

create table if not exists public.profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  nickname    text not null check (char_length(nickname) between 1 and 20),
  gender      text not null check (gender in ('M', 'F')),
  age         int  not null check (age between 19 and 99),
  seek        text not null check (seek in ('M', 'F', 'any')),
  age_min     int  not null check (age_min >= 19),
  age_max     int  not null check (age_max >= age_min),
  hobbies     text[] not null default '{}',
  face_vec    float8[] not null check (array_length(face_vec, 1) = 10),
  pref_vec    float8[] not null check (array_length(pref_vec, 1) = 10),
  face_type   text not null,
  mbti        text check (mbti ~ '^[EI][SN][TF][JP]$'),
  pref_mbti   text[] not null default '{}',
  bio         text not null default '' check (char_length(bio) <= 300),
  region      text,
  importance  jsonb,       -- 항목별 중요도 {"face":3,"mbti":1,"interest":3,"distance":1}
  updated_at  timestamptz not null default now()
);

-- 이미 테이블을 만든 경우: 새 컬럼(MBTI·자기소개·지역)만 추가
alter table public.profiles add column if not exists mbti text check (mbti ~ '^[EI][SN][TF][JP]$');
alter table public.profiles add column if not exists pref_mbti text[] not null default '{}';
alter table public.profiles add column if not exists bio text not null default '' check (char_length(bio) <= 300);
alter table public.profiles add column if not exists region text;
alter table public.profiles add column if not exists importance jsonb;

alter table public.profiles enable row level security;

-- 로그인한 사용자는 다른 회원 프로필을 읽을 수 있음 (매칭 계산용)
drop policy if exists "read profiles" on public.profiles;
create policy "read profiles" on public.profiles
  for select to authenticated using (true);

-- 자기 프로필만 생성/수정/삭제 가능
drop policy if exists "insert own" on public.profiles;
create policy "insert own" on public.profiles
  for insert to authenticated with check ((select auth.uid()) = id);

drop policy if exists "update own" on public.profiles;
create policy "update own" on public.profiles
  for update to authenticated using ((select auth.uid()) = id) with check ((select auth.uid()) = id);

drop policy if exists "delete own" on public.profiles;
create policy "delete own" on public.profiles
  for delete to authenticated using ((select auth.uid()) = id);

-- ───────── 좋아요 · 매칭 ─────────
-- 한 줄 = from_id 가 to_id 에게 좋아요. 양방향 두 줄이 모두 있으면 매칭.
create table if not exists public.likes (
  from_id    uuid not null references public.profiles(id) on delete cascade,
  to_id      uuid not null references public.profiles(id) on delete cascade,
  created_at timestamptz not null default now(),
  primary key (from_id, to_id),
  check (from_id <> to_id)
);

alter table public.likes enable row level security;

-- 내가 보냈거나 받은 좋아요만 볼 수 있음 (다른 사람끼리의 좋아요는 안 보임)
drop policy if exists "read own likes" on public.likes;
create policy "read own likes" on public.likes
  for select to authenticated using ((select auth.uid()) in (from_id, to_id));

-- 좋아요는 내 이름으로만 보낼 수 있음
drop policy if exists "send like" on public.likes;
create policy "send like" on public.likes
  for insert to authenticated with check ((select auth.uid()) = from_id);

drop policy if exists "cancel like" on public.likes;
create policy "cancel like" on public.likes
  for delete to authenticated using ((select auth.uid()) = from_id);
