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
  face_vec    float8[] not null check (array_length(face_vec, 1) = 11),
  pref_vec    float8[] not null check (array_length(pref_vec, 1) = 11),
  face_type   text not null,
  updated_at  timestamptz not null default now()
);

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
