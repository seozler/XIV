-- PostgreSQL schema for analysed reference videos (idempotent).
CREATE TABLE IF NOT EXISTS ref_videos (
    video_id     text PRIMARY KEY,
    sha256       text NOT NULL,
    input_file   text,
    duration_s   double precision,
    width        integer,
    height       integer,
    fps          double precision,
    profile      jsonb,
    analysed_at  timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ref_scenes (
    video_id text REFERENCES ref_videos ON DELETE CASCADE, scene_id text, start_s double precision,
    end_s double precision, data jsonb);
CREATE TABLE IF NOT EXISTS ref_elements (
    video_id text REFERENCES ref_videos ON DELETE CASCADE, element_id text, role text,
    start_s double precision, end_s double precision, data jsonb);
CREATE TABLE IF NOT EXISTS ref_text_tracks (
    video_id text REFERENCES ref_videos ON DELETE CASCADE, track_id text, content text,
    start_s double precision, end_s double precision, data jsonb);
CREATE TABLE IF NOT EXISTS ref_audio_events (
    video_id text REFERENCES ref_videos ON DELETE CASCADE, type text, start_s double precision,
    end_s double precision, data jsonb);
CREATE TABLE IF NOT EXISTS ref_words (
    video_id text REFERENCES ref_videos ON DELETE CASCADE, word text, start_s double precision,
    end_s double precision, speaker text);
CREATE TABLE IF NOT EXISTS ref_sync_patterns (
    video_id text REFERENCES ref_videos ON DELETE CASCADE, rule text, hit_rate double precision,
    detected boolean, data jsonb);
CREATE TABLE IF NOT EXISTS ref_tasks (
    video_id text REFERENCES ref_videos ON DELETE CASCADE, task_id text, title text, subsystem text,
    priority text, data jsonb);
CREATE INDEX IF NOT EXISTS ref_elements_video ON ref_elements (video_id);
CREATE INDEX IF NOT EXISTS ref_words_video ON ref_words (video_id);
CREATE INDEX IF NOT EXISTS ref_profile_gin ON ref_videos USING gin (profile);
