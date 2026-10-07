// BullMQ bridge: lets the existing Node backend queue reference analyses.
//
//   npm i bullmq
//   REDIS_URL=redis://localhost:6379 node integrations/bullmq/worker.mjs
//
// Enqueue from the backend:
//   await new Queue("reference-analysis", { connection }).add("analyze", {
//     video: "/data/refs/clip.mp4", system: "/data/current-video-system.txt", mode: "full",
//   });
//
// The job result carries the workspace path and the improvement tasks, so the
// backend can store them (or call `reference-analyzer export --postgres`).
import { Worker } from "bullmq";
import { spawn } from "node:child_process";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

const OUT = process.env.REFANALYZER_OUT ?? "reference-analysis";
const BIN = process.env.REFANALYZER_BIN ?? "reference-analyzer";
const url = new URL(process.env.REDIS_URL ?? "redis://localhost:6379");
const connection = { host: url.hostname, port: Number(url.port || 6379), password: url.password || undefined };

function run(args, job) {
  return new Promise((resolve, reject) => {
    const p = spawn(BIN, args, { stdio: ["ignore", "pipe", "pipe"] });
    let log = "";
    const onData = (d) => {
      log += d;
      const m = /\[(\w+)\] (start|done|cached)/.exec(String(d));
      if (m) job.updateProgress({ stage: m[1], state: m[2] });
    };
    p.stdout.on("data", onData);
    p.stderr.on("data", onData);
    p.on("close", (code) => (code === 0 ? resolve(log) : reject(new Error(`exit ${code}\n${log.slice(-4000)}`))));
  });
}

new Worker(
  "reference-analysis",
  async (job) => {
    const { video, system, mode = "full", offline = false } = job.data;
    const args = ["--out", OUT, "analyze", "--video", video, "--mode", mode];
    if (system) args.push("--system", system);
    if (offline) args.push("--offline");
    const log = await run(args, job);
    const ws = /Workspace: (.+)/.exec(log)?.[1]?.trim();
    const tasks = ws ? JSON.parse(await readFile(path.join(ws, "reports/improvement-tasks.json"), "utf8")) : null;
    return { workspace: ws, reports: ws ? await readdir(path.join(ws, "reports")) : [], tasks };
  },
  { connection, concurrency: Number(process.env.CONCURRENCY ?? 1) },
);
