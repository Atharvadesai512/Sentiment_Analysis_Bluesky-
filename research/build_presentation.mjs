import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const runtime = process.env.ARTIFACT_RUNTIME;
const skill = "C:/Users/Atharva/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations";
const { Presentation, PresentationFile } = await import(pathToFileURL(
  path.join(runtime, "node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs")).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(
  path.join(skill, "container_tools/artifact_tool_utils.mjs")).href);
const workspaceDir = process.cwd();
const root = path.join(workspaceDir, "research");
const build = path.join(root, "presentation_build");
await fs.mkdir(build, { recursive: true });
await fs.mkdir(path.join(root, "submission"), { recursive: true });
const replay = JSON.parse(await fs.readFile(path.join(root, "replay/summary.json"), "utf8"));
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const ink = "#192638";
const accent = "#245C92";

function text(slide, value, x, y, w, h, size = 28, bold = false, color = ink) {
  const shape = slide.shapes.add({
    geometry: "textbox", position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = { fontFamily: "Arial", fontSize: size, bold, color };
  return shape;
}

function slide(title, note) {
  const s = deck.slides.add();
  s.background.fill = "#FFFFFF";
  text(s, title, 64, 42, 1152, 100, 42, true);
  text(s, String(deck.slides.items.length), 1180, 672, 40, 24, 16, false, "#666666");
  s.speakerNotes.text = note;
  return s;
}

function points(s, items) {
  items.forEach((item, i) => {
    text(s, String(i + 1).padStart(2, "0"), 64, 177 + i * 130, 70, 50, 30, true, accent);
    text(s, item, 150, 172 + i * 130, 1030, 105, 30);
  });
}

let s = slide("Sentiment classification under processing constraints",
  "Introduce the project as an empirical comparison. Add your name and guide details before presenting. The study uses a Twitter benchmark and a separate Bluesky search sample.");
text(s, "Benchmark accuracy, temporal interpretation and local queueing", 64, 225, 1080, 140, 36);
text(s, "M.Sc. Data Science and AI", 64, 460, 1080, 50, 28, true, accent);
text(s, "TweetEval and Bluesky experiments", 64, 530, 1080, 45, 24);

s = slide("Three questions guide the evaluation",
  "Explain which evidence answers each question. Accuracy is measured only on the labeled Twitter test set.");
points(s, [
  "How do the models compare on identical labeled posts?",
  "How does model choice change the historical sentiment trend?",
  "What happens to latency and completed posts as arrivals increase?",
]);

s = slide("Two datasets serve different evaluation roles",
  "Source: project CSVs and research/experiment_manifest.json. The search corpus covers 19 October through 1 November 2026 UTC. No manual annotation was conducted.");
text(s, "12,284", 64, 175, 500, 90, 70, true, accent);
text(s, "Labeled Twitter test examples", 64, 280, 500, 55, 30);
text(s, "45,615 training rows\n2,000 validation rows", 64, 375, 500, 120, 28);
text(s, "127,741", 690, 175, 510, 90, 70, true, accent);
text(s, "Cleaned Bluesky search posts", 690, 280, 510, 55, 30);
text(s, "14 days of AI search results\n194 requests reached the result cap", 690, 375, 510, 130, 28);

s = slide("The comparison includes a simple baseline",
  "Source: sentiment/benchmark.py and sentiment/rain_bilstm.py. VADER and RoBERTa are off-the-shelf. Training histories differ. BiLSTM scores use saved predictions because TensorFlow import is blocked.");
points(s, [
  "VADER supplies the rule-based baseline.",
  "TF-IDF with logistic regression tests a low-cost learned model.",
  "BiLSTM and Twitter RoBERTa supply the neural comparisons.",
]);

s = slide("The implementation keeps evidence traceable",
  "Source: db.py and sentiment/replay_benchmark.py. The live collector includes filtering. Controlled replay begins from stored cleaned text and excludes live transport.");
const stages = ["Collect and filter", "Classify posts", "Store in SQLite", "Aggregate over time"];
stages.forEach((value, i) => {
  const box = s.shapes.add({ geometry: "rect",
    position: { left: 64 + i * 300, top: 220, width: 270, height: 110 },
    fill: "#EAF0F6", line: { fill: "#C8D4E0", width: 1 } });
  box.text = value;
  box.text.style = { fontFamily: "Arial", fontSize: 26, color: ink, bold: true };
});
text(s, "Saved predictions, source hashes and timing records support reproduction.", 64, 410, 1135, 125, 32);

s = slide("RoBERTa leads the labeled benchmark",
  "Source: research/benchmark/*_metrics.json. All models use the same 12,284 local test rows. RoBERTa accuracy is 72.18% and macro F1 is 0.724. BiLSTM uses saved predictions.");
const chart = s.charts.add("bar", {
  position: { left: 70, top: 170, width: 1120, height: 425 },
  categories: ["VADER", "TF-IDF + LR", "BiLSTM saved", "RoBERTa"],
  series: [
    { name: "Accuracy", values: [.530039, .592234, .594676, .721833], fill: accent },
    { name: "Macro F1", values: [.528821, .560192, .558241, .724050], fill: "#999999" },
  ], hasLegend: true,
  yAxis: { minimumScale: 0, maximumScale: 1 },
});
applyPresentationChartFont(chart, { fontFamily: "Arial" });
text(s, "Score on a 0–1 scale. Neutral majority accuracy is 0.483.", 70, 610, 1100, 35, 20);

s = slide("Small score differences need uncertainty",
  "Source: research/benchmark/uncertainty.json. Paired bootstrap uses 1,000 resamples, seed 42, and assumes independent rows.");
points(s, [
  "RoBERTa accuracy interval is 71.38% to 72.95%.",
  "The TF-IDF versus BiLSTM accuracy-difference interval includes zero.",
  "BiLSTM misses many negative examples by assigning neutral.",
]);

s = slide("Classifier choice changes the temporal signal",
  "Source: research/temporal/summary.json and sentiment_h.csv. The plotted lines are six-hour trailing means. Opposite-sign counts use raw hourly net sentiment. VADER and RoBERTa differ in sign in 327 of 336 hours.");
const trendBytes = await fs.readFile(path.join(root, "figures/sentiment_trends.png"));
s.images.add({ blob: trendBytes.buffer.slice(trendBytes.byteOffset, trendBytes.byteOffset + trendBytes.byteLength), contentType: "image/png", fit: "contain",
  position: { left: 80, top: 145, width: 1110, height: 500 }, alt: "Historical sentiment trends and collection cap hits" });

s = slide("Inference cost depends on batching and workload",
  "Source: research/runtime/summary.csv. Fixed sample of 96 test posts, three repeats, CPU, four PyTorch threads. Median of run-specific metrics. Excludes queueing and storage.");
text(s, "26.45 ms", 64, 190, 540, 90, 66, true, accent);
text(s, "RoBERTa median run p50\nfor a single-post call", 64, 310, 540, 100, 30);
text(s, "59.60 posts/s", 690, 190, 525, 90, 58, true, accent);
text(s, "RoBERTa median throughput\nwith batches of eight", 690, 310, 525, 100, 30);
text(s, "These measurements describe warm local inference on the fixed test sample.", 64, 535, 1110, 80, 26);

s = slide("Controlled replay exposes queue overflow",
  "Source: research/replay/summary.json. Each condition offers 288 posts across three short runs. One worker, batch one, waiting capacity 16, actual SQLite commits. Network and filtering are excluded.");
const values = [["Model", "Offered posts/s", "Stored of 288", "Dropped of 288"]];
for (const model of ["VADER", "RoBERTa"]) {
  for (const rate of [10, 40, 80]) {
    const group = replay.filter(r => r.model === model && r.offered_rate === rate);
    values.push([model, String(rate), String(group.reduce((v, r) => v + r.stored, 0)),
      String(group.reduce((v, r) => v + r.dropped, 0))]);
  }
}
const table = s.tables.add({ rows: values.length, columns: 4, values,
  left: 70, top: 165, width: 1120, height: 420 });
for (let r = 0; r < values.length; r++) {
  for (let c = 0; c < 4; c++) {
    const cell = table.getCell(r, c);
    cell.text.style = { fontFamily: "Arial", fontSize: 24, bold: r === 0, color: ink };
    cell.fill = r === 0 ? "#EAF0F6" : "#FFFFFF";
  }
}
text(s, "Latency of completed posts must be interpreted alongside dropped arrivals.", 70, 612, 1120, 45, 22);

s = slide("The findings have explicit boundaries",
  "Explain these limitations directly in viva. The report does not claim Bluesky accuracy or production capacity. TensorFlow import prevents fresh BiLSTM timing.");
points(s, [
  "Bluesky labels are unverified, and search coverage is incomplete.",
  "Pretraining histories differ, and bootstrap assumes independent rows.",
  "Short local replay omits live delivery and long-running system effects.",
]);

s = slide("The study connects model quality with monitoring behaviour",
  "Sources: dissertation Chapters 5 and 6. Related work includes TweetEval, TimeLMs, CognitiveSky and BlueTempNet. Cite full references in the dissertation. Publication requires venue-specific review.");
points(s, [
  "RoBERTa has the highest observed labeled benchmark scores.",
  "The same historical posts produce different sentiment directions.",
  "Processing constraints affect which arriving posts reach storage.",
]);

const candidate = path.join(build, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidate);
const finalPath = path.join(root, "submission/Project Presentation.pptx");
await finalizePresentation({
  workspaceDir, candidatePath: candidate, finalPath,
  pythonExecutable: path.join(runtime, "python/python.exe"),
  integrityValidatorPath: path.join(skill, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skill, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  requiredNativeTableOwnerSlides: [],
  fontPolicy: { basis: "design", families: ["Arial"] },
  verifyArtifactToolImport: true,
  materializeLiteralChartWorkbooks: true,
  receiptPath: path.join(build, "validation.json"),
});
for (const [index, s] of deck.slides.items.entries()) {
  try {
    const image = await deck.export({ slide: s, format: "png", scale: 1 });
    await fs.writeFile(path.join(build, `slide-${index + 1}.png`), new Uint8Array(await image.arrayBuffer()));
  } catch (error) {
    console.warn("Preview renderer unavailable:", error.message);
    break;
  }
}
console.log(finalPath);
