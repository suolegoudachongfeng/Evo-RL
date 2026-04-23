const chartCanvas = document.getElementById("signalChart");
const metricButtons = document.querySelectorAll(".metric-btn");
const langButtons = document.querySelectorAll(".lang-btn");
const themeToggle = document.getElementById("themeToggle");
const themeLabel = document.getElementById("themeLabel");
const themeIcon = document.getElementById("themeIcon");

const STORAGE_LANG_KEY = "evorl_lang";
const STORAGE_THEME_KEY = "evorl_theme";

const validLangs = ["en", "zh"];
const validThemes = ["dark", "light"];

const i18n = {
  en: {
    pageTitle: "Evo-RL | Continuous Open-Source Real-World RL",
    pageDescription: "Evo-RL: continuous open-source real-world reinforcement learning on SO101 and AgileX PiPER, with a full reproducible CLI pipeline and offline value/policy training.",
    navFocus: "Focus",
    navPipeline: "Pipeline",
    navResults: "Results",
    navMedia: "Media",
    navReproduce: "Reproduce",
    trafficTitle: "Site Traffic",
    trafficVisitors: "Visitors",
    trafficViews: "Views",
    heroEyebrow: "Continuous Open-Source Real-World RL",
    heroSubline: "SO101 to AgileX PiPER",
    heroSummary:
      "A LeRobot-aligned, full-pipeline real-world offline RL workflow for real robots: collect rollout data, train value functions and policies offline, infer advantage tags, and iterate deployment in a closed loop.",
    newsLabel: "NEWS",
    newsDate: "February 26, 2026",
    newsText: "First SO101 real-world RL baseline + reproducible CLI released.",
    btnRepo: "GitHub Repo",
    btnDocs: "LeRobot Docs",
    statPlatforms: "Robot Platforms",
    statOpenTasks: "Open Task Datasets",
    statOpenModels: "Open Models",
    statReleaseYear: "Release Year",
    focusEyebrow: "Project Focus",
    focusTitle: "Open Infrastructure for Real-World RL",
    focus1Title: "Open real-world RL pipeline",
    focus1Body:
      "Runnable workflows on SO101 and AgileX PiPER covering collection, offline value/policy training, and iterative deployment.",
    focus2Title: "Reproducible assets",
    focus2Body:
      "Codebase, model layout, dataset structure, and command-line templates are released to reduce setup friction and improve reproducibility.",
    focus3Title: "Community co-evolution",
    focus3Body:
      "Reproduce existing real-world RL methods, extend value/policy variants, and continuously publish experiments and benchmarks for collective progress.",
    methodEyebrow: "Method",
    methodTitle: "6-Stage Closed-Loop Pipeline",
    step1: "<strong>System preparation</strong> to ensure robot, sensors, and control stack are ready.",
    step2: "<strong>Interactive data collection</strong> with human guidance and episode-level outcomes.",
    step3: "<strong>Value modeling</strong> to estimate trajectory quality from accumulated data.",
    step4: "<strong>Signal annotation</strong> to convert value outputs into usable training supervision.",
    step5: "<strong>Policy update and deployment</strong> to run the improved policy in real tasks.",
    step6: "<strong>Data refresh and iteration</strong> to continue improving robustness and success rate.",
    resultsEyebrow: "Results",
    resultsTitle: "Training Signals Across Iterations",
    metricValue: "Value",
    metricAdvantage: "Advantage",
    metricSuccess: "Success Rate",
    chartNote: "Illustrative trend view for project webpage. Replace with your run logs for paper-ready plots.",
    insightTitle: "What This Visual Captures",
    insight1: "Value trend stabilizes as rollouts and labels improve across rounds.",
    insight2: "Advantage separates better trajectories used for ACP tagging.",
    insight3: "Success rate rises with iterative deployment and data refresh.",
    noteStrong:
      "Current default ACP ratio in README pipeline: <code>--acp.positive_ratio=0.3</code>.",
    mediaEyebrow: "Qualitative Demos",
    mediaTitle: "Visual Results Gallery",
    mediaTag1: "VALUE VISUAL",
    mediaTitle1: "Success and Failure Case",
    mediaBody1: "Frame-level value visualization on real robot rollouts.",
    mediaTag2: "POLICY ROLLOUT",
    mediaTitle2: "Deployment Visual Results",
    mediaBody2: "Final policy behavior clips in target task execution.",
    mediaTag3: "HUMAN-IN-THE-LOOP",
    mediaTitle3: "Intervention and Recovery Results",
    mediaBody3: "Human guidance segments and autonomous recovery outcomes.",
    architectureEyebrow: "Architecture",
    architectureTitle: "LeRobot-Aligned VLA + Value Loop",
    architectureBody1:
      "Evo-RL reuses LeRobot's practical data and inference stack while adding value-guided filtering signals (<code>value</code>, <code>advantage</code>, <code>acp_indicator</code>) to improve iterative policy training.",
    architectureBody2:
      "The same design supports single-GPU and accelerate-based multi-GPU workflows for both value and policy stages.",
    relatedEyebrow: "Related Work",
    relatedTitle: "Context in Open Robot Learning",
    reproduceEyebrow: "Reproducibility",
    reproduceTitle: "Core CLI Commands",
    humanInloopRecord: "Human-in-loop Record",
    valueTrain: "Value Training",
    policyTrain: "Policy Training",
    communityEyebrow: "Model & Community",
    communityTitle: "Links and Contact",
    releaseStatus: "Release Status",
    release1: "Hugging Face model: coming soon",
    release2: "Hugging Face dataset: coming soon",
    release3: "Canonical version tags will be pinned once published",
    wechatTitle: "WeChat Group",
    wechatDesc: "Scan to join EvoMind's WeChat community and follow project updates.",
    wechatContact: "Contact",
    citationEyebrow: "Citation",
    copyBib: "Copy BibTeX",
    copied: "Copied",
    copyFailed: "Copy Failed",
    themeDark: "Night",
    themeLight: "Day",
    themeToggleAria: "Theme toggle",
    chartLatest: "Latest"
  },
  zh: {
    pageTitle: "Evo-RL | 持续开源真实世界强化学习",
    pageDescription: "Evo-RL：面向 SO101 与 AgileX PiPER 的真实世界强化学习开源项目，提供可复现的全流程命令行管线，并包含离线 value/policy 训练。",
    navFocus: "项目目标",
    navPipeline: "方法流程",
    navResults: "实验结果",
    navMedia: "视频展示",
    navReproduce: "复现命令",
    trafficTitle: "访问统计",
    trafficVisitors: "访客人数",
    trafficViews: "查看次数",
    heroEyebrow: "持续开源的真实世界强化学习",
    heroSubline: "从 SO101 到 AgileX PiPER",
    heroSummary:
      "基于 LeRobot 实践体系的真实机器人全流程离线 RL：采集 rollout 数据、离线训练 value/policy、推理 advantage 标签，并在闭环中持续迭代部署。",
    newsLabel: "最新进展",
    newsDate: "2026年2月26日",
    newsText: "首个 SO101 真实世界 RL baseline 与可复现 CLI 工作流已发布。",
    btnRepo: "GitHub 仓库",
    btnDocs: "LeRobot 文档",
    statPlatforms: "机器人平台",
    statOpenTasks: "社区开源任务数据",
    statOpenModels: "社区开源模型",
    statReleaseYear: "发布日期",
    focusEyebrow: "项目目标",
    focusTitle: "真实世界 RL 的开源基础设施",
    focus1Title: "开放真实世界 RL 全流程",
    focus1Body: "在 SO101 与 AgileX PiPER 上提供可运行流程，覆盖采集、离线 value/policy 训练与迭代部署。",
    focus2Title: "可复现资产",
    focus2Body: "持续发布代码、模型组织方式、数据结构与命令模板，降低复现实验门槛。",
    focus3Title: "社区共演化",
    focus3Body: "复现已有真实世界 RL 方法，扩展 value/policy 变体，并持续发布实验与基准推动社区进步。",
    methodEyebrow: "方法",
    methodTitle: "6阶段闭环流程",
    step1: "<strong>系统准备</strong>：确保机器人、传感器与控制链路可稳定运行。",
    step2: "<strong>交互式数据采集</strong>：通过人类引导采集轨迹并记录回合结果。",
    step3: "<strong>价值建模</strong>：基于当前数据学习轨迹质量评估能力。",
    step4: "<strong>信号标注</strong>：将价值输出转化为可用于训练的监督信号。",
    step5: "<strong>策略更新与部署</strong>：训练改进后的策略并投入真实任务执行。",
    step6: "<strong>数据刷新与迭代</strong>：持续循环优化鲁棒性与任务成功率。",
    resultsEyebrow: "结果",
    resultsTitle: "跨轮次训练信号",
    metricValue: "Value",
    metricAdvantage: "Advantage",
    metricSuccess: "成功率",
    chartNote: "当前为示意曲线。替换为你的真实训练日志后可直接用于论文或展示。",
    insightTitle: "这张图反映了什么",
    insight1: "随着轮次增加和标注改进，Value 曲线逐步稳定。",
    insight2: "Advantage 更好地区分高质量轨迹，用于 ACP 标签化。",
    insight3: "在闭环部署和数据刷新下，任务成功率持续提升。",
    noteStrong: "README 默认 ACP 比例：<code>--acp.positive_ratio=0.3</code>。",
    mediaEyebrow: "定性展示",
    mediaTitle: "可视化结果画廊",
    mediaTag1: "VALUE 可视化",
    mediaTitle1: "成功与失败案例",
    mediaBody1: "真实机器人 rollout 的逐帧 value 可视化。",
    mediaTag2: "策略 Rollout",
    mediaTitle2: "策略部署可视化结果",
    mediaBody2: "最终策略在目标任务中的执行片段。",
    mediaTag3: "人类在环",
    mediaTitle3: "干预与恢复结果",
    mediaBody3: "展示人工引导片段与自主恢复结果。",
    architectureEyebrow: "架构",
    architectureTitle: "LeRobot 对齐的 VLA + Value 闭环",
    architectureBody1:
      "Evo-RL 复用了 LeRobot 的实用数据与推理栈，并加入 value 引导过滤信号（<code>value</code>、<code>advantage</code>、<code>acp_indicator</code>）以提升迭代训练效果。",
    architectureBody2: "同一套设计同时支持单卡与 accelerate 多卡的 value/policy 训练流程。",
    relatedEyebrow: "相关工作",
    relatedTitle: "开源机器人学习语境",
    reproduceEyebrow: "复现",
    reproduceTitle: "核心 CLI 命令",
    humanInloopRecord: "Human-in-loop 采集",
    valueTrain: "Value 训练",
    policyTrain: "Policy 训练",
    communityEyebrow: "模型与社区",
    communityTitle: "链接与联系",
    releaseStatus: "发布状态",
    release1: "Hugging Face 模型：即将发布",
    release2: "Hugging Face 数据集：即将发布",
    release3: "发布后会在此固定 canonical 版本标签",
    wechatTitle: "微信交流群",
    wechatDesc: "扫码加入 EvoMind 微信社区，获取 Evo-RL 的最新项目动态。",
    wechatContact: "联系邮箱",
    citationEyebrow: "引用",
    copyBib: "复制 BibTeX",
    copied: "已复制",
    copyFailed: "复制失败",
    themeDark: "夜间",
    themeLight: "白天",
    themeToggleAria: "切换白天夜间模式",
    chartLatest: "最新"
  }
};

const metrics = {
  value: {
    labels: { en: "Mean Value", zh: "平均 Value" },
    unit: "",
    color: "#2af2d0",
    fill: "rgba(42, 242, 208, 0.16)",
    points: [0.41, 0.53, 0.69, 0.78, 0.92, 1.01, 1.09, 1.15]
  },
  advantage: {
    labels: { en: "Mean Advantage", zh: "平均 Advantage" },
    unit: "",
    color: "#f8b84e",
    fill: "rgba(248, 184, 78, 0.14)",
    points: [0.07, 0.11, 0.14, 0.21, 0.24, 0.28, 0.3, 0.33]
  },
  success: {
    labels: { en: "Task Success Rate", zh: "任务成功率" },
    unit: "%",
    color: "#7db6ff",
    fill: "rgba(125, 182, 255, 0.15)",
    points: [22, 29, 37, 46, 54, 61, 67, 73]
  }
};

const rounds = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"];
let activeMetric = "value";
let currentLang = validLangs.includes(localStorage.getItem(STORAGE_LANG_KEY))
  ? localStorage.getItem(STORAGE_LANG_KEY)
  : "en";
let currentTheme = validThemes.includes(localStorage.getItem(STORAGE_THEME_KEY))
  ? localStorage.getItem(STORAGE_THEME_KEY)
  : "light";

const textBindings = [
  { selector: "nav.site-nav a[href='#focus']", key: "navFocus" },
  { selector: "nav.site-nav a[href='#pipeline']", key: "navPipeline" },
  { selector: "nav.site-nav a[href='#results']", key: "navResults" },
  { selector: "nav.site-nav a[href='#media']", key: "navMedia" },
  { selector: "nav.site-nav a[href='#reproduce']", key: "navReproduce" },
  { selector: "#traffic .traffic-title", key: "trafficTitle" },
  { selector: "#traffic .traffic-item:nth-child(2) .traffic-label", key: "trafficVisitors" },
  { selector: "#traffic .traffic-item:nth-child(3) .traffic-label", key: "trafficViews" },
  { selector: ".hero .eyebrow", key: "heroEyebrow" },
  { selector: ".hero-left h1 span", key: "heroSubline" },
  { selector: ".hero-summary", key: "heroSummary" },
  { selector: ".news-pill span", key: "newsLabel" },
  { selector: ".news-pill strong", key: "newsDate" },
  { selector: ".news-pill em", key: "newsText" },
  { selector: ".hero-actions .btn-primary", key: "btnRepo" },
  { selector: ".hero-actions .btn-ghost", key: "btnDocs" },
  { selector: ".hero-stats .stat-card:nth-child(1) .stat-label", key: "statPlatforms" },
  { selector: ".hero-stats .stat-card:nth-child(2) .stat-label", key: "statOpenTasks" },
  { selector: ".hero-stats .stat-card:nth-child(3) .stat-label", key: "statOpenModels" },
  { selector: ".hero-stats .stat-card:nth-child(4) .stat-label", key: "statReleaseYear" },
  { selector: "#focus .section-heading .eyebrow", key: "focusEyebrow" },
  { selector: "#focus .section-heading h2", key: "focusTitle" },
  { selector: "#focus .card-grid .glass-card:nth-child(1) h3", key: "focus1Title" },
  { selector: "#focus .card-grid .glass-card:nth-child(1) p", key: "focus1Body" },
  { selector: "#focus .card-grid .glass-card:nth-child(2) h3", key: "focus2Title" },
  { selector: "#focus .card-grid .glass-card:nth-child(2) p", key: "focus2Body" },
  { selector: "#focus .card-grid .glass-card:nth-child(3) h3", key: "focus3Title" },
  { selector: "#focus .card-grid .glass-card:nth-child(3) p", key: "focus3Body" },
  { selector: "#pipeline .section-heading .eyebrow", key: "methodEyebrow" },
  { selector: "#pipeline .section-heading h2", key: "methodTitle" },
  { selector: "#pipeline .pipeline-list li:nth-child(1) p", key: "step1", html: true },
  { selector: "#pipeline .pipeline-list li:nth-child(2) p", key: "step2", html: true },
  { selector: "#pipeline .pipeline-list li:nth-child(3) p", key: "step3", html: true },
  { selector: "#pipeline .pipeline-list li:nth-child(4) p", key: "step4", html: true },
  { selector: "#pipeline .pipeline-list li:nth-child(5) p", key: "step5", html: true },
  { selector: "#pipeline .pipeline-list li:nth-child(6) p", key: "step6", html: true },
  { selector: "#results .section-heading .eyebrow", key: "resultsEyebrow" },
  { selector: "#results .section-heading h2", key: "resultsTitle" },
  { selector: ".chart-toolbar .metric-btn[data-metric='value']", key: "metricValue" },
  { selector: ".chart-toolbar .metric-btn[data-metric='advantage']", key: "metricAdvantage" },
  { selector: ".chart-toolbar .metric-btn[data-metric='success']", key: "metricSuccess" },
  { selector: ".chart-note", key: "chartNote" },
  { selector: ".insights-card h3", key: "insightTitle" },
  { selector: ".insights-card ul li:nth-child(1)", key: "insight1" },
  { selector: ".insights-card ul li:nth-child(2)", key: "insight2" },
  { selector: ".insights-card ul li:nth-child(3)", key: "insight3" },
  { selector: ".note-strong", key: "noteStrong", html: true },
  { selector: "#media .section-heading .eyebrow", key: "mediaEyebrow" },
  { selector: "#media .section-heading h2", key: "mediaTitle" },
  { selector: "#media .viz-lane:nth-child(1) .lane-kicker", key: "mediaTag1" },
  { selector: "#media .viz-lane:nth-child(1) h3", key: "mediaTitle1" },
  { selector: "#media .viz-lane:nth-child(1) .lane-note", key: "mediaBody1" },
  { selector: "#media .viz-lane:nth-child(2) .lane-kicker", key: "mediaTag2" },
  { selector: "#media .viz-lane:nth-child(2) h3", key: "mediaTitle2" },
  { selector: "#media .viz-lane:nth-child(2) .lane-note", key: "mediaBody2" },
  { selector: "#media .viz-lane:nth-child(3) .lane-kicker", key: "mediaTag3" },
  { selector: "#media .viz-lane:nth-child(3) h3", key: "mediaTitle3" },
  { selector: "#media .viz-lane:nth-child(3) .lane-note", key: "mediaBody3" },
  { selector: "#architecture .section-heading .eyebrow", key: "architectureEyebrow" },
  { selector: "#architecture .section-heading h2", key: "architectureTitle" },
  { selector: "#architecture .architecture-block div p:nth-child(1)", key: "architectureBody1", html: true },
  { selector: "#architecture .architecture-block div p:nth-child(2)", key: "architectureBody2" },
  { selector: "#related .section-heading .eyebrow", key: "relatedEyebrow" },
  { selector: "#related .section-heading h2", key: "relatedTitle" },
  { selector: "#reproduce .section-heading .eyebrow", key: "reproduceEyebrow" },
  { selector: "#reproduce .section-heading h2", key: "reproduceTitle" },
  { selector: "#reproduce .card-grid .glass-card:nth-child(1) h3", key: "humanInloopRecord" },
  { selector: "#reproduce .card-grid .glass-card:nth-child(2) h3", key: "valueTrain" },
  { selector: "#reproduce .card-grid .glass-card:nth-child(3) h3", key: "policyTrain" },
  { selector: "#community .section-heading .eyebrow", key: "communityEyebrow" },
  { selector: "#community .section-heading h2", key: "communityTitle" },
  { selector: "#community .card-grid .glass-card:nth-child(1) h3", key: "releaseStatus" },
  { selector: "#community .card-grid .glass-card:nth-child(1) ul li:nth-child(1)", key: "release1" },
  { selector: "#community .card-grid .glass-card:nth-child(1) ul li:nth-child(2)", key: "release2" },
  { selector: "#community .card-grid .glass-card:nth-child(1) ul li:nth-child(3)", key: "release3" },
  { selector: "#community .community-grid .qr-card h3", key: "wechatTitle" },
  { selector: "#community .community-grid .qr-card .qr-desc", key: "wechatDesc" },
  { selector: "#community .community-grid .qr-card .qr-contact-label", key: "wechatContact" },
  { selector: "#citation .section-heading .eyebrow", key: "citationEyebrow" }
];

function t(key) {
  return i18n[currentLang][key] || i18n.en[key] || key;
}

function setTextContent(selector, value, useHtml = false) {
  const el = document.querySelector(selector);
  if (!el) return;
  if (useHtml) {
    el.innerHTML = value;
    return;
  }
  el.textContent = value;
}

function applyTranslations() {
  document.title = t("pageTitle");
  const description = document.querySelector("meta[name='description']");
  if (description) {
    description.setAttribute("content", t("pageDescription"));
  }

  textBindings.forEach((binding) => {
    setTextContent(binding.selector, t(binding.key), Boolean(binding.html));
  });

  if (themeToggle) {
    themeToggle.setAttribute("aria-label", t("themeToggleAria"));
  }

  const copyButton = document.getElementById("copyBib");
  if (copyButton) {
    copyButton.textContent = t("copyBib");
  }
}

function applyLanguage(lang) {
  if (!validLangs.includes(lang)) return;
  currentLang = lang;
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  localStorage.setItem(STORAGE_LANG_KEY, lang);

  langButtons.forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.lang === lang);
  });

  applyTranslations();
  updateThemeToggle();
  drawChart(activeMetric);
}

function updateThemeToggle() {
  if (!themeLabel || !themeIcon) return;
  const isDark = currentTheme === "dark";
  themeIcon.textContent = isDark ? "◐" : "◑";
  themeLabel.textContent = isDark ? t("themeDark") : t("themeLight");
}

function applyTheme(theme) {
  if (!validThemes.includes(theme)) return;
  currentTheme = theme;
  document.body.dataset.theme = theme;
  localStorage.setItem(STORAGE_THEME_KEY, theme);
  updateThemeToggle();
  drawChart(activeMetric);
}

function drawChart(metricKey) {
  if (!chartCanvas) return;
  const metric = metrics[metricKey];
  const ctx = chartCanvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  const cssWidth = chartCanvas.clientWidth || 960;
  const cssHeight = Math.max(320, Math.round(cssWidth * 0.5));
  chartCanvas.width = Math.round(cssWidth * dpr);
  chartCanvas.height = Math.round(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const style = getComputedStyle(document.body);
  const chartBg = style.getPropertyValue("--chart-bg").trim() || "rgba(4, 10, 15, 0.95)";
  const chartGrid = style.getPropertyValue("--chart-grid").trim() || "rgba(112, 153, 186, 0.25)";
  const chartAxis = style.getPropertyValue("--chart-axis").trim() || "#93aec0";
  const chartLatest = style.getPropertyValue("--chart-latest").trim() || "#d1e3ef";

  const w = cssWidth;
  const h = cssHeight;
  const pad = { top: 28, right: 24, bottom: 42, left: 52 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = chartBg;
  ctx.fillRect(pad.left, pad.top, plotW, plotH);

  const values = metric.points;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const xStep = plotW / (values.length - 1);
  const toX = (i) => pad.left + i * xStep;
  const toY = (val) => pad.top + plotH - ((val - min) / range) * plotH;

  ctx.strokeStyle = chartGrid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(toX(0), toY(values[0]));
  values.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
  ctx.lineTo(toX(values.length - 1), pad.top + plotH);
  ctx.lineTo(toX(0), pad.top + plotH);
  ctx.closePath();
  ctx.fillStyle = metric.fill;
  ctx.fill();

  ctx.beginPath();
  values.forEach((v, i) => {
    const x = toX(i);
    const y = toY(v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = metric.color;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  values.forEach((v, i) => {
    const x = toX(i);
    const y = toY(v);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = metric.color;
    ctx.fill();
  });

  ctx.fillStyle = chartAxis;
  ctx.font = '12px "IBM Plex Mono", monospace';
  ctx.textAlign = "center";
  rounds.forEach((r, i) => {
    ctx.fillText(r, toX(i), h - 14);
  });

  ctx.save();
  ctx.translate(16, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  const label = metric.labels[currentLang] || metric.labels.en;
  ctx.fillText(label + (metric.unit ? ` (${metric.unit})` : ""), 0, 0);
  ctx.restore();

  ctx.textAlign = "right";
  ctx.fillStyle = chartLatest;
  const end = values[values.length - 1];
  ctx.fillText(`${t("chartLatest")}: ${end}${metric.unit}`, w - 16, 18);
}

metricButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    metricButtons.forEach((item) => item.classList.remove("is-active"));
    btn.classList.add("is-active");
    activeMetric = btn.dataset.metric;
    drawChart(activeMetric);
  });
});

langButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    applyLanguage(btn.dataset.lang);
  });
});

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    applyTheme(currentTheme === "dark" ? "light" : "dark");
  });
}

window.addEventListener("resize", () => drawChart(activeMetric));

const revealElements = document.querySelectorAll(".reveal");
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in-view");
      }
    });
  },
  { threshold: 0.12 }
);
revealElements.forEach((el) => observer.observe(el));

function animateCount(el, target, suffix = "") {
  const duration = 900;
  const start = performance.now();
  const from = 0;

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(from + (target - from) * eased);
    el.textContent = `${current}${suffix}`;
    if (progress < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

function animateNumber(el, from, to, duration, formatter) {
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(from + (to - from) * eased);
    el.textContent = formatter(current);
    if (progress < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

function parseNumericText(raw) {
  const normalized = String(raw || "").replace(/[^\d]/g, "");
  if (!normalized) return Number.NaN;
  return Number(normalized);
}

function setupTrafficCounterAnimation(containerId, valueId) {
  const activate = () => {
    const valueEl = document.getElementById(valueId);
    if (!valueEl) return false;
    const target = parseNumericText(valueEl.textContent);
    if (!Number.isFinite(target)) return false;
    animateNumber(valueEl, 0, target, 1100, (value) => value.toLocaleString("en-US"));
    return true;
  };

  if (activate()) return;

  const containerEl = document.getElementById(containerId);
  const observerTarget = containerEl || document.body;
  const observer = new MutationObserver(() => {
    if (activate()) observer.disconnect();
  });

  observer.observe(observerTarget, {
    childList: true,
    characterData: true,
    subtree: true
  });

  setTimeout(() => observer.disconnect(), 30000);
}

const statValues = document.querySelectorAll(".stat-value[data-target]");
statValues.forEach((el) => {
  const target = Number(el.dataset.target);
  const suffix = el.dataset.suffix || "";
  animateCount(el, target, suffix);
});

setupTrafficCounterAnimation("busuanzi_container_site_uv", "busuanzi_value_site_uv");
setupTrafficCounterAnimation("busuanzi_container_site_pv", "busuanzi_value_site_pv");

const copyButton = document.getElementById("copyBib");
const bibtex = document.getElementById("bibtex");
if (copyButton && bibtex) {
  copyButton.addEventListener("click", () => {
    navigator.clipboard
      .writeText(bibtex.innerText.trim())
      .then(() => {
        copyButton.textContent = t("copied");
        setTimeout(() => {
          copyButton.textContent = t("copyBib");
        }, 1200);
      })
      .catch(() => {
        copyButton.textContent = t("copyFailed");
        setTimeout(() => {
          copyButton.textContent = t("copyBib");
        }, 1200);
      });
  });
}

const yearEl = document.getElementById("year");
if (yearEl) {
  yearEl.textContent = String(new Date().getFullYear());
}

applyTheme(currentTheme);
applyLanguage(currentLang);
drawChart(activeMetric);
