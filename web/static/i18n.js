/* ═══════════════════════════════════════════════════════════
   MLP Dashboard – i18n  (Chinese / English)
   ═══════════════════════════════════════════════════════════ */

const I18N = {
    // ── Sidebar ──────────────────────────────────────────────
    "sidebar.title":    { zh: "MLP 控制台",           en: "MLP Dashboard" },
    "sidebar.subtitle": { zh: "对抗训练调度面板",      en: "Adversarial Training Panel" },
    "sidebar.batches":  { zh: "批次管理",              en: "Batch Management" },
    "sidebar.new":      { zh: "+ 新建批次",            en: "+ New Batch" },
    "sidebar.refresh":  { zh: "刷新列表",              en: "Refresh List" },
    "sidebar.empty":    { zh: "暂无批次",              en: "No batches yet" },

    // ── Topbar ──────────────────────────────────────────────
    "topbar.placeholder": { zh: "选择或新建一个批次",  en: "Select or create a batch" },
    "topbar.new_hint":    { zh: "尚未开始运行",         en: "Not started yet" },
    "topbar.download":    { zh: "下载模型 (.npz)",     en: "Download Model (.npz)" },
    "topbar.delete":      { zh: "删除批次",            en: "Delete Batch" },

    // ── Tabs ────────────────────────────────────────────────
    "tab.pipeline":  { zh: "流水线",    en: "Pipeline" },
    "tab.results":   { zh: "训练结果",  en: "Results" },
    "tab.preview":   { zh: "图像预览",  en: "Preview" },
    "tab.logs":      { zh: "日志",      en: "Logs" },

    // ── Parameters ──────────────────────────────────────────
    "params.title":         { zh: "参数配置",          en: "Parameters" },
    "params.n.label":       { zh: "每类图像数量",      en: "Images per class" },
    "params.n.hint":        { zh: "sphere / cube / tetrahedron 各生成 N 张", en: "N images each for sphere, cube, tetrahedron" },
    "params.size.label":    { zh: "图像尺寸",          en: "Image size" },
    "params.size.hint":     { zh: "N×N 像素 (默认 32×32=1024)", en: "N×N pixels (default 32×32=1024)" },
    "params.seed.label":    { zh: "随机种子",          en: "Random seed" },
    "params.seed.hint":     { zh: "保证可复现性",      en: "For reproducibility" },
    "params.ss.label":      { zh: "超采样抗锯齿",      en: "Supersampling" },
    "params.ss.hint":       { zh: "内部渲染倍率 (1=无, 3=推荐)", en: "Internal render scale (1=off, 3=recommended)" },
    "params.epochs.label":  { zh: "主模型训练轮数",    en: "Main model epochs" },
    "params.epochs.hint":   { zh: "主对抗训练 Epochs", en: "Main adversarial training epochs" },
    "params.sur.label":     { zh: "替代模型训练轮数",   en: "Surrogate epochs" },
    "params.sur.hint":      { zh: "每个 surrogate 的 Epochs", en: "Epochs per surrogate model" },
    "params.num_sur.label": { zh: "替代模型数量", en: "Number of surrogates" },
    "params.num_sur.hint":  { zh: "集成攻击中使用的替代模型数 (默认 5)", en: "Surrogate models in ensemble attack (default 5)" },

    // ── Architecture Config ─────────────────────────────────
    "arch.title":        { zh: "替代模型架构配置",    en: "Surrogate Architecture Config" },
    "arch.mode":         { zh: "架构模式:",           en: "Architecture mode:" },
    "arch.auto":         { zh: "默认自动",            en: "Default Auto" },
    "arch.custom":       { zh: "自定义",              en: "Custom" },
    "arch.auto_hint":    { zh: "根据替代模型数量自动分配多样化架构", en: "Auto-assign diverse architectures based on count" },
    "arch.custom_hint":  { zh: "手动指定每个替代模型的隐藏层和 Dropout", en: "Manually specify hidden layers and dropout for each surrogate" },
    "arch.col_h1":       { zh: "隐藏层 1",    en: "Hidden Layer 1" },
    "arch.col_h2":       { zh: "隐藏层 2",    en: "Hidden Layer 2" },
    "arch.col_dropout":  { zh: "Dropout",      en: "Dropout" },
    "arch.col_params":   { zh: "参数量",       en: "Params" },
    "arch.row_remove":   { zh: "移除",         en: "Remove" },

    // ── PGD Parameters ──────────────────────────────────────
    "pgd.title":            { zh: "PGD 攻击参数",         en: "PGD Attack Parameters" },
    "pgd.surrogate":        { zh: "替代模型 PGD",         en: "Surrogate PGD" },
    "pgd.sur_eps_hint":     { zh: "L∞ 扰动上限 (默认 0.03)", en: "L∞ perturbation bound (default 0.03)" },
    "pgd.sur_alpha_hint":   { zh: "每次迭代步长 (默认 0.01)", en: "Step size per iteration (default 0.01)" },
    "pgd.sur_steps_hint":   { zh: "攻击迭代次数 (默认 5)",    en: "Attack iterations (default 5)" },
    "pgd.main":             { zh: "主模型对抗训练 PGD",   en: "Main Model Adversarial PGD" },
    "pgd.main_eps_hint":    { zh: "L∞ 扰动上限 (默认 0.06)", en: "L∞ perturbation bound (default 0.06)" },
    "pgd.main_alpha_hint":  { zh: "每次迭代步长 (默认 0.01)", en: "Step size per iteration (default 0.01)" },
    "pgd.main_steps_hint":  { zh: "攻击迭代次数 (默认 10)",   en: "Attack iterations (default 10)" },
    "pgd.lambda_hint":      { zh: "loss = (1-λ)×clean + λ×adv (默认 0.6)", en: "loss = (1-λ)×clean + λ×adv (default 0.6)" },
    "pgd.eps_label":       { zh: "ε (扰动半径)",          en: "ε (Perturbation Radius)" },
    "pgd.alpha_label":     { zh: "α (步长)",              en: "α (Step Size)" },
    "pgd.steps_label":     { zh: "PGD 迭代次数",          en: "PGD Iterations" },
    "pgd.lambda_label":    { zh: "λ (对抗损失权重)",      en: "λ (Adversarial Loss Weight)" },

    // ── Pipeline controls ───────────────────────────────────
    "pipeline.title":     { zh: "流水线控制",           en: "Pipeline Control" },
    "pipeline.run_all":   { zh: "▶ 一键运行全部",       en: "▶ Run All" },
    "pipeline.step1":     { zh: "1. 生成图像",          en: "1. Generate Images" },
    "pipeline.step2":     { zh: "2. 归一化",            en: "2. Normalize" },
    "pipeline.step3":     { zh: "3. 准备数据",          en: "3. Prepare Data" },
    "pipeline.step4":     { zh: "4. 训练替代模型",       en: "4. Train Surrogates" },
    "pipeline.step5":     { zh: "5. 主对抗训练",         en: "5. Main Training" },
    "pipeline.cancel":    { zh: "取消运行",             en: "Cancel" },

    // ── Progress states ─────────────────────────────────────
    "progress.pending":   { zh: "等待中",  en: "Pending" },
    "progress.running":   { zh: "运行中",  en: "Running" },
    "progress.completed": { zh: "已完成",  en: "Completed" },
    "progress.failed":    { zh: "失败",    en: "Failed" },

    // ── Results ─────────────────────────────────────────────
    "results.title":        { zh: "训练结果",            en: "Training Results" },
    "results.empty":        { zh: "训练完成后此处将显示结果", en: "Results will appear here after training" },
    "results.accuracy":     { zh: "总体准确率",          en: "Overall Accuracy" },
    "results.table.class":  { zh: "类别",               en: "Class" },
    "results.table.prec":   { zh: "精确率",             en: "Precision" },
    "results.table.recall": { zh: "召回率",             en: "Recall" },
    "results.table.f1":     { zh: "F1",                 en: "F1" },
    "results.table.support":{ zh: "样本数",             en: "Support" },
    "results.chart_title":  { zh: "训练曲线",            en: "Training Curves" },
    "results.clean_acc":    { zh: "干净准确率",          en: "Clean Acc" },
    "results.adv_acc":      { zh: "对抗准确率",          en: "Adv Acc" },

    // ── Preview ─────────────────────────────────────────────
    "preview.title":   { zh: "图像预览",              en: "Image Preview" },
    "preview.sphere":  { zh: "球体",                  en: "Sphere" },
    "preview.cube":    { zh: "立方体",                en: "Cube" },
    "preview.tetra":   { zh: "四面体",                en: "Tetrahedron" },
    "preview.empty":   { zh: "暂无图像数据 — 请先生成", en: "No image data — generate first" },

    // ── Logs ────────────────────────────────────────────────
    "logs.title":        { zh: "运行日志",            en: "Run Logs" },
    "logs.select_hint":  { zh: "选择步骤查看日志…",    en: "Select a step to view log…" },
    "logs.empty":        { zh: "(日志为空)",           en: "(empty log)" },
    "logs.step.generate":  { zh: "生成图像",          en: "Generate" },
    "logs.step.normalize": { zh: "归一化",            en: "Normalize" },
    "logs.step.prepare":   { zh: "准备数据",          en: "Prepare" },
    "logs.step.surrogate": { zh: "替代模型训练",       en: "Surrogate Training" },
    "logs.step.train_main":{ zh: "主对抗训练",         en: "Main Training" },

    // ── Dynamic JS strings ──────────────────────────────────
    "batch.running":   { zh: "运行中",   en: "Running" },
    "batch.done":      { zh: "完成",     en: "Done" },
    "batch.failed":    { zh: "失败",     en: "Failed" },
    "steps_complete":  { zh: "步骤完成",  en: "steps done" },
    "cancel_confirm":  { zh: "确认取消当前运行？", en: "Confirm cancel current run?" },
    "delete_confirm":  { zh: "确认删除此批次及其所有数据？此操作不可撤销。", en: "Delete this batch and all its data? This cannot be undone." },
    "delete_failed":   { zh: "删除失败: ", en: "Delete failed: " },
    "error_unknown":   { zh: "未知错误", en: "Unknown error" },
    "launch_failed":   { zh: "启动失败: ", en: "Launch failed: " },
    "class.sphere":    { zh: "球体",      en: "Sphere" },
    "class.cube":      { zh: "立方体",    en: "Cube" },
    "class.tetrahedron":{ zh: "四面体",   en: "Tetrahedron" },
    "side.quick":      { zh: "快捷操作",  en: "Quick Actions" },
    "server.label":    { zh: "服务器:",    en: "Server:" },

    // ── Noise Configuration ───────────────────────────────────
    "noise.title":          { zh: "噪声配置",            en: "Noise Configuration" },
    "noise.preset_clean":   { zh: "仅干净数据",          en: "Clean Only" },
    "noise.preset_uniform": { zh: "均匀分布",            en: "Uniform" },
    "noise.total":          { zh: "总计",               en: "Total" },

    // ── Brightness Shift Attack ───────────────────────────────
    "shift.title":            { zh: "亮度偏移攻击",        en: "Brightness Shift Attack" },
    "shift.magnitude":        { zh: "偏移强度",            en: "Shift Magnitude" },
    "shift.magnitude_hint":   { zh: "0=无, 255=最大 (可完全翻转)", en: "0=off, 255=max (can fully invert)" },
    "shift.probability":      { zh: "发生概率",            en: "Probability" },
    "shift.probability_hint": { zh: "图像被偏移的概率 (%)",   en: "Chance an image gets shifted (%)" },

    // ── Adversarial Evaluation ────────────────────────────────
    "tab.eval":              { zh: "对抗评估",           en: "Adversarial Eval" },
    "eval.title":            { zh: "对抗评估",           en: "Adversarial Evaluation" },
    "eval.model_label":      { zh: "选择模型 (.npz)",   en: "Select Model (.npz)" },
    "eval.model_placeholder":{ zh: "-- 选择已训练的模型 --", en: "-- Select a trained model --" },
    "eval.data_label":       { zh: "选择数据集",          en: "Select Dataset" },
    "eval.data_placeholder": { zh: "-- 选择批次数据集 --", en: "-- Select batch dataset --" },
    "eval.run":              { zh: "▶ 运行评估",          en: "▶ Run Evaluation" },
    "eval.running":          { zh: "评估中…",            en: "Evaluating…" },
    "eval.overview":         { zh: "评估概览",           en: "Overview" },
    "eval.overall_acc":      { zh: "总体准确率",          en: "Overall Accuracy" },
    "eval.avg_conf":         { zh: "平均置信度",          en: "Average Confidence" },
    "eval.total_samples":    { zh: "样本总数",           en: "Total Samples" },
    "eval.model_info":       { zh: "模型: {model} | 数据集: {data}", en: "Model: {model} | Dataset: {data}" },
    "eval.per_class":        { zh: "每类准确率",          en: "Per-Class Accuracy" },
    "eval.class_label":      { zh: "类别",               en: "Class" },
    "eval.class_acc":        { zh: "准确率",             en: "Accuracy" },
    "eval.class_count":      { zh: "样本数",             en: "Count" },
    "eval.class_bar":        { zh: "比例",               en: "Bar" },
    "eval.deciles":          { zh: "置信度十分组",        en: "Confidence Deciles" },
    "eval.deciles_hint":     { zh: "按置信度从低到高排序后分为10组", en: "Sorted by confidence (low→high), split into 10 groups" },
    "eval.decile_group":     { zh: "分组",               en: "Group" },
    "eval.decile_range":     { zh: "置信度区间",          en: "Confidence Range" },
    "eval.decile_acc":       { zh: "准确率",             en: "Accuracy" },
    "eval.decile_conf":      { zh: "平均置信度",          en: "Avg Confidence" },
    "eval.decile_count":     { zh: "样本数",             en: "Count" },
    "eval.decile_bar":       { zh: "准确率",             en: "Accuracy" },
    "eval.chart_title":      { zh: "置信度-准确率曲线",    en: "Confidence vs Accuracy" },
    "eval.no_data":          { zh: "暂无评估结果",        en: "No evaluation results yet" },
    "eval.error":            { zh: "评估失败: ",          en: "Evaluation failed: " },
};

// ── current language (persisted) ─────────────────────────────
let LANG = localStorage.getItem("mlp-lang") || "zh";

function setLang(lang) {
    LANG = lang;
    localStorage.setItem("mlp-lang", lang);
    applyLanguage();
}

function t(key) {
    const entry = I18N[key];
    if (!entry) return key;           // missing key → show key itself for debugging
    return entry[LANG] || entry["en"] || key;
}

function applyLanguage() {
    // 1. elements with data-i18n
    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.dataset.i18n;
        el.textContent = t(key);
    });

    // 2. elements with data-i18n-placeholder
    document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
        el.placeholder = t(el.dataset.i18nPlaceholder);
    });

    // 3. update dynamic UI state
    refreshBatches(true);
    if (currentBatch) {
        pollStatus(true);
        // refresh current tab content
        const activeTab = document.querySelector(".tab-content.active");
        if (activeTab) {
            if (activeTab.id === "tab-results") loadResults();
            if (activeTab.id === "tab-preview") loadPreviews(previewCls);
            if (activeTab.id === "tab-logs") loadLog(document.getElementById("log-step-selector").value);
        }
    }

    // 4. language toggle label
    const toggle = document.getElementById("lang-toggle");
    if (toggle) {
        toggle.textContent = LANG === "zh" ? "EN" : "中";
    }

    // 5. rebuild arch editor if custom mode is active
    if (typeof archMode !== "undefined" && archMode === "custom" && typeof buildArchEditor === "function") {
        buildArchEditor();
    }
}

// auto-apply on DOM ready
document.addEventListener("DOMContentLoaded", () => {
    applyLanguage();
});
