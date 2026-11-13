"""Build the Step 1 venue-neutral manuscript from reproducible evidence."""
from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/publication/manuscript"
OUT.mkdir(parents=True, exist_ok=True)
EVIDENCE = ROOT / "research/publication/evidence"

MODEL_ORDER = ["vader", "linear", "bilstm_saved", "roberta"]
MODEL_NAMES = {"vader": "VADER", "linear": "TF-IDF + LR", "bilstm_saved": "BiLSTM", "roberta": "RoBERTa"}


def set_font(run, name="Times New Roman", size=None, bold=None, italic=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_border(cell, color="D9D9D9"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), color)


def style_table(table):
    table.style = "Table Grid"
    for row_i, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell_border(cell)
            cell.margin_top = Pt(3)
            cell.margin_bottom = Pt(3)
            if row_i == 0:
                shade(cell, "1F4E79")
            elif row_i % 2 == 0:
                shade(cell, "F2F6FA")
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    set_font(run, size=7.2, bold=(row_i == 0), color=((255, 255, 255) if row_i == 0 else None))


def para(doc, text="", bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.0
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_font(r, size=8.8, bold=True)
        r = p.add_run(text[len(bold_lead):])
        set_font(r, size=8.8)
    else:
        r = p.add_run(text)
        set_font(r, size=8.8)
    return p


def heading(doc, number, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(3)
    label = f"{number}. {title.upper()}" if title else str(number).upper()
    r = p.add_run(label)
    set_font(r, size=9.2, bold=True)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    set_font(r, size=7.6, italic=True)
    return p


def table_caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_font(r, size=7.7, bold=True)
    return p


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    for idx, value in enumerate(headers):
        table.cell(0, idx).text = str(value)
    for values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(values):
            cells[idx].text = str(value)
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Inches(width)
    style_table(table)
    return table


def main():
    bench = {m: json.loads((ROOT / f"research/benchmark/{m}_metrics.json").read_text()) for m in MODEL_ORDER}
    q = pd.read_csv(EVIDENCE / "quantification_summary.csv")
    q_primary = q[(q.direction == 0) & (q.bag_size == 500)].set_index(["model", "method"])
    robustness = pd.read_csv(EVIDENCE / "robustness_extended.csv")
    strata = pd.read_csv(EVIDENCE / "labeled_error_strata.csv")
    contributions = pd.read_csv(EVIDENCE / "labeled_error_contribution.csv")
    bluesky_strata = pd.read_csv(EVIDENCE / "bluesky_disagreement_strata.csv").set_index("stratum")
    validation = json.loads((EVIDENCE / "validation_audit.json").read_text(encoding="utf-8"))
    historical = json.loads((EVIDENCE / "analysis_summary.json").read_text())["historical"]
    replay = json.loads((ROOT / "research/replay/summary.json").read_text())
    replay_totals = {(model, rate): (sum(x["stored"] for x in replay if x["model"] == model and x["offered_rate"] == rate), sum(x["dropped"] for x in replay if x["model"] == model and x["offered_rate"] == rate)) for model in ["VADER", "RoBERTa"] for rate in [10, 40, 80]}

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(.68)
    section.bottom_margin = Inches(.68)
    section.left_margin = Inches(.72)
    section.right_margin = Inches(.72)
    section.header_distance = Inches(.25)
    section.footer_distance = Inches(.3)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(8.8)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(5)
    r = title.add_run("How Classifier Errors Change Sentiment Trends under Processing Constraints")
    set_font(r, size=17, bold=True)
    byline = doc.add_paragraph()
    byline.alignment = WD_ALIGN_PARAGRAPH.CENTER
    byline.paragraph_format.space_after = Pt(8)
    r = byline.add_run("Author details supplied at submission")
    set_font(r, size=9, italic=True)
    abstract = doc.add_paragraph()
    abstract.paragraph_format.space_after = Pt(5)
    r = abstract.add_run("Abstract - ")
    set_font(r, size=8.5, bold=True)
    r = abstract.add_run(
        "The useful question for a sentiment monitor is not only which classifier is more accurate, but how its mistakes move the reported trend. "
        "This reproducible study compares VADER, TF-IDF logistic regression, an existing BiLSTM and a Twitter RoBERTa checkpoint on 12,284 labeled TweetEval posts, then traces those errors into aggregate net sentiment. "
        "RoBERTa reaches 0.722 accuracy and 0.724 macro F1. VADER's errors raise mean net sentiment by +0.221, mostly by labeling negative and neutral posts as positive. RoBERTa's smaller error, -0.047, runs in the opposite direction. "
        "A split-calibrated quantification diagnostic on 1,980 bags of 500 posts shows that adjusted counting reduces prevalence mean absolute error from 0.158 to 0.045 for VADER and from 0.099 to 0.026 for RoBERTa. "
        "On 127,741 historical Bluesky posts about AI, the same two models produce opposite aggregate signs. The gap remains after duplicate-text removal, restriction to uncapped collection hours, and a cap of five posts per author. "
        "These Bluesky figures are model outputs, not verified public opinion: only 2 of 300 sampled posts have a human label. "
        "A bounded-queue replay also shows that slower inference can drop records before they are stored."
    )
    set_font(r, size=8.5)
    keywords = doc.add_paragraph()
    keywords.paragraph_format.space_after = Pt(8)
    r = keywords.add_run("Index Terms - ")
    set_font(r, size=8.2, bold=True)
    r = keywords.add_run("sentiment quantification, Bluesky, social media monitoring, RoBERTa, reproducibility")
    set_font(r, size=8.2, italic=True)

    body = doc.add_section(WD_SECTION_START.CONTINUOUS)
    columns = body._sectPr.find(qn("w:cols"))
    columns.set(qn("w:num"), "2")
    columns.set(qn("w:space"), "300")

    heading(doc, "I", "Introduction")
    para(doc, "A sentiment dashboard reports an aggregate, not a single post. Classify-and-count turns each post-level mistake into a change in that aggregate. Two models can therefore disagree about whether a topic is becoming more positive even when both are used on the same posts. This paper measures that effect. It asks which errors move mean net sentiment, whether the movement survives ordinary data-cleaning choices, and whether processing cost changes which posts are stored at all.")
    para(doc, "The contribution is measurement evidence, not a new Bluesky monitor. Monitoring systems for the AT Protocol already exist, including CognitiveSky [9]. The present study instead reports three results that those systems do not settle: how confusion between classes shifts a trend on labeled data, whether a historical Bluesky trend changes sign across classifiers after duplicate, coverage and author-cap checks, and how inference cost changes completed throughput in a bounded queue.")

    heading(doc, "II", "Related Work")
    para(doc, "TweetEval provides a standardized Twitter classification benchmark [1]. Twitter-specific language models and the Cardiff NLP checkpoint used here account for temporal and domain adaptation in social-media language [2], [3]. VADER supplies a transparent lexicon baseline [4], while bidirectional recurrent models provide the basis for the existing BiLSTM [5], [6].")
    para(doc, "Sentiment quantification estimates class prevalence in an unlabeled collection. Moreo and Sebastiani show that post-level accuracy can be the wrong evaluation when the downstream goal is prevalence estimation [7]. Daughton and Paul study the bias introduced by classify-and-count [8]. CognitiveSky demonstrates scalable sentiment and narrative analysis for decentralized social media [9], and the AT Protocol makes public Bluesky research technically feasible [10]. This paper does not propose another monitoring architecture. It uses one fixed corpus to show how classifier disagreement and processing limits change the quantity a monitor would publish.")

    heading(doc, "III", "Experimental Design")
    para(doc, "All four models receive the same 12,284 TweetEval test rows. VADER applies compound thresholds of -0.05 and +0.05. The learned baseline is TF-IDF over unigrams and bigrams with logistic regression trained only on 45,615 local training rows. The BiLSTM uses row-checked saved predictions because fresh TensorFlow inference is unavailable on this host. RoBERTa uses cardiffnlp/twitter-roberta-base-sentiment-latest at a cached revision, original local text, 128-token truncation and CPU inference.")
    para(doc, "For the quantification diagnostic, the labeled test rows are stratified into disjoint halves of 6,141 and 6,143 rows with seed 42. One half estimates each model's predicted-by-true three-class confusion matrix. The other half supplies 66 prevalence settings, each repeated 30 times at bag sizes 100, 500 and 1,000. Samples are drawn with replacement within each true class, and every model receives the same bags. Classify-and-count uses prediction shares. Adjusted classify-and-count estimates the non-negative class-prevalence vector that best explains observed prediction shares under the calibration matrix. The primary result uses n=500; the direction is reversed at n=500 as a sensitivity check. This is a diagnostic on an already inspected benchmark, not an independent benchmark or a correction asserted for Bluesky.")
    para(doc, "The historical corpus contains 127,741 cleaned English posts returned for the literal query AI between 19 October and 1 November 2026 UTC. A search response cap can truncate a window. We therefore report duplicate-text, uncapped-hour, literal-token and equal-author sensitivity variants. Net sentiment is the mean of label scores: negative=-1, neutral=0 and positive=1. No historical post labels are used as ground truth.")

    table_caption(doc, "TABLE I. COMMON LABELED TEST PERFORMANCE")
    benchmark_rows = []
    for model in MODEL_ORDER:
        report = bench[model]["report"]
        benchmark_rows.append([MODEL_NAMES[model], f"{report['accuracy']:.3f}", f"{report['macro avg']['f1-score']:.3f}", f"{report['macro avg']['recall']:.3f}"])
    add_table(doc, ["Model", "Accuracy", "Macro F1", "Macro recall"], benchmark_rows, [1.10, .52, .60, .68])
    caption(doc, "Scores use the same 12,284 TweetEval test posts. BiLSTM values come from validated saved predictions.")

    heading(doc, "IV", "Results")
    para(doc, "RoBERTa obtains the strongest observed benchmark result, with 0.722 accuracy and 0.724 macro F1. The linear baseline and saved BiLSTM scores are close on this split, so their apparent ordering should not be treated as an architecture-only finding. RoBERTa's superior post-level result motivates testing whether its advantage remains meaningful for the aggregate task.")
    table_caption(doc, "TABLE II. PRIMARY QUANTIFICATION DIAGNOSTIC AT BAG SIZE 500")
    quant_rows = []
    for model in MODEL_ORDER:
        cc = q_primary.loc[(model, "CC")]
        acc = q_primary.loc[(model, "ACC")]
        quant_rows.append([MODEL_NAMES[model], f"{cc.prevalence_mae:.3f}", f"{acc.prevalence_mae:.3f}", f"{cc.strict_sign_error_rate:.1%}", f"{acc.strict_sign_error_rate:.1%}"])
    add_table(doc, ["Model", "CC P-MAE", "ACC P-MAE", "CC sign err.", "ACC sign err."], quant_rows, [.78, .50, .56, .60, .65])
    caption(doc, "P-MAE is mean absolute error across negative, neutral and positive prevalence. Sign error excludes true net sentiment within +/-0.1; 1,800 bags are eligible.")
    figure = doc.add_paragraph()
    figure.alignment = WD_ALIGN_PARAGRAPH.CENTER
    figure.paragraph_format.space_after = Pt(1)
    figure.add_run().add_picture(str(OUT / "quantification_diagnostic.png"), width=Inches(3.25))
    caption(doc, "Fig. 1. Disjoint split-calibrated quantification diagnostic at n=500. Adjusted counting improves each observed model under this evaluation protocol.")
    para(doc, "At n=500, simple counting has prevalence mean absolute error of 0.099 for RoBERTa, 0.158 for VADER, 0.193 for the linear baseline and 0.211 for the saved BiLSTM. Adjusted counting reduces these values to 0.026, 0.045, 0.044 and 0.048. RoBERTa's aggregate net-sentiment error falls from 0.123 to 0.034. The reverse split keeps the same ordering for adjusted-count prevalence error. Accuracy and trend error are related, but they are not the same result: the weaker classifiers do not merely miss more posts, they shift the reported mean in a consistent direction.")

    overall = strata[strata.scope == "all_labeled_test"].set_index("model")
    vader_bias = overall.loc["VADER", "mean_net_error"]
    roberta_bias = overall.loc["RoBERTa", "mean_net_error"]
    table_caption(doc, "TABLE III. LARGEST CONTRIBUTIONS TO MEAN NET-SENTIMENT ERROR")
    error_rows = []
    for model in ["VADER", "RoBERTa"]:
        part = contributions[contributions.model == model]
        part = part[part.true_label != part.predicted_label].sort_values(
            "contribution_to_mean_net_error", key=lambda s: s.abs(), ascending=False).head(3)
        for _, row in part.iterrows():
            error_rows.append([
                model, f"{row.true_label} → {row.predicted_label}", f"{int(row.posts):,}",
                f"{row.contribution_to_mean_net_error:+.3f}",
            ])
    add_table(doc, ["Model", "Error", "Posts", "Contribution"], error_rows, [0.62, 1.25, 0.55, 0.85])
    caption(doc, "Contribution is (count × score change) / 12,284. Scores are negative=−1, neutral=0, positive=+1. The sum of all nine cells equals the model's mean net error.")
    para(doc, f"On the labeled test set, VADER's mean net sentiment is {vader_bias:+.3f} above the gold mean. The two largest cells are negative posts called positive (+0.167) and neutral posts called positive (+0.168). Together they more than account for the upward shift; other errors pull the mean back down. RoBERTa's mean error is {roberta_bias:+.3f}. Its largest cell is the opposite kind of mistake: 1,372 neutral posts called negative, contributing −0.112. VADER marks a non-positive post as positive on {overall.loc['VADER', 'positive_overcall_share']:.1%} of test rows, compared with {overall.loc['RoBERTa', 'positive_overcall_share']:.1%} for RoBERTa. This is the mechanism behind the quantification result. Classify-and-count does not add random noise. It adds a signed bias that follows each model's confusion pattern.")
    para(doc, "Surface cues do not overturn that pattern, and several of them are too rare for a separate accuracy claim. VADER remains positively biased on posts that contain a URL (n=376, mean net error +0.165) and on posts with a news-style cue (n=89, +0.124). RoBERTa's error on those same slices is +0.000 and −0.045. Short posts, defined as six words or fewer (n=828), are easier for every model, and VADER's bias there falls to −0.018. Only 11 labeled posts match the sarcasm cue, so that slice is an illustration, not a result.")

    table_caption(doc, "TABLE IV. ILLUSTRATIVE DISAGREEMENTS")
    add_table(
        doc,
        ["Setting", "Gold", "VADER", "BiLSTM", "RoBERTa", "Excerpt"],
        [
            ["Labeled", "neutral", "positive", "neutral", "neutral", "Bike parking still free."],
            ["Labeled", "neutral", "negative", "positive", "neutral", "An interesting security vulnerability — albeit not for the everyday car thief"],
            ["Labeled", "neutral", "negative", "neutral", "neutral", "…yeah right!!!"],
            ["Bluesky", "none", "positive", "negative", "negative", "…ZERO way I can trust a company pushing this level of AI evangelism…"],
            ["Bluesky", "none", "neutral", "neutral", "negative", "TVA using Ai images yuck"],
            ["Bluesky", "none", "positive", "neutral", "negative", "Apparently her lawsuit was dismissed … www.theguardian.com/…"],
        ],
        [0.48, 0.42, 0.46, 0.48, 0.52, 0.94],
    )
    caption(doc, "Labeled excerpts have TweetEval gold labels. Bluesky excerpts have no gold label. They show model disagreement only. The full strings and row identifiers are in research/publication/evidence/error_examples.csv.")
    para(doc, "The excerpts show the same mechanism in ordinary text. VADER treats an apparently factual short post containing \"free\" as positive. A negated, strongly worded Bluesky complaint about AI is positive under VADER and negative under both neural models. A one-line critical post is negative only under RoBERTa. A news-style post with a URL splits three ways. None of the Bluesky lines can be scored as correct or incorrect.")

    table_caption(doc, "TABLE V. HISTORICAL NET SENTIMENT AFTER PROCESSING CHECKS")
    rows = []
    labels = [
        ("all_posts", "All posts"),
        ("unique_normalized_text", "Unique normalized text"),
        ("uncapped_hours", "Uncapped hours only"),
        ("at_most_five_posts_per_author", "At most five posts per author"),
        ("unique_uncapped_and_capped", "Unique, uncapped, and capped"),
    ]
    for analysis, label in labels:
        part = robustness[robustness.analysis == analysis].set_index("model")
        rows.append([
            label, f"{int(part.loc['VADER'].units):,}",
            f"{part.loc['VADER'].net_sentiment:+.3f}",
            f"{part.loc['RoBERTa'].net_sentiment:+.3f}",
            f"{part.loc['VADER_minus_RoBERTa'].net_sentiment:+.3f}",
        ])
    add_table(doc, ["Variant", "Posts", "VADER", "RoBERTa", "Gap"], rows, [1.22, 0.50, 0.50, 0.55, 0.48])
    caption(doc, "The gap is VADER minus RoBERTa. Positive and negative values are model outputs. They are not estimates of public opinion.")
    figure = doc.add_paragraph()
    figure.alignment = WD_ALIGN_PARAGRAPH.CENTER
    figure.paragraph_format.space_after = Pt(1)
    figure.add_run().add_picture(str(OUT / "robustness_diagnostic.png"), width=Inches(3.25))
    caption(doc, "Fig. 2. The VADER and RoBERTa historical means keep opposite signs after each processing check.")
    ai_gap = bluesky_strata.loc["ai_term"]
    short_gap = bluesky_strata.loc["short_post"]
    para(doc, f"The two models give opposite signs in all {historical['days']} observed daily means. The daily equal-weight VADER-minus-RoBERTa difference is {historical['daily_equal_weight_vader_minus_roberta']:.3f}, with a day-resampling interval from {historical['day_resampling_95_interval'][0]:.3f} to {historical['day_resampling_95_interval'][1]:.3f}. The most active 1% of authors produce {historical['top_one_percent_authors_post_share']:.1%} of posts, so author dominance is a real risk. It does not create the sign split. Keeping each author's earliest five posts leaves VADER at +0.076 and RoBERTa at −0.318. Applying unique text, uncapped hours and that author cap together leaves 49,728 posts and widens the gap to +0.408. At least one pair of models disagrees on {bluesky_strata.loc['all_posts'].disagreement_rate:.1%} of posts. The gap is +{ai_gap.vader_minus_roberta:.3f} on posts with an explicit AI term ({ai_gap.posts:,.0f} posts) and only +{short_gap.vader_minus_roberta:.3f} on short posts. Short text is therefore not the main source of the historical disagreement.")
    para(doc, f"The controlled replay offers 288 posts per model-rate condition across three runs. VADER stores all 288 records at 10, 40 and 80 posts/s. RoBERTa stores {replay_totals[('RoBERTa', 10)][0]}, {replay_totals[('RoBERTa', 40)][0]} and {replay_totals[('RoBERTa', 80)][0]} records respectively, with the remaining posts dropped when the queue is full. These finite tests include classifier execution and SQLite insertion, but exclude live network delivery, filtering and collector orchestration.")

    heading(doc, "V", "Discussion and Limitations")
    para(doc, "The finding is that classifier errors change the trend a monitor would publish. On labeled data, VADER's positive over-calls raise mean net sentiment by about 0.22, while RoBERTa's errors lower it by about 0.05. Adjusted counting reduces that source-domain error, which shows the damage is systematic rather than an unavoidable property of aggregation. On the historical Bluesky corpus, the same models keep opposite signs after the checks a reviewer would ask for: duplicate text, truncated collection windows, and highly active users. A reader should treat those Bluesky series as a sensitivity result. They show that model choice is large enough to reverse the headline number. They do not show what Bluesky users thought about AI.")
    para(doc, f"Target-domain validation is the remaining gap. A 300-post sample was drawn for human labeling, but {validation['labeled_posts']} posts have a label and {validation['unlabeled_posts']} do not. Two labels cannot support accuracy, kappa, or a corrected Bluesky prevalence. A useful validation set would contain 100 to 200 posts, labeled without seeing model output, with a second annotator on a subset. Until that exists, every Bluesky number in this paper is a model output.")
    para(doc, "Other limits are narrower. The quantification diagnostic reuses TweetEval, so the adjusted-count correction is not shown to transfer to Bluesky. The historical search uses one query, two weeks, and an API that can truncate a window. Polarity is not stance toward AI. Sarcasm and news cues are string heuristics; the labeled sarcasm slice has only 11 posts. The BiLSTM benchmark uses saved predictions because TensorFlow cannot be imported on this host. The replay uses one machine and a finite workload. Disagreement excerpts are illustrations chosen from the first matching rows, not a random sample of errors.")

    heading(doc, "VI", "Conclusion")
    para(doc, "RoBERTa is the strongest of the four classifiers on the shared labeled test, and its errors move aggregate net sentiment less than VADER's. That post-level advantage matters because the errors are signed: VADER systematically inflates the trend, and the two models then produce opposite Bluesky aggregates under duplicate, coverage and author-cap checks. Adjusted counting reduces the labeled trend error, and the replay shows that the more accurate model can also store fewer posts when the queue is bounded. The study supports a narrower claim than a public-opinion paper. Classifier choice and processing constraints change the measured trend. They do not, on the present evidence, establish what Bluesky users believed.")

    heading(doc, "Data, Ethics, and Reproducibility", "")
    para(doc, "Historical posts were retrieved from public Bluesky search for the query AI. The released evidence contains aggregate tables, prediction files without historical text, and short disagreement excerpts. Full historical text, databases and model weights are kept out of version control. The work follows the platform's access method for public posts and does not attempt to identify private accounts beyond the public author identifier needed for the author-cap check. No Bluesky user was contacted. TweetEval is used under its published benchmark terms. An AI coding assistant helped implement the analysis scripts and edit the prose against the author's result tables. The author remains responsible for the design, the checks, and every reported claim. Regeneration steps are in the project README; the new tables come from sentiment/msc_paper_analysis.py.")
    heading(doc, "References", "")
    references = [
        "[1] F. Barbieri, J. Camacho-Collados, L. Espinosa Anke and L. Neves, \"TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification,\" Findings of EMNLP, pp. 1644-1650, 2020.",
        "[2] D. Loureiro, F. Barbieri, L. Neves, L. Espinosa Anke and J. Camacho-Collados, \"TimeLMs: Diachronic Language Models from Twitter,\" ACL System Demonstrations, pp. 251-260, 2022.",
        "[3] Cardiff NLP, \"twitter-roberta-base-sentiment-latest,\" model card, accessed Nov. 2026.",
        "[4] C. J. Hutto and E. Gilbert, \"VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text,\" ICWSM, pp. 216-225, 2014.",
        "[5] S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.",
        "[6] M. Schuster and K. K. Paliwal, \"Bidirectional Recurrent Neural Networks,\" IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673-2681, 1997.",
        "[7] A. Moreo and F. Sebastiani, \"Tweet Sentiment Quantification: An Experimental Re-Evaluation,\" PLOS ONE, vol. 17, no. 9, e0263449, 2022.",
        "[8] A. R. Daughton and M. J. Paul, \"A Bootstrapping Approach to Social Media Quantification,\" Social Network Analysis and Mining, vol. 11, art. 73, 2021.",
        "[9] G. Chhetri, A. Dutta and S. Das, \"CognitiveSky: Scalable Sentiment and Narrative Analysis for Decentralized Social Media,\" HICSS, 2026.",
        "[10] M. Kleppmann et al., \"Bluesky and the AT Protocol: Usable Decentralized Social Media,\" DIN, 2024.",
    ]
    for item in references:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(item)
        set_font(r, size=7.5)

    footer = body.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("Venue-neutral manuscript draft")
    set_font(r, size=7, italic=True, color=(90, 90, 90))
    final = OUT / "Step 1 Quantification Manuscript.docx"
    doc.save(final)
    print(final)


if __name__ == "__main__":
    main()
