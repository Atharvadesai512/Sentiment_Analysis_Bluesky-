# Submission package

## What to use

- **Dissertation.docx** is the college-format report with the completed experiments.
- **Dissertation.pdf** is its rendered reading and printing copy.
- **Research Manuscript.docx** is the concise two-column manuscript draft.
- **Research Manuscript.pdf** is its rendered reading copy.
- **Project Presentation.pptx** is the viva presentation with speaker notes.
- **Project Presentation.pdf** is a slide reading copy.
- **Assessment Guide.md** maps the work to the supplied marking scheme.
- **Viva Preparation.md** contains questions, answers and a demonstration sequence.
- **Evidence** contains the numerical results, predictions, timing records and figures.
- **Source** contains the project code and reproducible analysis scripts.

## Details to fill before submission

Enter your full name, SAP number, guide name, official department name and
academic year in the dissertation title page. Enter author and affiliation
information in the manuscript. The degree is M.Sc. Data Science and AI.

Add the college-issued certificate, declaration, approval and signature pages
when required. These were excluded at your request. Add the official institute
emblem and binding cover through the college template. Update the Word contents
and lists after inserting those pages.

The report uses the supplied M.A./M.Sc. assessment scheme. Confirm it with your
guide. The report interprets the specified 2.5-line paragraph gap as 30 pt after
12 pt paragraphs. Confirm this interpretation before final printing.

## What the experiments establish

All four model instances were scored on the same 12,284 labeled Twitter test
rows. The historical Bluesky analysis measures model outputs and temporal
differences. No manual annotation is required for the completed study.

BiLSTM scores use existing predictions because application control blocks
TensorFlow import on this machine. Its fresh runtime remains unavailable.
Controlled replay measures a queue, classifier and actual project SQLite
insertion. It excludes live network delivery and collector filtering.

The manuscript uses a generic two-column layout. A specific IEEE venue has not
been selected, and publication has not been submitted or accepted. Apply the
chosen venue's template and requirements before submitting.

## Reproduction

Keep the original project folder and its data/model files. The submission
archive includes source code and research evidence, while the original raw
corpus and trained model remain in the project folder to avoid duplicating
large post-text datasets in a submission archive.

The project README lists the execution commands. Run runtime experiments
sequentially after other inference work finishes. Source fingerprints and
package versions are recorded in Evidence/experiment_manifest.json.
