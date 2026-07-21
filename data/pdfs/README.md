# Sample earnings PDF corpus

Retrieved on 2026-07-09. Each file was checked for a `%PDF` signature and
opened with `pypdf.PdfReader`.

## Apple

- File: `apple/FY25_Q1_Consolidated_Financial_Statements-normalized.pdf`
- Original direct asset URL: https://www.apple.com/newsroom/pdfs/fy2025-q1/FY25_Q1_Consolidated_Financial_Statements.pdf
- Source landing page: https://www.apple.com/newsroom/2025/01/apple-reports-first-quarter-results/
- Publisher: Apple Inc.
- Period: FY2025 Q1, three months ended 2024-12-28
- Publication date: 2025-01-30
- Document type: Unaudited condensed consolidated financial statements,
  locally normalized PDF
- Retrieval date: 2026-07-09
- File size: 3,132,440 bytes
- SHA-256: `b2a27fff704a3b1b0aab6df26a6c63d3e3893f193ddc0cdd7159dd4bec1ace79`
- Normalization: The publisher PDF's cross-reference table was not
  zero-indexed. Because `qpdf` was not installed, it was read with
  `pypdf.PdfReader(strict=False)` and rewritten with `pypdf.PdfWriter`
  version 5.4.0. A strict read of the normalized file produces no warning or
  error. Page count remains 3, and the SHA-256 signatures of extracted text for
  all three pages match before and after normalization:
  `9641ca6d909198bb9fca14662e2857c35890a35071878c23c04c07f03bec24d9`,
  `c6b6155fb120ff138f5fb373e9d5f9e5df696435e2d8a8ce3fa8eecca83f9848`,
  and `f2022b0f18a46f7d8c29c3ce2923a32e1d50bbfc3ae6912a1eb30e94a7efd32c`.
- Normalization method:

  ```python
  reader = PdfReader(source_path, strict=False)
  writer = PdfWriter()
  writer.append_pages_from_reader(reader)
  writer.add_metadata({"/Producer": "pypdf 5.4.0 structural normalization"})
  with open(output_path, "wb") as output:
      writer.write(output)
  ```

## Microsoft

- File: `microsoft/TranscriptFY25Q2-DOCX-locally-converted.pdf`
- Source landing page: https://www.microsoft.com/en-us/investor/events/fy-2025/earnings-fy-2025-q2
- Official asset package URL: https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/FY25Q2-zip
- Exact package member: `TranscriptFY25Q2.docx`
- Publisher: Microsoft Corporation
- Period: FY2025 Q2, quarter ended 2024-12-31
- Event/publication date: 2025-01-29
- Document type: Official earnings-call transcript DOCX locally converted to
  PDF
- Retrieval date: 2026-07-09
- File size: 103,391 bytes
- SHA-256: `95d205b770bc836a566efb0855d656cf8fff0cbb2f32ef315e9640b7a6699e2a`
- Conversion tools: `/usr/bin/textutil` bundled with macOS 26.5.1 converted the
  DOCX to local HTML; Google Chrome 150.0.7871.114 printed that document-only
  HTML to PDF; `pypdf` 5.4.0 performed a final metadata-controlled rewrite.
- Provenance note: The official asset package contained no PDF. This file was
  locally converted from the official `TranscriptFY25Q2.docx` package member,
  not from the investor-relations webpage. It contains the document content
  without webpage navigation, live stock information, or promotions. It is not
  a publisher-supplied PDF, and there is no separate direct PDF asset URL.
- Exact conversion command pattern (`$PACKAGE`, `$WORK`, `$RAW_PDF`, and
  `$OUTPUT_PDF` are explicit local paths):

  ```sh
  unzip -p "$PACKAGE" "TranscriptFY25Q2.docx" > "$WORK/TranscriptFY25Q2.docx"
  /usr/bin/textutil -convert html -output "$WORK/TranscriptFY25Q2.html" "$WORK/TranscriptFY25Q2.docx"
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer --run-all-compositor-stages-before-draw --virtual-time-budget=1000 --user-data-dir="$WORK/chrome-profile" --print-to-pdf="$RAW_PDF" "file://$WORK/TranscriptFY25Q2.html"
  python - "$RAW_PDF" "$OUTPUT_PDF" <<'PY'
  from pypdf import PdfReader, PdfWriter
  import sys

  reader = PdfReader(sys.argv[1], strict=True)
  writer = PdfWriter()
  writer.append_pages_from_reader(reader)
  writer.add_metadata({
      "/Producer": "pypdf 5.4.0 deterministic rewrite after textutil/Chrome conversion"
  })
  with open(sys.argv[2], "wb") as output:
      writer.write(output)
  PY
  ```

## Alphabet (Google)

- File: `google/2024-q4-earnings-transcript.pdf`
- Direct asset URL: https://s206.q4cdn.com/479360582/files/doc_financials/2024/q4/2024-q4-earnings-transcript.pdf
- Source landing page: https://abc.xyz/investor/events/event-details/2025/2024-Q4-Earnings-Call/
- Publisher: Alphabet Inc.
- Period: Q4 and FY2024, periods ended 2024-12-31
- Event/publication date: 2025-02-04
- Document type: Earnings-call transcript
- Retrieval date: 2026-07-09
- File size: 321,834 bytes
- SHA-256: `57b929e8b34fc71bb19a6774e357f3112954c691f5f4e882e36caa89a2ba3f8a`

## Meta

- File: `meta/META-Q4-2024-Prepared-Remarks.pdf`
- Direct asset URL: https://s21.q4cdn.com/399680738/files/doc_financials/2024/q4/META-Q4-2024-Prepared-Remarks.pdf
- Source landing page: https://investor.atmeta.com/investor-events/event-details/2025/Q4-2024-Earnings-Call/
- Publisher: Meta Platforms, Inc.
- Period: Q4 and FY2024, periods ended 2024-12-31
- Event/publication date: 2025-01-29
- Document type: Earnings-call prepared remarks
- Retrieval date: 2026-07-09
- File size: 67,505 bytes
- SHA-256: `be2a33804cfbe7ec4818e7cf2e3b5f08febaa36a14a0fd6ad23230c9c4ef2f42`

## Amazon

- File: `amazon/AMZN-Q4-2024-Earnings-Release.pdf`
- Direct asset URL: https://s2.q4cdn.com/299287126/files/doc_financials/2024/q4/AMZN-Q4-2024-Earnings-Release.pdf
- Source landing page: https://ir.aboutamazon.com/news-release/news-release-details/2025/Amazon-com-Announces-Fourth-Quarter-Results/
- Publisher: Amazon.com, Inc.
- Period: Q4 and FY2024, periods ended 2024-12-31
- Publication date: 2025-02-06
- Document type: Earnings release and financial tables
- Retrieval date: 2026-07-09
- File size: 358,812 bytes
- SHA-256: `0435f36d49cc40373c1ec7d6a8aa77abf83f90f7774b477d69f28b3859e7fe72`

## Rights and redistribution

These documents were publicly available from official publisher-controlled
investor-relations or newsroom sources on the retrieval date. Public
availability does **not** imply an open license. The documents and their
contents remain copyrighted by their respective publishers. Before tracking,
redistributing, publishing, or using them beyond applicable legal exceptions,
users should review the source terms and verify that they have the necessary
rights and permissions. No open-license claim is made for any document in this
corpus.
