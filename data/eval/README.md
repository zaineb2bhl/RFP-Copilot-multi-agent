# Gold data for evaluation

## Requirement gold files (`*.gold.json`)

One file per RFP, named `<document>.gold.json`:

```json
{
  "rfp": "data/rfps/<file>.pdf",
  "description": "Where the RFP comes from (URL, date, buyer)",
  "annotator": "your name",
  "requirements": [
    {"id": "G-001", "text": "The supplier must ...", "type": "mandatory", "source_page": 4}
  ]
}
```

- `rfp` is relative to the repo root. RFP PDFs are not committed, so download them into
  `data/rfps/`.
- `type` is one of:
  - `mandatory`: must, shall, required, or rejection if missing;
  - `desirable`: should, may, preferred, or advantage;
  - `administrative`: submission format, deadlines, forms, signatures, eligibility documents.
- `source_page` is the 1-based page number of the PDF file, not the page number printed on
  the page.

### Annotation guidelines

1. Annotate every obligation or preference that a bidder must satisfy or answer: technical,
   functional, security, service levels, team, experience, and submission rules.
2. Use one item per testable requirement. If a sentence contains two distinct obligations
   ("residents must sign in securely; staff must use SSO"), split it. Keep a list such as
   "including A, B and C" as one item.
3. Copy the wording from the PDF. You may trim leading context ("In addition, ..."), but do not
   paraphrase, because matching uses word overlap.
4. Do not annotate background, evaluation criteria for the buyer's own use, or questions to
   the buyer.
5. Annotate at least 30 to 60 requirements per RFP. Annotating a full section is better than
   sampling across sections, so that recall stays meaningful.

`sample_rfp.gold.json` covers the synthetic sample RFP. Its requirements are known by
construction, and a test checks that every one of them appears on its annotated page.

## Downloaded RFPs

Five public RFPs are already in `data/rfps/` (not committed). Sources, checksums and notes are
in `data/rfps/SOURCES.md`. Copy `gold_template.json.example` to `<pdf name>.gold.json` and
annotate it. Gold files only count once they exist, and `sample_rfp.gold.json` works on its own.

## Which public RFPs to download (to add more)

Pick **3 to 5 English, text-based (not scanned) PDFs of 20 to 80 pages** for IT or digital
services. These match the synthetic knowledge base (portals, cloud, integration,
accessibility, security, support). Good public sources:

- **UK Find a Tender / Contracts Finder**: tender packs for council digital services, such
  as citizen portals, CRM replacement or website redevelopment. These often include a
  "Specification" or "Statement of Requirements" PDF with numbered MUST/SHOULD requirements.
- **US SAM.gov**: the Statement of Work or Performance Work Statement attachments of IT
  service solicitations, such as web modernization or cloud migration.
- **World Bank or UNDP procurement notices**: Terms of Reference or Requests for Proposals
  for information systems or e-government platforms.
- **EU TED**: tenders that publish English technical specifications.

Check that each document can be redistributed for evaluation, or keep it local only
(`data/rfps/` PDFs other than the sample are your choice to commit). Only use public
documents with the Gemini free-tier profile.
