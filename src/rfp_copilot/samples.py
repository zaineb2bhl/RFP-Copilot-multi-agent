"""Synthetic sample RFP (fictional buyer) for demos and tests, rendered to PDF with PyMuPDF."""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf

SAMPLE_RFP_TITLE = "RFP-2026-014 Digital Citizen Services Portal"

SAMPLE_RFP_HTML = """
<h1>City of Riverton - Request for Proposal RFP-2026-014</h1>
<h1>Digital Citizen Services Portal</h1>
<p><i>Synthetic document for demonstration purposes. The City of Riverton is fictional.</i></p>

<h2>1. Introduction</h2>
<p>The City of Riverton (population 180,000) invites qualified suppliers to submit proposals
for the design, build, hosting and support of a Digital Citizen Services Portal. The portal
will allow residents to request services, make payments and track their requests online.</p>
<p>The contract will run for an initial period of three years, with an option to extend for
two additional years.</p>

<h2>2. Background</h2>
<p>Today, most of the City's 20 resident services are handled by telephone, email or paper
forms. Service data is stored in a Salesforce CRM that is more than eight years old. The City
wants to increase the share of requests submitted online and reduce processing times.</p>

<h2>3. Scope of Work</h2>
<p>The supplier shall design and deliver a responsive web portal covering at least 12
resident services in the first release, including waste collection requests, parking
permits and complaints.</p>
<p>The supplier shall integrate the portal with the City's existing Salesforce CRM and its
online payment provider through documented APIs.</p>
<p>The supplier shall migrate open service requests and resident accounts from the legacy
systems, with reconciliation reports approved by the City.</p>
<p>The supplier should propose a roadmap for additional services after the first release.</p>

<h2>4. Technical Requirements</h2>
<p>The solution must be hosted in a public cloud region located in the EU or the UK.</p>
<p>Residents must be able to create a single account and sign in securely; City staff must
sign in with single sign-on using the City's Azure AD.</p>
<p>The portal must conform to WCAG 2.1 level AA, confirmed by an independent audit before
go-live.</p>
<p>The platform must achieve at least 99.9% monthly availability.</p>
<p>Pages should load in under 2 seconds for 95% of requests at peak load.</p>
<p>The supplier shall provide a disaster recovery plan with a recovery time objective of no
more than 8 hours.</p>

<h2>5. Security and Data Protection</h2>
<p>The supplier must hold a current ISO/IEC 27001 certification.</p>
<p>All personal data must be encrypted at rest and in transit.</p>
<p>The supplier shall support the City in completing a Data Protection Impact Assessment
before go-live.</p>
<p>An independent penetration test must be carried out before go-live and annually
thereafter.</p>
<p>The supplier must notify the City of any security incident within 24 hours.</p>

<h2>6. Project Management</h2>
<p>The supplier shall use an agile delivery approach with regular demonstrations to the
City's product owner.</p>
<p>The supplier shall nominate a project manager holding a recognised project management
certification such as PMP or PRINCE2.</p>
<p>The supplier shall provide monthly progress reports and attend a monthly steering
committee.</p>

<h2>7. Training and Support</h2>
<p>The supplier shall train City staff and administrators before go-live.</p>
<p>The supplier must provide support during business hours with a response time of no more
than 1 hour for critical incidents.</p>
<p>The supplier should offer a warranty period of at least 3 months after go-live.</p>
<p>The supplier shall provide an exit plan ensuring the transfer of code, data and
documentation at the end of the contract.</p>

<h2>8. Supplier Experience and Team</h2>
<p>Bidders must provide at least two references for similar public-sector portal projects
delivered in the last five years.</p>
<p>Bidders should include CVs of the key team members, including the solution architect and
the accessibility lead.</p>

<h2>9. Submission Instructions</h2>
<p>Proposals must be submitted as a single PDF file of no more than 30 pages, excluding CVs.</p>
<p>The deadline for submission is 15 December 2026 at 12:00 noon local time.</p>
<p>Pricing must be submitted in a separate file using the pricing template in Annex A.</p>
"""

_CSS = """
body { font-family: sans-serif; font-size: 10.5pt; }
h1 { font-size: 17pt; font-weight: bold; }
h2 { font-size: 13.5pt; font-weight: bold; margin-top: 12pt; }
p { margin-bottom: 6pt; }
"""


def write_sample_rfp(path: Path | str) -> Path:
    """Render the synthetic RFP to `path` (A4, with "Page N of M" footers) and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (60, 60, -60, -70)
    story = pymupdf.Story(html=SAMPLE_RFP_HTML, user_css=_CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    with pymupdf.open("pdf", buffer.getvalue()) as doc:
        total = doc.page_count
        for number, page in enumerate(doc, start=1):
            footer = (mediabox.width / 2 - 25, mediabox.height - 35)
            page.insert_text(footer, f"Page {number} of {total}", fontsize=8)
        doc.set_metadata({"title": SAMPLE_RFP_TITLE, "author": "City of Riverton (fictional)"})
        doc.save(path)
    return path
