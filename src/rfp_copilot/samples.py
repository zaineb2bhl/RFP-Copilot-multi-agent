"""Synthetic sample RFP (fictional buyer) for demos and tests, rendered to PDF with PyMuPDF."""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf

SAMPLE_RFP_TITLE = "RFP-2026-031 City Website Redesign, Hosting and Support"

SAMPLE_RFP_HTML = """
<h1>City of Riverton - Request for Proposal RFP-2026-031</h1>
<h1>City Website Redesign, Hosting and Support</h1>
<p><i>Synthetic document for demonstration purposes. The City of Riverton is fictional.</i></p>

<h2>1. Introduction</h2>
<p>The City of Riverton (population 58,000) invites qualified firms to submit proposals for
the redesign, hosting and ongoing support of the City's official website. The selected firm
will deliver a modern, accessible website that residents can use easily on any device.</p>

<h2>2. Background</h2>
<p>The current website runs on an outdated content management system with about 1,800 pages
and 3,000 documents. Content is managed by 40 staff members across 14 departments, most of
whom have no technical background.</p>

<h2>3. Design and User Experience</h2>
<p>The vendor shall conduct discovery with City staff and residents before design begins.</p>
<p>The vendor shall provide at least two homepage design concepts that reflect the City's
brand.</p>
<p>The website must use a mobile-first, responsive design that works on phones, tablets and
desktops.</p>

<h2>4. Content Management System</h2>
<p>The CMS must allow non-technical staff to update content without HTML knowledge.</p>
<p>The CMS must provide role-based permissions so that each department edits only its own
pages, with an optional approval workflow.</p>
<p>Staff must be able to schedule content to publish and to expire at a set date and time.</p>
<p>The CMS should allow editors to edit content directly on the front end of the site.</p>
<p>Hidden or deleted pages must not continue to appear in search engine results.</p>
<p>The website shall include a single community calendar that all departments can post to,
with filtering by category.</p>
<p>The website shall include a section to post bids and RFPs with addenda and vendor
notifications.</p>
<p>Staff must sign in using single sign-on with the City's Microsoft Entra ID.</p>

<h2>5. Integrations</h2>
<p>The website must provide access to the City's online utility payment portal.</p>
<p>The website should integrate with the City's parks and recreation registration system.</p>
<p>The vendor should provide a native mobile application for iOS and Android.</p>

<h2>6. Accessibility</h2>
<p>The website must conform to WCAG 2.1 Level AA and comply with ADA Title II requirements.</p>
<p>The vendor shall not use accessibility overlay tools or widgets.</p>

<h2>7. Content Migration</h2>
<p>The vendor shall migrate existing content and documents and set up redirects from old
page addresses to preserve search rankings.</p>

<h2>8. Hosting and Security</h2>
<p>The vendor must host the website with at least 99.9% uptime.</p>
<p>The vendor must provide SSL certificates, daily backups and a disaster recovery plan.</p>
<p>The vendor should offer an AI chatbot that answers resident questions in real time.</p>

<h2>9. Training and Support</h2>
<p>The vendor shall provide training for administrators and content editors, with written
documentation.</p>
<p>The vendor must respond to critical issues within 2 hours, 24 hours a day.</p>
<p>The vendor should provide a warranty period after launch.</p>

<h2>10. Qualifications</h2>
<p>Proposers must provide at least three references from municipal website projects completed
in the last five years.</p>
<p>Proposers must carry general liability and cyber liability insurance.</p>

<h2>11. Submission Instructions</h2>
<p>Proposals must be submitted as a single PDF of no more than 25 pages.</p>
<p>The deadline for submission is 20 January 2027 at 2:00 p.m. local time.</p>
<p>Pricing must be submitted separately using the pricing form in Attachment B.</p>
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
