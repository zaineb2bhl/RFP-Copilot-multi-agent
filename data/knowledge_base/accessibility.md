# Methodology: accessibility and ADA compliance

> Synthetic data.

## Standard
Every website we build conforms to WCAG 2.1 Level AA, the standard required for state and local government websites under the U.S. Department of Justice ADA Title II web accessibility rule. Since 2025 we also design and test new sites against WCAG 2.2 Level AA.

## No overlays
We do not use accessibility overlays or third-party "accessibility widgets". Accessibility is built into the design system, templates and code.

## How we ensure conformance
- Accessible design system: color contrast, focus states, heading structure and keyboard navigation are checked at the design stage.
- Automated testing on every release with axe-core in our build pipeline.
- Manual testing of key pages and templates with keyboard only and with the NVDA, JAWS and VoiceOver screen readers.
- An accessibility conformance report (VPAT 2.5, WCAG edition) for Northbridge CivicCMS, updated every year.

## Keeping content accessible after launch
- The CivicCMS editor blocks or warns about common content errors (missing alternative text, skipped heading levels, low contrast).
- Monthly automated scans of the whole site with a dashboard of issues per page.
- PDF accessibility checks on upload, with guidance for editors; PDF remediation of existing documents is available as an optional service through our remediation partner.
- Accessibility training is part of every editor training session.

## Documentation
We publish an accessibility statement on each client site with a contact for reporting barriers, and we provide a written test report before launch.
