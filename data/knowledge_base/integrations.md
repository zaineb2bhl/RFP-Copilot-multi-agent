# Methodology: integrations with third-party systems

> Synthetic data.

## Approach
Local governments rely on specialized systems for payments, permits, recreation and records. Our website does not replace them: it gives residents one clear entry point and connects to each system in the most reliable way available:
1. **Deep links and single entry points**: clear buttons and service pages that send residents to the right screen of the third-party system.
2. **Embedding**: embedding the vendor's widgets or pages (for example registration catalogs, maps or payment forms) when the vendor supports it securely.
3. **Data feeds and APIs**: importing data through the vendor's API, RSS or iCal feeds, for example events into the community calendar or meeting agendas into meeting pages.
4. **Single sign-on** for staff through SAML 2.0 or OpenID Connect.

During discovery we inventory every system the city uses, check what each vendor supports, and agree on the integration method before design starts.

## Integration experience
Systems we have integrated on past municipal projects include:
- online payment portals for utility bills, taxes and fees;
- parks and recreation registration systems (activity catalogs and online registration);
- agenda and meeting management systems (agendas, minutes and video);
- published codes of ordinances from code publishers (linking to the current code, by chapter);
- Esri ArcGIS maps and dashboards;
- citizen request (311-style) applications;
- social media feeds.

## Limits
When a vendor offers no API or embedding, we integrate through deep links and consistent design. We confirm feasibility for each named system with the vendor during discovery.
