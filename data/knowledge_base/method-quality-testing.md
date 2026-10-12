# Methodology: Quality assurance and testing

> Synthetic data.

## Approach
Quality is built into each sprint: acceptance criteria are written with the product owner, and a story is only done when its automated tests pass.

## Test levels
- Unit and integration tests with a minimum of 80% code coverage on business logic.
- End-to-end tests of key user journeys with Playwright, run on every merge.
- Performance tests with k6 against agreed targets (for example, 95% of pages under 2 seconds at peak load).
- Accessibility tests: automated axe-core checks in CI and manual testing with screen readers.
- Security tests: static analysis, dependency scanning and penetration testing.

## User acceptance testing
We prepare UAT scripts with the client, support UAT sessions, and track defects to closure. Go-live requires no open critical or high defects.
