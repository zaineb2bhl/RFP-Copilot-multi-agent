# Past project: Regional health data platform

> Synthetic data. The Northern Valleys Health Agency is a fictional client.

## Context
The Northern Valleys Health Agency needed a secure platform to consolidate data from 9 hospitals for capacity planning and public reporting.

## Solution
From September 2022 to October 2023 we built a data platform on Azure (Data Lake Storage, Azure Data Factory and Databricks) with role-based access control and pseudonymisation of patient identifiers. All data is encrypted at rest with customer-managed keys and in transit with TLS 1.2 or higher. A Data Protection Impact Assessment was completed with the agency's Data Protection Officer before go-live.

## Results
- 9 hospital feeds ingested daily with automated data-quality checks.
- Reporting time for weekly capacity dashboards reduced from 3 days to 4 hours.
- Zero security incidents since go-live; annual penetration test with no critical findings.

## Team
Solution architecture by Daniel Okafor; security and privacy by Priya Raman.
