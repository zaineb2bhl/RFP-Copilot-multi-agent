# Past project: CRM and billing integration for Westmere Water

> Synthetic data. Westmere Water is a fictional client.

## Context
Westmere Water operated a legacy billing system and a separate CRM, and agents re-keyed customer data between them.

## Solution
In 2021 Northbridge Digital designed an integration layer with an API gateway and an event-driven middleware (Azure Service Bus). We exposed 22 documented REST APIs (OpenAPI 3 specifications) and synchronised customer, meter and billing data in near real time. Legacy data was migrated in three waves with reconciliation reports signed off by the client after each wave.

## Results
- 1.2 million customer records migrated with a reconciliation accuracy of 99.98%.
- Average call handling time reduced by 21%.
- Integration monitoring with alerting on failed messages and automatic retries.
