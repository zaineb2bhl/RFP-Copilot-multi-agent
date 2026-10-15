# Methodology: hosting, security and availability

> Synthetic data.

## Infrastructure
Client websites run on Amazon Web Services in U.S. regions only, with servers in two availability zones. A content delivery network (CDN) serves pages and files close to visitors and absorbs traffic peaks, for example during storms or elections.

## Availability
We commit to 99.9% monthly uptime in our hosting service level agreement, measured by independent external monitoring every minute. Service credits apply if the commitment is missed. Our measured average uptime across all client sites in 2025 was 99.98%.

## Security
- TLS (HTTPS) on every site with automatically renewed certificates, at no extra cost.
- Web application firewall and DDoS protection in front of all sites.
- Security patches for the platform and servers applied within 14 days, and critical patches within 48 hours.
- Multi-factor authentication for all Northbridge staff access to hosting systems.
- Annual penetration test of the CivicCMS platform by an independent firm; a summary letter is available to clients.
- Annual SOC 2 Type II audit of our hosting environment.

## Backups and disaster recovery
Full backups every day, kept for 30 days in a separate AWS region. Recovery point objective (RPO) of 24 hours and recovery time objective (RTO) of 4 hours. Restore procedures are tested every quarter.

## Domains
We support .gov domains and assist with DNS configuration and the domain transfer at launch. The client always remains the owner of its domain names and content, and can export all content at any time.
