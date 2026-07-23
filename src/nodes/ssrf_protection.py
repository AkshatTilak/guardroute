"""SSRF (Server-Side Request Forgery) protection utility for GuardRoute nodes.

Enforces security constraints on outbound HTTP network calls (Webhook, APICall, etc.)
by blocking internal/private IP ranges and metadata service targets.
"""

import ipaddress
import socket
import urllib.parse
import logging

logger = logging.getLogger("guardroute.nodes.ssrf_protection")

# Private and non-routable IP networks to block
BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "metadata",
    "kubernetes.default.svc",
}


class SSRFValidationError(ValueError):
    """Raised when an outbound URL violates SSRF security boundaries."""
    pass


def validate_url_for_ssrf(url: str) -> bool:
    """Validates if a URL is safe to call from the backend.

    Raises SSRFValidationError if the host resolves to a private/reserved IP or blocked domain.
    """
    if not url:
        raise SSRFValidationError("URL cannot be empty.")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFValidationError(f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFValidationError("Invalid URL host.")

    hostname_lower = hostname.lower()
    if hostname_lower in BLOCKED_HOSTNAMES:
        raise SSRFValidationError(f"Access to hostname '{hostname}' is blocked due to security policy.")

    # Check if direct IP
    try:
        ip_obj = ipaddress.ip_address(hostname)
        for net in BLOCKED_NETWORKS:
            if ip_obj in net:
                raise SSRFValidationError(f"Access to IP address '{hostname}' is blocked (private/reserved subnet).")
        return True
    except ValueError:
        pass  # Hostname is a domain name, resolve it

    # Resolve domain to IP address
    try:
        resolved_ips = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        for res in resolved_ips:
            sockaddr = res[4]
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            for net in BLOCKED_NETWORKS:
                if ip_obj in net:
                    raise SSRFValidationError(
                        f"Host '{hostname}' resolved to blocked IP '{ip_str}' (private/reserved network)."
                    )
    except socket.gaierror as e:
        raise SSRFValidationError(f"Failed to resolve host '{hostname}': {e}")
    except SSRFValidationError:
        raise
    except Exception as e:
        raise SSRFValidationError(f"DNS resolution check failed for '{hostname}': {e}")

    return True
