# Security Policy / 安全响应与漏洞披露政策

## Supported Versions / 支持版本范围

Security fixes and vulnerability responses are prioritized for the following branches and releases of FlagTree:

| Branch / Release | Supported | Notes |
| --- | :---: | --- |
| `main` (upstream tracking) | Yes | Active development branch |
| `feat/f09-pr4-graph-bucketing-profiler-and-partition-planner` / F09 release tags | Yes | FlagOS F09 Dynamic Shape Bucketing & Parallel Runtime release line |
| Legacy `triton_v3.0.x` – `triton_v3.2.x` branches | Best effort | Upgrade to current release line recommended |

---

## Reporting a Vulnerability / 漏洞报告渠道

**Please do NOT report security vulnerabilities through public GitHub issues or pull requests.**

To report a suspected security vulnerability (including compiler/runtime memory corruption, cache poisoning, path traversal in `TRITON_CACHE_DIR` or policy loaders, distributed guard denial-of-service, or accidental credential exposure):

1. **Preferred Channel**: Open a private report via **GitHub Security Advisories** ("Security" tab -> "Report a vulnerability" in this repository).
2. **Direct Maintainer Contact**: If private advisories are unavailable, contact the repository maintainers listed in [MAINTAINERS.md](./MAINTAINERS.md) with the subject prefix `[SECURITY][FlagTree]`.

Please include:
- Affected component/file path and exact commit SHA or release tag.
- Minimal reproduction steps or proof-of-concept script (without destructive payloads).
- Impact assessment (e.g., arbitrary code execution, cache tampering, denial-of-service, information disclosure).
- Any suggested fix or mitigation if known.

---

## Response & Disclosure Timeline / 响应与披露 SLA

| Milestone | Target SLA |
| --- | --- |
| Initial acknowledgement of report | Within **3 business days** |
| Triage, severity confirmation (CVSS), and remediation plan | Within **7 business days** |
| Patch development & private verification (Critical / High) | Target **14–30 days** depending on complexity |
| Coordinated public disclosure & advisory publication | Upon verified fix release (or agreed 90-day disclosure window) |

---

## Security Scope & Operational Hardening / 安全边界与加固建议

- **Policy & Cache Integrity**: Do not load untrusted `FLAGTREE_SHAPE_POLICY_PATH` JSON files or share writable `TRITON_CACHE_DIR` directories across mutually untrusted OS users.
- **Distributed Guard Side-Channel**: When enabling `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1`, bind `torch.distributed` rendezvous endpoints (`MASTER_ADDR`, `file://` store paths) strictly to trusted internal networks or private temporary directories with restricted permissions.
- **Secret Hygiene**: Never commit API keys, private keys, internal cluster credentials, or personal machine paths to source files, test fixtures, or release logs.
