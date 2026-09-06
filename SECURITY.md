# Security Policy

## Scope

This repository contains model training code, adapters, and offline exam
tooling. It ships **no network services** and **no secrets**: all model
weights referenced by the manifests are pinned by SHA-256 and downloaded from
their upstream hosts (Hugging Face, GitHub releases).

## Reporting

Report vulnerabilities or safety issues (unsafe model output, dangerous
procedures in the corpus, license violations) via GitHub Security Advisories
("Report a vulnerability" on the Security tab) rather than public issues.

## Model-output safety

Survival guidance can be safety-critical. Outputs are best-effort and the
application layer is responsible for: citation clamping (`e ⊆ evidence`),
displaying best-effort status, medical-escalation prompts, and the risk
acknowledgement flow. Do not bypass these in integration code.

## Supported versions

Only the latest `main`.
