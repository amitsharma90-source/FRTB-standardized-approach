"""
DRC securitisation (non-CTP) sub-branch of the FRTB SA / DRC pipeline.

Authoritative spec: MAR22.33-35 (trading-book overlay) wrapping CRE40-CRE45
(banking-book SEC framework).

Module roles per FRTB SA/CLAUDE.md "Phase 3 - securitisation (non-CTP) rules":
    sec_loader      : read portfolio sec rows + sec config; produce flat per-position frame
    sec_classifier  : pool classification (40.15-17), senior derivation (40.18)
    sec_hierarchy   : approach routing (40.41-47)
    sec_dd          : due-diligence gate (40.31-36)
    sec_irba        : SEC-IRBA (CRE44)
    sec_erba        : SEC-ERBA (CRE42)
    sec_iaa         : SEC-IAA (CRE43) -> SEC-ERBA mapping
    sec_sa          : SEC-SA (CRE41)
    sec_npl         : NPL layer (CRE45)
    sec_caps        : look-through (40.50-51) + aggregated K_P*P (40.52-55)
    sec_crm         : protection decomposition (40.56-65)
    sec_engine      : 10-step orchestrator; enforces per-tranche netting invariant
    sec_qa          : post-run reconciliation
"""
