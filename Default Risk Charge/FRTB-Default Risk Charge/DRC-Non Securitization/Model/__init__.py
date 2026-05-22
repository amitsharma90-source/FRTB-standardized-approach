"""
DRC non-securitisation sub-branch of the FRTB SA / DRC pipeline.

Authoritative spec: MAR22.11-32 (gross JTD, netting, buckets, RW, HBR, capital).

Module roles per FRTB SA/CLAUDE.md "Phase 3 - instrument-treatment rules":
    nonsec_loader   : read Sheet 2 (Portfolio_MV_Decomposed) + Combined Holdings,
                      join legs to parent metadata, derive seniority / LGD / RW
    nonsec_jtd      : compute gross JTD per leg (sign convention, LGD, P&L)
    nonsec_netting  : within-obligor MAR22.19 asymmetric netting
    nonsec_engine   : orchestrator: bucket -> RW -> HBR -> capital

Instrument decomposition (callable bond split + equity index option look-through)
is performed UPSTREAM by the sensitivity engine and emitted on Sheet 2 of
FRTB_Sensitivities.xlsx. DRC consumes the already-decomposed legs and does not
re-apply the decomposition rules — see CLAUDE.md → "Phase 3 — instrument-treatment
rules" for the regulatory citations.
"""
