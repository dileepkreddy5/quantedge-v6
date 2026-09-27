"""Company Intelligence — the frozen data contract.

Three layers, append-only. Nothing here is ever UPDATEd or DELETEd; an
amendment is a new ci_raw_evidence row pointing back via supersedes_id, a
better extractor is a new ci_derived row with a new extractor_version.

  ci_raw_evidence   immutable primary source (the filing as retrieved)
  ci_events         OBSERVED — deterministic classification of the source
  ci_derived        DERIVED (what the source explicitly states) and
                    HYPOTHESIS (what it might mean) — versioned, cited

Point-in-time discipline: available_at is when the information became
public (SEC acceptance timestamp), never the economic event_date. Every
downstream consumer reads available_at.
"""

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ci_raw_evidence (
    id             BIGSERIAL PRIMARY KEY,
    source_type    TEXT NOT NULL,            -- 'SEC'
    source_id      TEXT NOT NULL,            -- accession number
    form_type      TEXT NOT NULL,            -- '8-K', '4', '10-K' ...
    cik            TEXT NOT NULL,
    filed_at       TIMESTAMPTZ NOT NULL,     -- SEC acceptance datetime
    retrieved_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_payload    JSONB NOT NULL,
    content_hash   TEXT NOT NULL,
    supersedes_id  BIGINT REFERENCES ci_raw_evidence(id),
    UNIQUE (source_type, source_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_ci_raw_cik ON ci_raw_evidence (cik, filed_at DESC);

CREATE TABLE IF NOT EXISTS ci_events (
    id             BIGSERIAL PRIMARY KEY,
    company_id     TEXT NOT NULL,            -- CIK
    ticker         TEXT NOT NULL,
    event_date     DATE,                     -- economic date (reportDate)
    available_at   TIMESTAMPTZ NOT NULL,     -- = filed_at; the PIT timestamp
    retrieved_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    evidence_id    BIGINT NOT NULL REFERENCES ci_raw_evidence(id),
    item_code      TEXT,                     -- '1.01', '5.02', 'FORM4'
    event_type     TEXT NOT NULL,
    significance   TEXT NOT NULL CHECK (significance IN ('ROUTINE','RELEVANT','MATERIAL','UNKNOWN')),
    title          TEXT NOT NULL,
    UNIQUE (evidence_id, item_code)
);
CREATE INDEX IF NOT EXISTS idx_ci_events_ticker ON ci_events (ticker, available_at DESC);

CREATE TABLE IF NOT EXISTS ci_derived (
    id                 BIGSERIAL PRIMARY KEY,
    event_id           BIGINT NOT NULL REFERENCES ci_events(id),
    layer              TEXT NOT NULL CHECK (layer IN ('DERIVED','HYPOTHESIS')),
    extractor_version  TEXT NOT NULL,
    field              TEXT NOT NULL,
    value              JSONB,
    citation           JSONB NOT NULL,       -- {accession, location, document, excerpt_hash?}
    confidence         TEXT NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ('observed','confirmed','hypothesis','needs_validation','invalidated')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ci_derived_event ON ci_derived (event_id);
"""

# Deterministic significance from 8-K item codes. 8.01 is UNKNOWN by
# design: it needs content inspection (Session 3A), not a guess.
ITEM_RULES = {
    "1.01": ("material_agreement",        "MATERIAL", "Entry into a material definitive agreement"),
    "1.02": ("agreement_termination",     "MATERIAL", "Termination of a material definitive agreement"),
    "1.03": ("bankruptcy",                "MATERIAL", "Bankruptcy or receivership"),
    "2.01": ("acquisition_disposition",   "MATERIAL", "Completion of acquisition or disposition of assets"),
    "2.02": ("results",                   "ROUTINE",  "Results of operations and financial condition"),
    "2.03": ("financial_obligation",      "RELEVANT", "Creation of a direct financial obligation"),
    "2.04": ("triggering_event",          "MATERIAL", "Triggering events that accelerate a financial obligation"),
    "2.05": ("exit_costs",                "RELEVANT", "Costs associated with exit or disposal activities"),
    "2.06": ("impairment",                "MATERIAL", "Material impairments"),
    "3.01": ("delisting_notice",          "MATERIAL", "Notice of delisting or failure to satisfy listing rule"),
    "3.02": ("unregistered_equity_sale",  "RELEVANT", "Unregistered sales of equity securities"),
    "3.03": ("security_holder_rights",    "RELEVANT", "Material modification to rights of security holders"),
    "4.01": ("auditor_change",            "RELEVANT", "Changes in registrant's certifying accountant"),
    "4.02": ("non_reliance",              "MATERIAL", "Non-reliance on previously issued financial statements"),
    "5.01": ("change_in_control",         "MATERIAL", "Changes in control of registrant"),
    "5.02": ("officer_change",            "RELEVANT", "Departure or appointment of directors or officers"),
    "5.03": ("bylaw_amendment",           "ROUTINE",  "Amendments to articles or bylaws"),
    "5.07": ("shareholder_vote",          "ROUTINE",  "Submission of matters to a vote of security holders"),
    "7.01": ("reg_fd",                    "ROUTINE",  "Regulation FD disclosure"),
    "8.01": ("other_event",               "UNKNOWN",  "Other events (content inspection required)"),
}
IGNORE_ITEMS = {"9.01"}   # exhibits: companion to another item, never an event itself
