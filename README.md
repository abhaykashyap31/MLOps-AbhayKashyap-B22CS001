# Data Contracts & YAML Validation

**Course:** CSL7120: DLOps
**Topic:** Data Quality, Governance, and Contract Authoring

---

## Overview

This assignment focuses on the design and implementation of **data contracts** as a formal interface between data producers and consumers. The goal is to ensure **schema stability**, **data quality**, **privacy governance**, and **operational reliability** across different domains such as ride-sharing, e-commerce, IoT, and financial systems.
Each scenario demonstrates how data contracts act as **circuit breakers**, preventing bad or unexpected data from propagating downstream and causing failures in machine learning models, dashboards, or fraud detection pipelines.
All contracts are authored in **YAML** following an **Open Data Contract–style structure**, with explicit quality rules and enforcement strategies.

---

## Assignment Objectives

The assignment addresses the following learning outcomes:

* Understand the role of **data contracts** in modern data platforms
* Separate **physical schemas** from **logical, business-friendly interfaces**
* Encode **data quality rules** as executable validations
* Apply **PII tagging** and governance metadata
* Define **SLAs** such as freshness and availability
* Reason about **hard vs soft enforcement (circuit breakers)**
* Produce **YAML-valid, production-style contracts**

---

## Submission Structure

The submission consists of **four (4) independent YAML files**, one per scenario:

```
.
├── rides_contract.yaml        # Scenario 1: Ride-Share (Comprehensive)
├── orders_contract.yaml       # Scenario 2: E-commerce Flash Sale
├── thermostat_contract.yaml   # Scenario 3: IoT Smart Thermostats
├── fintech_contract.yaml      # Scenario 4: Financial Transactions
└── README.md                  # This file
```

Each YAML file is self-contained and can be validated independently.

---

## Scenario Summary

### Scenario 1: Ride-Share Data Contract

* Demonstrates full producer–consumer negotiation
* Logical renaming of cryptic database columns
* Data quality rules for fare, rating, and distance
* Explicit **PII tagging**
* SLA definition for data freshness
* Hard circuit breakers to protect ML pipelines

### Scenario 2: E-commerce Order Stream

* Non-negative order totals
* Enum mapping from status codes to business values
* Rejection of unmapped status codes
* Prevents dashboard crashes during high-volume events

### Scenario 3: IoT Smart Thermostat Fleet

* Range validation for temperature readings
* Battery level sanity checks
* Protects aggregate analytics from faulty sensors

### Scenario 4: FinTech Transaction Log

* Regex-based validation of account identifiers
* Explicit **hard circuit breaker** annotation
* Ensures fraud detection systems do not silently fail

---

## YAML Validation

All YAML files were validated using **yamllint**, an industry-standard YAML linter.

**Validator used:**
[https://www.yamllint.com/](https://www.yamllint.com/)

Validation checks include:

* Proper indentation and structure
* Valid lists and mappings
* Correct string and regex formatting
* No syntax or parsing errors

Only YAML files that pass validation without warnings should be submitted.

---

## Enforcement Strategy

This assignment deliberately uses **hard enforcement** for critical rules where downstream failure would be costly or dangerous (e.g., ML model crashes, fraud detection gaps).

* **Hard enforcement:** Pipeline must block on violation
* **Soft enforcement:** (not used here) Log or warn only

This reflects real-world data platform practices in regulated or high-impact systems.

---

## Design Principles Followed

* **Logical over Physical:** Business-friendly field names are exposed
* **Explicit Contracts:** Expectations are documented and enforceable
* **Fail Fast:** Bad data is stopped as early as possible
* **Governance-Aware:** PII is clearly identified
* **Production-Style YAML:** Readable, structured, and maintainable

## Notes

* No backend or execution framework is assumed.
* The contracts are platform-agnostic and focus on **design correctness**.
* Thresholds and rules were chosen to be **realistic and enforceable** by the producer.

## Author

**Abhay Kashyap**
B22CS001


