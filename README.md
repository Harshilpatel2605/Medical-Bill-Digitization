# Medical Bills Digitization

> **An AI-powered auditor that instantly converts unstructured scanned documents into verified, structured data while simultaneously screening for potential fraud.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![AI Model](https://img.shields.io/badge/AI-Forensic_Audit-red)

## Overview

This solution is a specialized API designed to automate the review and digitization of complex medical invoices. Unlike standard OCR tools that simply scrape text, this engine utilizes domain-specific logic to understand the unique structure of medical bills. It acts as an automated auditor, seamlessly processing images and PDFs to extract granular data while performing deep forensic analysis to ensure **Payment Accuracy**.

## Key Capabilities

### 1. Granular Line-Item Extraction
The engine moves beyond full-text reading to understand the row-by-row context of a bill.
* **Itemization:** Identifies and extracts specific charge rows (e.g., "MRI Scan", "Consultation Fee", "Pharmacy - Dolo 650").
* **Attribute Mapping:** Isolates `Description`, `Quantity`, `Unit Rate`, and `Total Amount`, filtering out headers, footers, and non-billing metadata.
* **Multimodal Reading:** Successfully processes mixed-input documents, reading printed computer text and handwritten doctor notes on the same page.

### 2. AI-Native Forensic Fraud Detection
A visual "health check" is performed on the document before data processing to identify tampering.
* **Font Consistency Analysis:** Scans numerical fields to detect digital forgery (e.g., a '9' in '9000' using a different font style or size than surrounding digits).
* **Alteration Detection:** Flags visual anomalies such as whitener marks, digital smudges, or handwritten overrides on top of printed totals.

### 3. Automated Logic & Math Validation
Ensures financial integrity through real-time arithmetic validation.
* **Cross-Verification:** Calculates the sum of all extracted line items vs. the printed "Grand Total."
* **Discrepancy Flagging:** If $\sum (Line Items) \neq Grand Total$, the specific mismatch is flagged to prevent overpayment.

### 4. Intelligent Latency Optimization
* **Payload Optimization:** A pre-processing layer analyzes input quality and automatically optimizes document resolution and compression.
* **High-Volume Ready:** Processes high-resolution mobile scans with the speed of lightweight text files.

### 5. Structured Standardization
Returns universal, standardized data regardless of hospital format (Apollo, Fortis, local clinics).
* **Date Normalization:** All dates converted to `DD-MM-YYYY`.
* **Currency Standardization:** Symbols (₹, $) are stripped for calculation utility.
* **Null Handling:** Missing fields are handled gracefully (e.g., `Tax: 0.0`) for immediate database entry.

---

## API Response Structure

The engine returns a standardized JSON object containing extracted data, validation results, and fraud signals.

```json
[
    
    {
        "document" : "DOCUMENT_URL"
    }, 
    
    {
        "is_success": true,
        "data": {
            "pagewise_line_items": [
                {
                    "page_no": "1",
                    "bill_items": [
                        {
                            "item_name": "Consultation (Dr. Neo Church Tharsis(Diabetologist, General Medicine))",
                            "item_amount": 4000.00,
                            "item_rate": 1000.00,
                            "item_quantity": 4.00
                        },
                        {
                            "item_name": "RENAL FUNCTION TEST (RFT)",
                            "item_amount": 240.00,
                            "item_rate": 240.00,
                            "item_quantity": 1.00
                        },
                        {
                            "item_name": "ELECTROLYTES",
                            "item_amount": 450.00,
                            "item_rate": 450.00,
                            "item_quantity": 1.00
                        },
                        {
                            "item_name": "URINE COMPLETE ANALYSIS",
                            "item_amount": 250.00,
                            "item_rate": 250.00,
                            "item_quantity": 1.00
                        },
                        {
                            "item_name": "GLUCOSE FASTING (FBS)",
                            "item_amount": 50.00,
                            "item_rate": 50.00,
                            "item_quantity": 1.00
                        },
                        {
                            "item_name": "X-Ray",
                            "item_amount": 500.00,
                            "item_rate": 500.00,
                            "item_quantity": 1.00
                        },
                        {
                            "item_name": "Registration",
                            "item_amount": 300.00,
                            "item_rate": 300.00,
                            "item_quantity": 1.00
                        },
                        {
                            "item_name": "Room Ward Charges",
                            "item_amount": 6000.00,
                            "item_rate": 2000.00,
                            "item_quantity": 3.00
                        },
                        {
                            "item_name": "DMO,(Ward)",
                            "item_amount": 1500.00,
                            "item_rate": 500.00,
                            "item_quantity": 3.00
                        },
                        {
                            "item_name": "Nursing Charge (Ward)",
                            "item_amount": 1500.00,
                            "item_rate": 500.00,
                            "item_quantity": 3.00
                        },
                        {
                            "item_name": "Maintenance Charges",
                            "item_amount": 600.00,
                            "item_rate": 200.00,
                            "item_quantity": 3.00
                        },
                        {
                            "item_name": "Nebulization",
                            "item_amount": 1000.00,
                            "item_rate": 100.00,
                            "item_quantity": 10.00
                        }
                    ]
                }
            ],
            "total_item_count": 12,
            "reconciled_amount": 16390.00
        }
    }

    
]

