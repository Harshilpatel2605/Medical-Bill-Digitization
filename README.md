# Intelligent Medical Bill Digitization Engine 🏥

> **An AI-powered auditor that instantly converts unstructured scanned documents into verified, structured data while simultaneously screening for potential fraud.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![AI Model](https://img.shields.io/badge/AI-Forensic_Audit-red)

## 📋 Overview

This solution is a specialized API designed to automate the review and digitization of complex medical invoices. Unlike standard OCR tools that simply scrape text, this engine utilizes domain-specific logic to understand the unique structure of medical bills. It acts as an automated auditor, seamlessly processing images and PDFs to extract granular data while performing deep forensic analysis to ensure **Payment Accuracy**.

## 🚀 Key Capabilities

### 1. Granular Line-Item Extraction
The engine moves beyond full-text reading to understand the row-by-row context of a bill.
* **Itemization:** Identifies and extracts specific charge rows (e.g., "MRI Scan", "Consultation Fee", "Pharmacy - Dolo 650").
* **Attribute Mapping:** Isolates `Description`, `Quantity`, `Unit Rate`, and `Total Amount`, filtering out headers, footers, and non-billing metadata.
* **Multimodal Reading:** Successfully processes mixed-input documents, reading printed computer text and handwritten doctor notes on the same page.

### 2. AI-Native Forensic Fraud Detection 🕵️‍♂️
A visual "health check" is performed on the document before data processing to identify tampering.
* **Font Consistency Analysis:** Scans numerical fields to detect digital forgery (e.g., a '9' in '9000' using a different font style or size than surrounding digits).
* **Alteration Detection:** Flags visual anomalies such as whitener marks, digital smudges, or handwritten overrides on top of printed totals.

### 3. Automated Logic & Math Validation
Ensures financial integrity through real-time arithmetic validation.
* **Cross-Verification:** Calculates the sum of all extracted line items vs. the printed "Grand Total."
* **Discrepancy Flagging:** If $\sum (Line Items) \neq Grand Total$, the specific mismatch is flagged to prevent overpayment.

### 4. Intelligent Latency Optimization ⚡
* **Payload Optimization:** A pre-processing layer analyzes input quality and automatically optimizes document resolution and compression.
* **High-Volume Ready:** Processes high-resolution mobile scans with the speed of lightweight text files.

### 5. Structured Standardization
Returns universal, standardized data regardless of hospital format (Apollo, Fortis, local clinics).
* **Date Normalization:** All dates converted to `DD-MM-YYYY`.
* **Currency Standardization:** Symbols (₹, $) are stripped for calculation utility.
* **Null Handling:** Missing fields are handled gracefully (e.g., `Tax: 0.0`) for immediate database entry.

---

## 💻 API Response Structure

The engine returns a standardized JSON object containing extracted data, validation results, and fraud signals.

```json
{
  "status": "success",
  "meta": {
    "hospital_name": "City General Hospital",
    "date": "12-10-2023",
    "currency": "INR"
  },
  "line_items": [
    {
      "description": "MRI Brain Scan",
      "quantity": 1,
      "unit_rate": 4000.00,
      "total": 4000.00
    },
    {
      "description": "Consultation Fee",
      "quantity": 1,
      "unit_rate": 500.00,
      "total": 500.00
    }
  ],
  "validation": {
    "calculated_total": 4500.00,
    "printed_total": 4500.00,
    "is_match": true
  },
  "fraud_analysis": {
    "tampering_detected": false,
    "font_consistency_score": 0.99,
    "flags": []
  }
}
