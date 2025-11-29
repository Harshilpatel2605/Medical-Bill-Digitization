#### best so far...
import time
import os
import io
import re
import json
import logging
import asyncio
import requests
import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
import uvicorn

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash" 

# Concurrency Limit (Prevents 429 Errors on Free Tier)
MAX_CONCURRENT_PAGES = 5 

app = FastAPI(title="Medical Bill Extractor (Final)")

# --- DATA MODELS (With Visual Reasoning & Validation) ---

class LineItem(BaseModel):
    item_name: str = Field(..., description="Exact description of service/product")
    item_amount: float = Field(..., description="Net Amount post discounts")
    item_rate: float = Field(..., description="Unit price/rate. If missing, use amount.")
    item_quantity: float = Field(..., description="Quantity. If missing, use 1.0.")

class PageExtraction(BaseModel):
    page_type: str = Field(..., description="One of: 'Bill Detail', 'Final Bill', 'Pharmacy'")
    visual_reasoning: str = Field(..., description="Brief report on visual corrections (strikethroughs) and semantic checks.") 
    bill_items: List[LineItem] = Field(..., description="List of line items on this page")
    explicit_page_total: Optional[float] = Field(None, description="The final total written on this page.")

class ExtractionRequest(BaseModel):
    document: str = Field(..., description="URL or Local File Path")

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class PageItemResponse(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[LineItem]
    sub_total: float = Field(..., description="Sum of item_amounts on this specific page")
    validation_status: str = Field("OK", description="Indicates if calculated sum matches the page's written total.")

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageItemResponse]
    total_item_count: int
    reconciled_amount: float = Field(..., description="Sum of all individual line items across all pages")

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractionData] = None
    error: Optional[str] = None


# --- HIGH ACCURACY PREPROCESSING (CLAHE) ---
# This is crucial for fixing the "163 -> 43" error
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    try:
        img = np.array(pil_image)
        if img.shape[-1] == 3: 
             img = img[:, :, ::-1].copy() 
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE (Contrast Enhancement) - Makes strikethroughs visible
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        processed = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(processed)

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return pil_image 

def load_document_sync(source: str) -> List[Image.Image]:
    """Blocking loader (will be run in a thread)"""
    logger.info(f"Loading document: {source}")
    images = []
    doc_bytes = None
    content_type = ""

    try:
        if os.path.exists(source):
            with open(source, "rb") as f:
                doc_bytes = f.read()
            if source.lower().endswith('.pdf'):
                content_type = 'application/pdf'
        elif source.startswith("http"):
            drive_pattern = r"drive\.google\.com\/file\/d\/([a-zA-Z0-9_-]+)"
            match = re.search(drive_pattern, source)
            if match:
                file_id = match.group(1)
                source = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = requests.get(source, timeout=30)
            response.raise_for_status()
            doc_bytes = response.content
            content_type = response.headers.get('content-type', '').lower()
        else:
            raise ValueError("Input is neither a valid local path nor a URL.")

        if not doc_bytes:
             raise ValueError("Document is empty.")

        if source.lower().endswith('.pdf') or 'pdf' in content_type or doc_bytes.startswith(b'%PDF'):
            with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
                for page in doc:
                    mat = fitz.Matrix(2, 2) # High DPI for accuracy
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
        else:
            img = Image.open(io.BytesIO(doc_bytes))
            images.append(img)
            
        return images
    except Exception as e:
        raise ValueError(f"Failed to load document: {str(e)}")


# --- ASYNC AI ENGINE (Faster than Threads) ---

async def process_single_page(page_num: str, image: Image.Image, semaphore: asyncio.Semaphore) -> tuple[Optional[PageExtraction], dict]:
    async with semaphore: # Safety Lock
        try:
            # Run CPU heavy preprocessing in a separate thread
            processed_img = await asyncio.to_thread(preprocess_image, image)
            
            img_byte_arr = io.BytesIO()
            processed_img.save(img_byte_arr, format='JPEG', quality=95)
            img_bytes = img_byte_arr.getvalue()

            # VISUAL CHAIN-OF-THOUGHT PROMPT
            prompt = """
            Role: Expert Medical Bill Forensic Auditor.
            Objective: Extract clean, legally accurate line-item data.

            [SYSTEM INSTRUCTION: VISUAL FORENSICS FIRST]
            Perform a "Visual Forensic Scan" and "Semantic Validation".

            ### STEP 1: VISUAL FORENSICS (Strikethrough Rule)
            - **Deletion:** If "1~~63~~", value is "1".
            - **Replacement:** If crossed out and handwritten, use handwriting.
            
            ### STEP 2: DATA EXTRACTION & SEMANTIC VALIDATION
            Extract valid line items.
            *** GUARD AGAINST INTERPRETATION ERRORS ***
            - A number is ONLY an Amount if it falls under a 'Price', 'Rate', or 'Amount' column.
            - Do NOT extract Invoice Dates, Invoice Numbers, or Times as amounts.

            ### STEP 3: MATH VERIFICATION
            - Sum extracted items.
            - Compare with Page Total.
            - **Self-Correction:** If mismatch, re-examine Strikethroughs or Date-vs-Price errors.

            ### OUTPUT FORMAT
            Return ONLY a valid JSON object. 
            Fill 'visual_reasoning' with your analysis.
            """

            # Use Async Client (Non-blocking)
            response = await client.aio.models.generate_content(
                model=MODEL_ID,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PageExtraction,
                    temperature=0.1 
                )
            )
            
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
            
            return response.parsed, usage

        except Exception as e:
            logger.error(f"Page {page_num} Error: {e}")
            return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# --- API ENDPOINT ---

@app.post("/api/v1/hackrx/run", response_model=APIResponse, response_model_exclude_none=True)
async def extract_bill_data(request: ExtractionRequest):
    try:
        # Load Document (Threaded to prevent blocking)
        try:
            images = await asyncio.to_thread(load_document_sync, request.document)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not images:
            raise HTTPException(status_code=400, detail="No readable pages found.")

        # Parallel Execution
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
        tasks = [process_single_page(str(i + 1), img, semaphore) for i, img in enumerate(images)]
        
        logger.info(f"Starting parallel processing for {len(images)} pages...")
        results = await asyncio.gather(*tasks)

        # Aggregate Results
        pagewise_results = []
        total_tokens = {"input": 0, "output": 0, "total": 0}
        global_item_count = 0
        global_reconciled_amount = 0.0

        for i, (extraction, usage) in enumerate(results):
            page_num = str(i + 1)
            
            total_tokens["input"] += usage["input_tokens"]
            total_tokens["output"] += usage["output_tokens"]
            total_tokens["total"] += usage["total_tokens"]

            if extraction and extraction.bill_items:
                page_sub_total = round(sum(item.item_amount for item in extraction.bill_items), 2)
                
                # Validation Logic
                validation_msg = "OK"
                if extraction.explicit_page_total:
                    diff = abs(page_sub_total - extraction.explicit_page_total)
                    if diff > 1.0:
                        validation_msg = f"MISMATCH: Calc {page_sub_total} != Page Total {extraction.explicit_page_total}"
                        logger.warning(f"Page {page_num}: {validation_msg}")

                global_reconciled_amount += page_sub_total
                global_item_count += len(extraction.bill_items)
                
                pagewise_results.append({
                    "page_no": page_num,
                    "page_type": extraction.page_type,
                    "bill_items": extraction.bill_items,
                    "sub_total": page_sub_total,
                    "validation_status": validation_msg
                })

        global_reconciled_amount = round(global_reconciled_amount, 2)

        return APIResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=total_tokens["total"],
                input_tokens=total_tokens["input"],
                output_tokens=total_tokens["output"]
            ),
            data=ExtractionData(
                pagewise_line_items=pagewise_results,
                total_item_count=global_item_count,
                reconciled_amount=global_reconciled_amount
            )
        )

    except HTTPException as he:
        return APIResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            error=he.detail
        )
    except Exception as e:
        logger.exception("Server Error")
        return APIResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            error=f"Internal Server Error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)