import os
import io
import asyncio
import logging
import json
import re
import time
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
import uvicorn

# --- SETUP & CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalBillExtractor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("🚨 CRITICAL: GEMINI_API_KEY is missing from .env file")
    logger.error("Please:")
    logger.error("1. Get a new API key from https://aistudio.google.com/app/apikey")
    logger.error("2. Replace the placeholder in .env file with your new API key")
    logger.error("3. DO NOT commit .env to git (it's in .gitignore)")
    raise ValueError("GEMINI_API_KEY is missing")

# Initialize the V1 Client
client = genai.Client(api_key=GEMINI_API_KEY)

# ✅ CONFIGURATION FOR BEST BALANCE
# We use 1.5-flash because it allows 15 requests/min. 
# The "Pro" model allows only 2/min (too slow).
MODEL_ID = "gemini-2.5-flash" 

app = FastAPI(title="Medical Bill Extractor (Robust)")

# --- DATA MODELS ---

class LineItem(BaseModel):
    item_name: str = Field(..., description="Full description of service/product.")
    item_amount: float = Field(..., description="Final net amount for this item.")
    item_qty: float = Field(1.0, description="Quantity. Default to 1 if not found.")

class PageExtraction(BaseModel):
    page_type: str = Field(..., description="'Bill Detail', 'Summary', or 'Irrelevant'")
    bill_items: List[LineItem] = Field(default_factory=list)
    page_total: Optional[float] = Field(None, description="Explicit total written on page")

class ExtractionRequest(BaseModel):
    document: str  # Local path

class APIResponse(BaseModel):
    is_success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    token_usage: dict = {}

# --- IMAGE PREPROCESSING (ACCURACY BOOSTER) ---

def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Applies CLAHE and Thresholding to make text pop out from 
    noisy/scanned medical bills.
    """
    try:
        # Convert to numpy
        img = np.array(pil_image)
        
        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Convert to Grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Fixes washed out scans
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Resizing for Latency Optimization
        # Sending 4k images is slow. Resize strict if width > 1500px
        h, w = enhanced.shape
        if w > 1500:
            scale = 1500 / w
            new_h = int(h * scale)
            enhanced = cv2.resize(enhanced, (1500, new_h))
            # Also resize the color version to match
            img = cv2.resize(img, (1500, new_h))

        return Image.fromarray(img) # Return original color (resized) for AI context

    except Exception as e:
        logger.warning(f"Preprocessing failed, using original: {e}")
        return pil_image

def load_document(source: str) -> List[Image.Image]:
    images = []
    try:
        if source.lower().endswith('.pdf'):
            with fitz.open(source) as doc:
                for page in doc:
                    # Matrix=2 for 2x DPI (better OCR accuracy)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
        else:
            # Images
            images.append(Image.open(source))
    except Exception as e:
        raise ValueError(f"Could not load file: {e}")
    return images

# --- ROBUST AI CALLER (ANTI-429) ---

async def process_page_with_retry(page_num: str, image: Image.Image):
    """
    Handles AI calls with automatic retry for Rate Limits (429).
    """
    # Preprocess (CPU bound, run in thread)
    processed_img = await asyncio.to_thread(preprocess_image, image)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    processed_img.save(img_byte_arr, format='JPEG', quality=85)
    img_bytes = img_byte_arr.getvalue()

    prompt = """
    You are an expert medical bill analyzer. Extract ALL line items from this medical bill image.
    
    CRITICAL RULES:
    1. Extract EVERY single line item visible - do not miss any entries
    2. Extract only service/product line items - ignore subtotals, tax, discounts, totals
    3. For each item, extract: name, amount (total for that line), quantity (if shown)
    4. If quantity not shown, use 1.0
    5. If only unit price visible, calculate amount from context
    6. Return ONLY the JSON schema - no additional text
    7. Ensure all amounts are floats, not strings
    
    Required JSON format:
    {
        "page_type": "Bill Detail",
        "bill_items": [
            {
                "item_name": "service/product description",
                "item_amount": 100.50,
                "item_qty": 1.0
            }
        ],
        "page_total": null
    }
    """

    # Retry logic
    max_retries = 3
    base_delay = 5 # seconds

    for attempt in range(max_retries + 1):
        try:
            # Call Google GenAI V1
            response = await client.aio.models.generate_content(
                model=MODEL_ID,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PageExtraction,
                    temperature=0.0 # Strict for data extraction
                )
            )
            
            # Extract Token Usage
            usage = {
                "total": response.usage_metadata.total_token_count if response.usage_metadata else 0
            }
            
            return response.parsed, usage

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str:
                if attempt < max_retries:
                    wait_time = base_delay * (2 ** attempt) # Exponential backoff: 5s, 10s, 20s
                    logger.warning(f"Rate Limit (429) on Page {page_num}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed Page {page_num} after retries: {e}")
                    return None, {"total": 0}
            else:
                logger.error(f"Error Page {page_num}: {type(e).__name__}: {e}")
                logger.debug(f"Full error details: {error_str}")
                return None, {"total": 0}

# --- API ENDPOINT ---

@app.post("/api/v1/hackrx/run", response_model=APIResponse)
async def extract_bill_data(request: ExtractionRequest):
    start_time = time.time()
    
    # 1. Load Images
    try:
        images = await asyncio.to_thread(load_document, request.document)
    except Exception as e:
        return APIResponse(is_success=False, error=str(e))

    if not images:
        return APIResponse(is_success=False, error="No images found")

    logger.info(f"Processing {len(images)} pages from: {os.path.basename(request.document)}")

    # 2. SEQUENTIAL PROCESSING LOOP
    # We purposefully avoid asyncio.gather here to prevent 429 bursts.
    
    extracted_pages = []
    total_bill_amount = 0.0
    total_tokens = 0

    for i, img in enumerate(images):
        page_num = str(i + 1)
        
        # Process
        result, usage = await process_page_with_retry(page_num, img)
        
        if result:
            # Validate Math
            page_sum = sum(item.item_amount for item in result.bill_items)
            
            logger.info(f"Page {page_num}: Found {len(result.bill_items)} items, Total: {page_sum}")
            
            extracted_pages.append({
                "page_number": page_num,
                "type": result.page_type,
                "items": [item.dict() for item in result.bill_items],
                "page_sum": round(page_sum, 2)
            })
            total_bill_amount += page_sum
            total_tokens += usage.get("total", 0)
        else:
            logger.warning(f"Page {page_num}: No data extracted (result was None)")
        
        # 3. SAFETY PAUSE (Critical for Free Tier)
        # If we just finished a page, wait a bit before the next
        if i < len(images) - 1:
            logger.info("Cooling down (2s) for API health...")
            await asyncio.sleep(2) 

    # Final Response Construction
    return APIResponse(
        is_success=True,
        data={
            "pagewise_results": extracted_pages,
            "reconciled_amount": round(total_bill_amount, 2),
            "total_pages": len(images)
        },
        token_usage={"total_tokens": total_tokens}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)