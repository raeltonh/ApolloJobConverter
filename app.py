import zipfile
import io
import os
import uuid
import re
import json
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageChops
import numpy as np
import warnings

# PDF generation imports
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- SYSTEM CONFIGURATION ---
# Increase pixel limit for large textile print files
Image.MAX_IMAGE_PIXELS = 500_000_000 
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# OCR Configuration (Optional, handles Mac/Linux differences)
HAS_OCR = False
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    pass

# --- VISUAL STYLES ---
def inject_corporate_styles():
    st.markdown("""
    <style>
    :root { --corp-blue: #8ab4f8; --corp-bg: #0d1117; --corp-panel: #161b22; --corp-gray: #c9d1d9; }
    body, .main, .block-container { background-color: var(--corp-bg) !important; color: var(--corp-gray) !important; }
    h1, h2, h3, h5 { color: var(--corp-blue) !important; }
    .stTextInput input { font-family: monospace; font-weight: bold; color: #8ab4f8; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; color: #8ab4f8; }
    </style>
    """, unsafe_allow_html=True)

# --- CHANNEL MAPPING (ATLAS -> APOLLO) ---
ATLAS_TO_APOLLO_CHANNEL_MAP = {
    "C": "C", "M": "M", "Y": "Y", "K": "K", 
    "R": "R", "G": "G", "W": "W",
    "Iw": "Fw", "Ic": "Fc", # Intensifier -> Fixation
    "Qw": "Fw", "Qc": "Fc", # Q.fix -> Fixation
    "Varnish": "Q.fix"
}

# --- PROCESSING FUNCTIONS ---

def extract_guid_from_image(image_bytes: bytes) -> str:
    """Attempts to extract GUID from a screenshot using OCR."""
    if not HAS_OCR: return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert('L')
            text = pytesseract.image_to_string(img)
            # Regex to find UUID format (xxxxxxxx-xxxx...)
            uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
            match = re.search(uuid_pattern, text)
            return match.group(0) if match else None
    except Exception: return None

def process_channel_image(image_bytes: bytes, filename: str, spray_percentage: int, force_dpi: int = 0) -> Tuple[Optional[bytes], str, str, Optional[Image.Image], dict]:
    """
    Processes the image, converts channels, and extracts METADATA (DPI/Size).
    Returns: bytes, base_name, new_suffix, thumbnail, metadata
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            name_stem = Path(filename).stem
            if "_" in name_stem:
                base_name, original_suffix = name_stem.rsplit('_', 1)
            else: return None, "", "", None, {}

            new_suffix = ATLAS_TO_APOLLO_CHANNEL_MAP.get(original_suffix)
            if not new_suffix:
                # Manual fallback for uppercase
                s_upper = original_suffix.upper()
                if s_upper == "IW": new_suffix = "Fw"
                elif s_upper == "IC": new_suffix = "Fc"
                elif s_upper == "QW": new_suffix = "Fw"
                elif s_upper == "QC": new_suffix = "Fc"
                else: return None, base_name, "", None, {}

            # --- SMART METADATA READING ---
            width_px, height_px = img.size
            
            # If manual DPI is forced, use it. Otherwise, auto-detect.
            if force_dpi > 0:
                dpi_x = dpi_y = force_dpi
            else:
                dpi = img.info.get('dpi', (600, 600))
                if isinstance(dpi, tuple): dpi_x, dpi_y = dpi
                else: dpi_x = dpi_y = dpi
            
            dpi_x = int(dpi_x) if dpi_x else 600
            dpi_y = int(dpi_y) if dpi_y else 600

            meta = {
                "width_px": width_px,
                "height_px": height_px,
                "dpi_x": dpi_x,
                "dpi_y": dpi_y
            }
            # ------------------------------

            if img.mode != 'L': img = img.convert('L')
            
            # Apply intensity reduction if it is a fluid channel
            if new_suffix in ["Q.fix", "Fw", "Fc"] and spray_percentage < 100:
                arr = np.array(img, dtype=float) * (spray_percentage / 100.0)
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                img.info['dpi'] = (dpi_x, dpi_y) # Maintain DPI

            # Create Thumbnail
            thumb = img.copy()
            thumb.thumbnail((800, 800))

            # Save
            buffer = io.BytesIO()
            img.save(buffer, format="TIFF", compression="tiff_lzw", dpi=(dpi_x, dpi_y))
            
            return buffer.getvalue(), base_name, new_suffix, thumb, meta
    except Exception: return None, "", "", None, {}

def generate_composite_preview(channels_data: List[dict]) -> Image.Image:
    """Generates a CMYK color composite preview."""
    ch_map = {item['suffix']: item['thumb'] for item in channels_data}
    if not ch_map: return None
    base_img = next(iter(ch_map.values()))
    composite = Image.new('RGB', base_img.size, (255, 255, 255))
    
    def blend(base, ch_img, color):
        if ch_img is None: return base
        if ch_img.size != base.size: ch_img = ch_img.resize(base.size)
        layer = ImageOps.colorize(ch_img, black=color, white="white")
        return ImageChops.multiply(base, layer)

    composite = blend(composite, ch_map.get('C'), "cyan")
    composite = blend(composite, ch_map.get('M'), "magenta")
    composite = blend(composite, ch_map.get('Y'), "yellow")
    composite = blend(composite, ch_map.get('K'), "black")
    return composite

def save_image_to_pdf(pil_image: Image.Image) -> bytes:
    """Saves only the image inside a PDF (clean layout)."""
    buffer = io.BytesIO()
    w, h = pil_image.size
    c = canvas.Canvas(buffer, pagesize=(w, h))
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    c.drawImage(ImageReader(img_buffer), 0, 0, w, h)
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def recursive_update(data, key_target, new_value):
    """Recursively updates a value in the JSON."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == key_target:
                if isinstance(value, (int, float, str)) or key_target == "SpraySettings":
                     if key_target == "SpraySettings" and "Percentage" in value:
                         value["Percentage"] = new_value
            elif isinstance(value, (dict, list)):
                recursive_update(value, key_target, new_value)
    elif isinstance(data, list):
        for item in data: recursive_update(item, key_target, new_value)

def update_kjob_full(kjob_bytes: bytes, new_guid: str, job_name: str, spray_percent: int, pdf_name: str, meta: dict) -> Tuple[str, float, float]:
    """
    Updates ID, Spray, File Names, and DIMENSIONS in the KJOB.
    """
    try:
        content_str = kjob_bytes.decode('utf-8', errors='ignore')
        data = json.loads(content_str)
        
        # 1. IDs and Name
        if "IdentificationDetails" in data:
            if "JobIdentifier" in data["IdentificationDetails"]:
                data["IdentificationDetails"]["JobIdentifier"]["Id"] = new_guid
            # Internal Name = JobName_GUID
            data["IdentificationDetails"]["JobName"] = f"{job_name}_{new_guid}"
        
        # 2. Spray (Update JSON where 'SpraySettings' is found)
        recursive_update(data, "SpraySettings", round(spray_percent / 100.0, 2))

        # 3. DIMENSION CALCULATION (Pixels -> mm)
        # Formula: (Pixels / DPI) * 25.4
        width_mm = round((meta['width_px'] / meta['dpi_x']) * 25.4, 2)
        height_mm = round((meta['height_px'] / meta['dpi_y']) * 25.4, 2)

        # 4. File and Size Updates
        png_name = f"Preview-{new_guid}.png"
        
        if "PrintSectionList" in data and isinstance(data["PrintSectionList"], list):
            for section in data["PrintSectionList"]:
                
                # A) Update Section Size
                if "Size" in section:
                    section["Size"]["Width"] = width_mm
                    section["Size"]["Height"] = height_mm
                if "RawSize" in section:
                    section["RawSize"]["Width"] = width_mm
                    section["RawSize"]["Height"] = height_mm

                # B) Update Images (Preview and PDF)
                if "ImageProperties" in section:
                    props = section["ImageProperties"]
                    
                    # Preview Image
                    if "PreviewImage" in props and props["PreviewImage"]:
                        props["PreviewImage"]["Name"] = png_name
                        if "Location" in props["PreviewImage"] and props["PreviewImage"]["Location"]:
                            base = os.path.dirname(props["PreviewImage"]["Location"])
                            props["PreviewImage"]["Location"] = os.path.join(base, png_name).replace("\\", "\\\\")

                    # PDF (ImagesForPrint)
                    if "ImagesForPrint" in props and isinstance(props["ImagesForPrint"], list):
                        for img_ref in props["ImagesForPrint"]:
                            img_ref["Name"] = pdf_name
                            # Update PDF size as well
                            if "Size" in img_ref:
                                img_ref["Size"]["Width"] = width_mm
                                img_ref["Size"]["Height"] = height_mm
                            if "Location" in img_ref and img_ref["Location"]:
                                base = os.path.dirname(img_ref["Location"])
                                img_ref["Location"] = os.path.join(base, pdf_name).replace("\\", "\\\\")

        return json.dumps(data, indent=2), width_mm, height_mm
    except Exception: return "", 0, 0

def create_apollo_package(atlas_zip_bytes: bytes, kjob_template_bytes: bytes, spray: int, guid: str, force_dpi: int) -> dict:
    output_zip_buffer = io.BytesIO()
    # Result dictionary to show on screen later
    job_result = {"name": "Job", "w_mm": 0, "h_mm": 0, "dpi": 0, "visuals": [], "preview": None}
    
    channels_data = [] 
    reference_meta = None 

    with zipfile.ZipFile(output_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
        # 1. Process TIFFs
        with zipfile.ZipFile(io.BytesIO(atlas_zip_bytes), "r") as zip_in:
            for file_info in zip_in.infolist():
                if file_info.filename.lower().endswith((".tif", ".tiff")) and not file_info.filename.startswith("__MACOSX"):
                    with zip_in.open(file_info) as file: image_data = file.read()
                    
                    # Process image and get metadata
                    processed_data, base_name, new_ch, thumb, meta = process_channel_image(image_data, file_info.filename, spray, force_dpi)
                    
                    if processed_data:
                        job_result["name"] = base_name
                        # Use the first valid file as reference for size/dpi
                        if reference_meta is None and meta:
                            reference_meta = meta
                            job_result["dpi"] = meta["dpi_x"]
                        
                        # Save TIFF to ZIP with GUID
                        new_filename = f"{base_name}_{guid}_{new_ch}.tif"
                        zip_out.writestr(new_filename, processed_data)
                        channels_data.append({"suffix": new_ch, "thumb": thumb})

        # 2. Generate Previews (PDF/PNG)
        if channels_data:
            job_result["visuals"] = channels_data
            comp = generate_composite_preview(channels_data)
            job_result["preview"] = comp
            
            # Save PNG: Preview-{GUID}.png
            png_buf = io.BytesIO()
            comp.save(png_buf, format='PNG')
            zip_out.writestr(f"Preview-{guid}.png", png_buf.getvalue())

            # Save PDF: CleanName.pdf
            pdf_name = f"{job_result['name']}.pdf"
            pdf_bytes = save_image_to_pdf(comp)
            zip_out.writestr(pdf_name, pdf_bytes)

        # 3. Update KJOB
        if kjob_template_bytes and reference_meta:
            new_kjob, w_final, h_final = update_kjob_full(
                kjob_template_bytes, guid, job_result["name"], spray, f"{job_result['name']}.pdf", reference_meta
            )
            if new_kjob:
                # Save KJOB: CleanName_GUID-job.kjob
                kjob_filename = f"{job_result['name']}_{guid}-job.kjob"
                zip_out.writestr(kjob_filename, new_kjob)
                job_result["w_mm"] = w_final
                job_result["h_mm"] = h_final
    
    return output_zip_buffer.getvalue(), job_result

# --- MAIN INTERFACE ---
def main():
    st.set_page_config(page_title="Atlas QPP -> Apollo Converter", layout="wide")
    inject_corporate_styles()

    if "master_guid" not in st.session_state: st.session_state["master_guid"] = str(uuid.uuid4())
    if "final_result" not in st.session_state: st.session_state["final_result"] = None

    st.title("Atlas QPP ➡️ Apollo Converter")
    st.markdown("**Core Features:** Auto-Dimensioning + Channel Correction + Chemical Control")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Input Files")
        st.markdown("##### Upload Atlas QPP ZIP")
        atlas_zip = st.file_uploader("Upload Atlas QPP ZIP", type=["zip"], label_visibility="collapsed")
        st.markdown("##### Upload Template .kjob")
        kjob_template = st.file_uploader("Upload Template .kjob", type=["kjob"], label_visibility="collapsed")
    
    with col2:
        st.subheader("2. Settings")
        
        # --- SMART DPI SELECTOR ---
        st.markdown("##### 📏 Image Resolution Strategy")
        dpi_option = st.radio(
            "How to detect resolution?",
            ["Auto-Detect (Recommended)", "Force Manual DPI"],
            horizontal=True
        )
        
        force_dpi = 0
        if dpi_option == "Force Manual DPI":
            force_dpi = st.number_input("Enter DPI (ex: 600 or 1200):", min_value=72, max_value=2400, value=600)
            st.warning("⚠️ Warning: Forcing incorrect DPI will result in wrong print size!")
        else:
            st.info("ℹ️ Using internal TIFF metadata. Safest option.")

        st.markdown("##### Chemical Intensity (Spray) %")
        spray = st.slider("Chemical Intensity (Spray) %", 0, 100, 100, label_visibility="collapsed")
        st.caption(f"KJOB Value: **{spray/100.0}**")
        
        st.markdown("#### 🔑 GUID (Identifier)")
        st.markdown("##### Active GUID:")
        guid_final = st.text_input("Active GUID:", value=st.session_state["master_guid"], key="input_guid_display", label_visibility="collapsed")
        st.session_state["master_guid"] = guid_final

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎲 Generate Random"):
                st.session_state["master_guid"] = str(uuid.uuid4())
                st.rerun()
        with c2:
            if HAS_OCR:
                uploaded_print = st.file_uploader("📷 OCR Scan (GUID)", type=["png", "jpg"])
                if uploaded_print:
                    with st.spinner("Scanning image..."):
                        found = extract_guid_from_image(uploaded_print.getvalue())
                        if found:
                            st.session_state["master_guid"] = found
                            st.rerun()

    st.markdown("---")

    if st.button("🚀 Process Job Package", type="primary"):
        if atlas_zip and kjob_template and st.session_state["master_guid"]:
            try:
                with st.spinner("Analyzing DPI, Calculating Dimensions & Generating Package..."):
                    zip_data, info = create_apollo_package(
                        atlas_zip.getvalue(),
                        kjob_template.getvalue(),
                        spray,
                        st.session_state["master_guid"],
                        force_dpi
                    )
                    st.session_state["final_result"] = (zip_data, info)
                
                st.success("Conversion Successful! Check technical data below.")
                
            except Exception as e:
                st.error(f"Critical Error: {e}")
        else:
            st.warning("Missing files or GUID.")

    # --- POST-PROCESSING REPORT ---
    if st.session_state["final_result"]:
        zip_file, data = st.session_state["final_result"]
        
        st.markdown("### 📋 Generated Job Technical Report")
        
        # Engineering Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detected Resolution", f"{data['dpi']} DPI")
        m2.metric("Final Width", f"{data['w_mm']} mm")
        m3.metric("Final Height", f"{data['h_mm']} mm")
        m4.metric("Spray Setting", f"{spray}%")
        
        st.download_button(
            label="⬇ Download Ready Apollo Package (.zip)",
            data=zip_file,
            file_name=f"Apollo_{st.session_state['master_guid']}.zip",
            mime="application/zip"
        )

        st.markdown("---")
        
        # Visualization (Machine Preview + Separate Channels)
        c1, c2 = st.columns([1, 2])
        if data["preview"]:
            with c1:
                st.image(data["preview"], caption=f"Preview-{st.session_state['master_guid']}.png (Machine Screen)")
            with c2:
                st.info("Composite visualization (Simulated CMYK). The generated PDF contains this same image.")

        st.subheader("🎨 Individual Channels (Verification)")
        vis_data = data["visuals"]
        # Sort channels
        order = {"C":1, "M":2, "Y":3, "K":4, "R":5, "G":6, "W":7, "Fw":8, "Fc":9, "Q.fix":10}
        vis_data.sort(key=lambda x: order.get(x["suffix"], 99))
        
        cols = st.columns(6)
        for idx, channel in enumerate(vis_data):
            with cols[idx % 6]:
                st.image(channel['thumb'], use_column_width=True)
                st.caption(f"**{channel['suffix']}**")

if __name__ == "__main__":
    main()