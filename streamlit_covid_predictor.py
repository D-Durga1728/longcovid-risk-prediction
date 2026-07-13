"""
Long COVID Risk Assessment System
MSc Data Analytics Research Prototype - Dublin City University
Author: Durga Prasad Narsing (A00050350) | Supervisors: Dr Martin Crane & Dr Tai Tan Mai
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Long COVID Risk Assessment - DCU",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=Roboto+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    /* DCU brand typeface Objektiv Mk2, with Inter then Arial (DCU fallback) */
    font-family: 'Objektiv Mk2', 'Inter', Arial, Helvetica, sans-serif !important;
    background: #E0F4FF !important;
    color: #0D1B3E !important;
}

/* ── Hide Streamlit top toolbar / decoration bar (removes the top gap) ── */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#stDecoration { display: none !important; height: 0 !important; }

/* ── Main content padding ── */
.main .block-container {
    padding: 0 2rem 3rem 2rem;
    max-width: 1360px;
    background: #E0F4FF;
}
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMainBlockContainer"] {
    background: #E0F4FF !important;
    padding-top: 0 !important;
}

/* ── Responsive DCU navbar (pure CSS flexbox, fluid clamp() sizing) ──────────
   The whole bar is one HTML flex row. Sticky is applied to the stLayoutWrapper
   (its parent spans the full page height, so sticky has room). Everything scales
   with the viewport via clamp()/vw, and links wrap on very small screens. */
/* The navbar's element-container is the sticky element (its parent - the main
   vertical block - spans the full page height, so sticky has room). Both
   stElementContainer and stLayoutWrapper are covered across Streamlit versions. */
[data-testid="stElementContainer"]:has(.dcu-navbar),
[data-testid="stLayoutWrapper"]:has(.dcu-navbar) {
    position: sticky !important;
    top: 0 !important;
    z-index: 9999 !important;
    background: #E0F4FF !important;
    border-bottom: 2px solid #BFDBFE !important;
    box-shadow: 0 3px 12px rgba(26,60,102,.08) !important;
    /* full-bleed across the whole viewport */
    width: 100vw !important;
    min-width: 100vw !important;
    flex-shrink: 0 !important;
    margin-left: calc(-50vw + 50%) !important;
    padding: clamp(8px, 1.2vw, 16px) clamp(12px, 3vw, 44px) !important;
}
.dcu-navbar {
    display: flex;
    flex-direction: column;       /* two tiers: logo row, then links row */
    gap: clamp(4px, .8vw, 10px);
    width: 100%;
}
/* Row 1: DCU logo · project title · Risk Tool (pushed right) */
.dcu-navbar .nav-row1 {
    display: flex; align-items: center; flex-wrap: wrap;
    gap: clamp(8px, 1.4vw, 18px);
}
.dcu-navbar .nav-row1 .risk-btn { margin-left: auto; }
.dcu-navbar .dcu-logo { display: inline-flex; align-items: center; line-height: 0; flex-shrink: 0; }
.dcu-navbar .dcu-logo img { height: clamp(40px, 4.6vw, 66px); width: auto; display: block; }
/* Project title shown as a small branding block after the DCU logo */
.dcu-navbar .proj-title {
    border-left: 3px solid #FFA700;
    padding-left: clamp(8px, 1vw, 12px); flex-shrink: 0;
    text-decoration: none; cursor: pointer; transition: opacity .15s;
}
.dcu-navbar .proj-title:hover { opacity: .82; }
.dcu-navbar .proj-title .pt1 {
    font-size: clamp(1rem, 1.5vw, 1.35rem); font-weight: 800;
    color: #1A3C66; line-height: 1.12;
}
.dcu-navbar .proj-title .pt2 {
    font-size: clamp(.6rem, .9vw, .8rem); color: #4F868E; margin-top: 2px; line-height: 1.2;
}
/* Row 2: nav links */
.dcu-navbar .dcu-links {
    display: flex; align-items: center; flex-wrap: wrap;
    gap: clamp(.7rem, 1.9vw, 2rem);
}
.dcu-navbar .navlink {
    font-size: clamp(.72rem, 1.05vw, .92rem);
    font-weight: 800; color: #1A3C66;
    text-transform: uppercase; letter-spacing: .03em;
    text-decoration: none; white-space: nowrap;
    padding: 4px 2px; transition: color .15s;
}
.dcu-navbar .navlink:hover { color: #E69500; }
.dcu-navbar .risk-btn {
    flex-shrink: 0;
    background: #FFA700; color: #1A3C66;
    border-radius: 50px; text-decoration: none; white-space: nowrap;
    padding: clamp(6px,.9vw,9px) clamp(12px,1.4vw,20px);
    font-size: clamp(.72rem, 1vw, .9rem); font-weight: 800;
    box-shadow: 0 3px 10px rgba(255,167,0,.35); transition: background .15s;
}
.dcu-navbar .risk-btn:hover { background: #E69500; }
/* Mobile only: stack row 1 vertically so Risk Tool sits left-aligned under the title */
@media (max-width: 640px) {
    .dcu-navbar .nav-row1 { flex-direction: column; align-items: flex-start; gap: 8px; }
    .dcu-navbar .nav-row1 .risk-btn { margin-left: 0 !important; }
}

/* ── Scrollbar - force visible on main content and sidebar ── */
html { overflow-y: scroll !important; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #DBEAFE; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #93C5FD; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3B82F6; }
[data-testid="stAppViewContainer"] { overflow-y: auto !important; }
[data-testid="stMainBlockContainer"] { overflow-y: visible !important; }

/* ── Remove ONLY the unwanted gaps ─────────────────────────────────────────
   Collapse the blank vertical space left by the zero-height helper iframes
   (nav / scroll JavaScript). IMPORTANT: do NOT display:none them - the iframe
   must stay in the DOM so its JS still runs (sticky nav + scroll-to). We only
   zero their height/margin so they take no space. Normal spacing is preserved
   everywhere else. */
[data-testid="stMainBlockContainer"] [data-testid="element-container"]:has(iframe[height="0"]) {
    height: 0 !important; min-height: 0 !important; margin: 0 !important; line-height: 0 !important;
}
[data-testid="stMainBlockContainer"] iframe[height="0"] { height: 0 !important; }
/* Leave room for the sticky navbar when scrolling to a section */
[id^="section-"] { scroll-margin-top: 130px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A3C66 0%, #2E5C92 50%, #1976D2 100%) !important;
    overflow-y: auto !important;
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"]::-webkit-scrollbar { width: 6px; }
[data-testid="stSidebar"]::-webkit-scrollbar-track { background: rgba(255,255,255,0.1); }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.35); border-radius: 4px; }
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #fff !important; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] > label {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    width: 100% !important;
    font-size: .9rem !important;
    font-weight: 500 !important;
    color: #fff !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] > label:hover {
    background: rgba(255,255,255,0.22) !important;
}

/* ── Nav button styles ── */
.nav-sl {
    font-size: .82rem;
    font-weight: 600;
    color: #374151;
    cursor: pointer;
    padding: 6px 11px;
    border-radius: 7px;
    border: none;
    background: transparent;
    transition: background .15s, color .15s;
    white-space: nowrap;
}
.nav-sl:hover { background: #DBEAFE; color: #1D4ED8; }

/* ── Tool-page nav link buttons - scoped to stColumn that CONTAINS .nav-tl ──
   NOTE: st.button() renders as a SIBLING of the .nav-tl div, not inside it.
   So ".nav-tl button" does NOT work. Use :has(.nav-tl) on the parent stColumn. */
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] button,
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] button:hover,
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] button:focus,
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] button:active,
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] button:focus-visible {
    font-size: .9rem !important;
    font-weight: 800 !important;
    color: #1A3C66 !important;            /* DCU Slate Blue, bold uppercase (matches dcu.ie) */
    text-transform: uppercase !important;
    letter-spacing: .03em !important;
    cursor: pointer !important;
    padding: 6px 8px !important;
    border-radius: 7px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    overflow: visible !important;
    min-height: 0 !important;
    height: auto !important;
    width: auto !important;
}
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] button:hover {
    background: #FFF3D6 !important;        /* soft gold tint on hover */
    color: #1A3C66 !important;
}
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] button p,
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] button:hover p {
    color: inherit !important;
    font-size: .9rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: .03em !important;
    white-space: nowrap !important;
}
/* Remove Streamlit wrapper gaps / margins inside nav-tl column area */
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
[data-testid="stColumn"]:has(.nav-tl) [data-testid="element-container"] {
    margin: 0 !important;
    padding: 0 !important;
}
/* Uniform gaps between nav links: shrink each link column to its label width,
   then apply ONE fixed gap between them (edge-to-edge gap is now identical from
   Overview → Contact us). Links start after the logo; leftover space stays on the right. */
[data-testid="stHorizontalBlock"]:has(.nav-tl) {
    gap: 1.7rem !important;
    justify-content: flex-start !important;
    flex-wrap: nowrap !important;
}
[data-testid="stColumn"]:has(.nav-tl) {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}
[data-testid="stColumn"]:has(.nav-tl) [data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
    width: auto !important;
}

/* Project-logo button (right) → opens the assessment page. DCU Burnt Gold pill. */
[data-testid="stColumn"]:has(.proj-logo) [data-testid="stButton"] button {
    background: #FFA700 !important; color: #1A3C66 !important;
    border: none !important; border-radius: 50px !important;
    font-size: .86rem !important; font-weight: 800 !important;
    padding: 8px 16px !important; white-space: nowrap !important;
    box-shadow: 0 3px 10px rgba(255,167,0,.35) !important;
}
[data-testid="stColumn"]:has(.proj-logo) [data-testid="stButton"] button:hover {
    background: #E69500 !important;
}
[data-testid="stColumn"]:has(.proj-logo) [data-testid="stButton"] button p {
    color: #1A3C66 !important; font-weight: 800 !important; white-space: nowrap !important;
}

.nb-home button,
.nb-home button:hover,
.nb-home button:focus,
.nb-home button:active,
.nb-home button:focus-visible {
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 8px 18px !important;
    font-size: .81rem !important;
    font-weight: 800 !important;
    height: auto !important;
    box-shadow: 0 3px 12px rgba(26,60,102,.3) !important;
}
.nb-home button         { background: #374151 !important; }
.nb-home button:hover   { background: #1F2937 !important; }
.nb-home button:active  { background: #111827 !important; }
/* Primary CTA - DCU Burnt Gold accent with Slate Blue text */
.nb-start button,
.nb-start button:hover,
.nb-start button:focus,
.nb-start button:active,
.nb-start button:focus-visible {
    color: #1A3C66 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 8px 18px !important;
    font-size: .81rem !important;
    font-weight: 800 !important;
    height: auto !important;
    box-shadow: 0 3px 12px rgba(255,167,0,.35) !important;
}
.nb-start button         { background: #FFA700 !important; }
.nb-start button:hover   { background: #E69500 !important; }
.nb-start button:active  { background: #FFA700 !important; }
.nb-start button p, .nb-start button:hover p,
.nb-start button:focus p { color: #1A3C66 !important; }
.nb-home button p, .nb-home button:hover p,
.nb-home button:focus p  { color: #fff !important; }
/* Keep nav action buttons on a single line (no wrapping) */
.nb-start button, .nb-home button,
.nb-start button p, .nb-home button p { white-space: nowrap !important; }

/* ── Hero Start-Assessment CTA - DCU Burnt Gold pill ── */
[data-testid="stColumn"]:has(.hero-cta) [data-testid="stButton"] button {
    background: #FFA700 !important; color: #1A3C66 !important;
    border: none !important; border-radius: 50px !important;
    font-size: .9rem !important; font-weight: 800 !important;
    padding: 11px 24px !important; white-space: nowrap !important;
    box-shadow: 0 5px 16px rgba(255,167,0,.40) !important;
}
[data-testid="stColumn"]:has(.hero-cta) [data-testid="stButton"] button:hover {
    background: #E69500 !important; box-shadow: 0 7px 20px rgba(255,167,0,.5) !important;
}
[data-testid="stColumn"]:has(.hero-cta) [data-testid="stButton"] button p {
    color: #1A3C66 !important; font-weight: 800 !important; white-space: nowrap !important;
}

/* ── Contact Us nav button ── */
.nb-contact button {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    border: 2px solid #BFDBFE !important;
    border-radius: 50px !important;
    padding: 6px 14px !important;
    font-size: .81rem !important;
    font-weight: 700 !important;
    height: auto !important;
    box-shadow: none !important;
    transition: all .15s !important;
    white-space: nowrap !important;
}
.nb-contact button p { white-space: nowrap !important; }
.nb-contact button:hover,
.nb-contact button:focus,
.nb-contact button:active {
    background: #DBEAFE !important;
    border-color: #3B82F6 !important;
    color: #1E40AF !important;
}

/* ── Assess Risk button ── */
.sb-run button,
.sb-run button:focus,
.sb-run button:focus-visible {
    background: #1A3C66 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 0 !important;
    font-size: .88rem !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 14px rgba(26,60,102,0.35) !important;
}
.sb-run button:hover  { background: #2E5C92 !important; color: #fff !important; }
.sb-run button:active { background: #1A3C66 !important; color: #fff !important; }
.sb-run button p, .sb-run button:hover p { color: #fff !important; }

/* ── Reset button ── */
.sb-reset button,
.sb-reset button:focus,
.sb-reset button:focus-visible {
    background: #fff !important;
    color: #1A3C66 !important;
    border: 2px solid #BFDBFE !important;
    border-radius: 8px !important;
    padding: 8px 0 !important;
    font-size: .9rem !important;
    font-weight: 700 !important;
}
.sb-reset button:hover  { background: #EFF6FF !important; color: #1A3C66 !important; }
.sb-reset button p, .sb-reset button:hover p { color: #1A3C66 !important; }

/* ── Sample patient buttons (no wrapper class - general stButton in form) ── */
[data-testid="stButton"] button {
    background: #EFF6FF !important;
    color: #1A3C66 !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
[data-testid="stButton"] button:hover {
    background: #DBEAFE !important;
    color: #1A3C66 !important;
    border-color: #93C5FD !important;
}
[data-testid="stButton"] button p { color: #1A3C66 !important; }

/* ── Sample patients - ONE clean table (Level | Threshold | Score | Load) ───
   Everything is wrapped in a single full-width stColumn containing
   .sp-patient-section, so these :has() rules scope tightly to that block. */

/* Vertical spacing between header + rows (positive gap prevents overlap) */
[data-testid="stColumn"]:has(.sp-patient-section) [data-testid="stVerticalBlock"] {
    gap: 5px !important;
}
/* Even horizontal gap between the 4 cells */
[data-testid="stColumn"]:has(.sp-patient-section) [data-testid="stHorizontalBlock"] {
    gap: 8px !important;
    margin: 0 !important;
}

/* Header cells - larger, bolder, clearly separated above the rows */
.sp-th {
    display: flex; align-items: center; justify-content: center;
    min-height: 28px; padding: 0 12px 4px; margin-bottom: 4px;
    font-size: .82rem; font-weight: 800; color: #1A3C66;
    text-transform: uppercase; letter-spacing: .05em; text-align: center;
    border-bottom: 2px solid #BFDBFE;
}
/* Data cells (Level / Threshold / Score) - compact, uniform height, centred */
.sp-cell {
    display: flex; align-items: center; justify-content: center;
    min-height: 32px; padding: 3px 12px; border-radius: 7px;
    border: 1px solid rgba(191,219,254,0.55);
}

/* Uniform Load button - clean navy, matches the form's primary action */
[data-testid="stColumn"]:has(.sp-patient-section) [data-testid="stButton"] button {
    min-height: 32px !important;
    height: 32px !important;
    border-radius: 7px !important;
    font-size: .78rem !important;
    font-weight: 700 !important;
    padding: 0 8px !important;
    background: linear-gradient(135deg,#1A3C66,#2E5C92) !important;
    color: #fff !important;
    border: 1px solid #1A3C66 !important;
    box-shadow: 0 2px 8px rgba(26,60,102,.22) !important;
    width: 100% !important;
}
[data-testid="stColumn"]:has(.sp-patient-section) [data-testid="stButton"] button:hover {
    background: linear-gradient(135deg,#0B357C,#1763CE) !important;
    box-shadow: 0 4px 14px rgba(26,60,102,.32) !important;
}
[data-testid="stColumn"]:has(.sp-patient-section) [data-testid="stButton"] button p {
    color: #fff !important; font-weight: 700 !important;
}

/* ── Selectbox: white bg + dark navy text ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #fff !important;
    border: 1.5px solid #BFDBFE !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: #0D1B3E !important;
    background: #fff !important;
}
.sb-sec {
    background: rgba(255,255,255,0.12);
    border-radius: 7px;
    padding: 6px 11px;
    margin: 12px 0 8px;
    font-size: .71rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65) !important;
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(13,27,62,.09);
    border: 1px solid #BFDBFE;
    margin-bottom: 20px;
}
.card-title {
    font-size: .76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #2E5C92;
    margin-bottom: 14px;
    padding-bottom: 9px;
    border-bottom: 2px solid #DBEAFE;
}

/* ── Section heading ── */
.sh h2 { font-size: 1.6rem; font-weight: 800; color: #0D1B3E; margin-bottom: 6px; }
.sh h2 span { color: #2E5C92; }
.sh p { font-size: .91rem; color: #374151; line-height: 1.7; max-width: 640px; margin-bottom: 18px; }

/* ── Feature grid ── */
.fg { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 26px; }
.fc { background: #fff; border-radius: 14px; padding: 24px 20px; box-shadow: 0 4px 20px rgba(13,27,62,.08); border: 1px solid #BFDBFE; transition: transform .2s, box-shadow .2s; }
.fc:hover { transform: translateY(-4px); box-shadow: 0 10px 36px rgba(13,27,62,.16); }
.fc-icon { font-size: 1.9rem; margin-bottom: 11px; }
.fc-title { font-size: .95rem; font-weight: 700; color: #0D1B3E; margin-bottom: 6px; }
.fc-text { font-size: .81rem; color: #374151; line-height: 1.65; }

/* ── Who can use ── */
.wu { background: #fff; border-radius: 18px; padding: 30px 26px; box-shadow: 0 4px 20px rgba(13,27,62,.08); border: 1px solid #BFDBFE; margin-bottom: 26px; }
.wu-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 13px; margin-top: 16px; }
.wu-card { background: #EFF6FF; border-radius: 11px; padding: 16px; border-left: 4px solid #1D4ED8; transition: transform .2s; }
.wu-card:hover { transform: translateX(4px); }
.wu-icon { font-size: 1.5rem; margin-bottom: 6px; }
.wu-title { font-size: .87rem; font-weight: 700; color: #0D1B3E; margin-bottom: 4px; }
.wu-text { font-size: .78rem; color: #374151; line-height: 1.55; }

/* ── Ethics ── */
.ethics { background: linear-gradient(135deg,#1A3C66,#2E5C92); border-radius: 18px; padding: 30px 34px; margin-bottom: 26px; }
.ethics h3 { font-size: 1rem; font-weight: 800; color: #fff; margin-bottom: 14px; }
.eg { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
.ep { display: flex; gap: 8px; font-size: .81rem; color: rgba(255,255,255,.88); line-height: 1.5; }
.epc { color: #93C5FD; font-weight: 700; flex-shrink: 0; }
.disc { margin-top: 14px; background: rgba(255,255,255,.1); border-left: 3px solid #93C5FD; border-radius: 7px; padding: 10px 14px; font-size: .76rem; color: rgba(255,255,255,.8); line-height: 1.6; }

/* ── Hero ── */
.hero { background: linear-gradient(135deg,#1A3C66 0%,#2E5C92 40%,#1976D2 72%,#42A5F5 100%); border-radius: 20px; padding: 56px 48px; margin: 20px 0 22px; display: grid; grid-template-columns: 1.3fr .7fr; gap: 44px; align-items: center; position: relative; overflow: hidden; animation: fadeUp .6s ease both; }
.hero::before { content: ''; position: absolute; top: -40%; right: -8%; width: 560px; height: 560px; background: radial-gradient(circle,rgba(147,197,253,.18) 0%,transparent 65%); pointer-events: none; }
.hero-badge { display: inline-block; background: #FFA700; border: 1px solid #FFA700; color: #1A3C66; border-radius: 50px; padding: 5px 15px; font-size: .71rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 18px; }
.hero h1 { font-size: 2.8rem; font-weight: 900; color: #fff; line-height: 1.1; margin-bottom: 13px; }
.hero h1 span { color: #93C5FD; }
.hero p { font-size: .93rem; color: rgba(255,255,255,.88); line-height: 1.75; margin-bottom: 24px; max-width: 490px; }
.hero-stats { display: flex; flex-direction: column; gap: 12px; position: relative; z-index: 1; }
.hero-stat { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.24); border-radius: 14px; padding: 18px 22px; transition: transform .2s; }
.hero-stat:hover { transform: translateX(-5px); }
.hero-stat .n { font-size: 1.85rem; font-weight: 900; color: #fff; line-height: 1; }
.hero-stat .l { font-size: .76rem; color: rgba(255,255,255,.8); margin-top: 3px; }

/* ── Stat bar ── */
.stat-bar { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }
.stc { background: #DBEAFE; border: 2px solid #93C5FD; border-radius: 14px; padding: 20px 16px; text-align: center; box-shadow: 0 2px 10px rgba(13,27,62,.09); transition: transform .2s, box-shadow .2s; }
.stc:hover { transform: translateY(-4px); box-shadow: 0 10px 32px rgba(13,27,62,.16); background: #BFDBFE; }
.stc .n { font-size: 2.1rem; font-weight: 900; color: #1A3C66; line-height: 1; }
.stc .l { font-size: .76rem; color: #1E40AF; margin-top: 5px; font-weight: 600; }

/* ── Results ── */
.tool-hdr { background: linear-gradient(135deg,#1A3C66,#2E5C92,#1976D2); border-radius: 16px; padding: 24px 30px; margin: 0 0 18px; display: flex; align-items: center; justify-content: space-between; }
.tool-hdr h2 { color: #fff; font-size: 1.3rem; font-weight: 800; margin: 0 0 3px; }
.tool-hdr p { color: rgba(255,255,255,.75); font-size: .82rem; margin: 0; }
.live-badge { background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3); color: #fff; border-radius: 50px; padding: 6px 16px; font-size: .77rem; font-weight: 600; }

.rrw { background: #fff; border-radius: 14px; padding: 24px; box-shadow: 0 4px 20px rgba(13,27,62,.09); text-align: center; border: 1px solid #BFDBFE; }
.rn { font-size: 4rem; font-weight: 900; line-height: 1; }
.rd { font-size: .97rem; color: #6B7280; }
.rb { display: inline-block; padding: 6px 20px; border-radius: 50px; font-weight: 800; font-size: .88rem; letter-spacing: .06em; text-transform: uppercase; }
.rb-critical { background: #DC2626; color: #fff; }
.rb-high     { background: #EA580C; color: #fff; }
.rb-medium   { background: #D97706; color: #fff; }
.rb-low      { background: #059669; color: #fff; }

.mc { background: #DBEAFE; border-radius: 11px; padding: 15px 17px; border-left: 4px solid #1D4ED8; margin-bottom: 10px; transition: transform .2s; }
.mc:hover { transform: translateY(-2px); }
.ml { font-size: .69rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #1E40AF; margin-bottom: 3px; }
.mv { font-size: 1.8rem; font-weight: 900; color: #1A3C66; line-height: 1; }

.lbl { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: .83rem; font-weight: 600; color: #0D1B3E; }
.lbb { height: 10px; border-radius: 5px; background: #DBEAFE; overflow: hidden; }
.lbf { height: 100%; border-radius: 5px; }

.fr { display: flex; align-items: center; gap: 11px; padding: 8px 0; border-bottom: 1px solid #DBEAFE; }
.fr:last-child { border-bottom: none; }
.fn { flex: 1; font-size: .83rem; font-weight: 600; color: #0D1B3E; }
.fbb { flex: 2; height: 7px; border-radius: 4px; background: #DBEAFE; overflow: hidden; }
.fbf { height: 100%; border-radius: 4px; background: linear-gradient(90deg,#2E5C92,#42A5F5); }
.fv { width: 46px; text-align: right; font-size: .77rem; font-weight: 700; color: #2E5C92; font-family: 'Roboto Mono',monospace; }

.ti { display: flex; gap: 12px; margin-bottom: 15px; }
.td { width: 33px; height: 33px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: .67rem; flex-shrink: 0; color: #fff; }
.tt { font-size: .86rem; font-weight: 700; margin-bottom: 2px; color: #0D1B3E; }
.tx { font-size: .79rem; color: #374151; line-height: 1.55; }

.chip { display: inline-block; background: #DBEAFE; border: 1px solid #93C5FD; color: #1E40AF; border-radius: 20px; padding: 3px 12px; font-size: .76rem; font-weight: 700; margin: 3px; }
.rfb { background: #FFF5F5; border: 2px solid #FCA5A5; border-radius: 11px; padding: 12px 15px; }
.rfb p { margin: 3px 0; font-size: .81rem; color: #7F1D1D; font-weight: 600; }

.tool-welcome { background: #fff; border-radius: 16px; padding: 36px; text-align: center; box-shadow: 0 4px 20px rgba(13,27,62,.09); border: 1px solid #BFDBFE; margin-bottom: 22px; }
.tool-welcome h3 { font-size: 1.35rem; font-weight: 800; color: #0D1B3E; margin-bottom: 9px; }
.tool-welcome p { font-size: .91rem; color: #374151; line-height: 1.7; max-width: 470px; margin: 0 auto; }

/* ── Footer ── */
.footer { background: linear-gradient(135deg,#1A3C66,#2E5C92,#1976D2); border-radius: 18px; padding: 34px 42px; margin-top: 34px; }
.fd { background: rgba(255,255,255,.1); border-left: 3px solid #93C5FD; border-radius: 7px; padding: 10px 14px; font-size: .76rem; color: rgba(255,255,255,.82); line-height: 1.6; margin-bottom: 20px; }
.fg2 { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 34px; margin-bottom: 22px; }
.fb { font-size: 1.1rem; font-weight: 800; color: #fff; margin-bottom: 8px; }
.ft2 { font-size: .81rem; color: rgba(255,255,255,.7); line-height: 1.65; }
.fch { font-size: .71rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: rgba(255,255,255,.42); margin-bottom: 11px; }
.fi { font-size: .8rem; color: rgba(255,255,255,.78); margin-bottom: 6px; display: block; }
.fbot { border-top: 1px solid rgba(255,255,255,.14); padding-top: 15px; display: flex; justify-content: space-between; font-size: .73rem; color: rgba(255,255,255,.42); }

.div { border: none; border-top: 2px solid #BFDBFE; margin: 18px 0; }

/* ── Methodology timeline (step strip) ──────────────────────────────────── */
.mtl { display:flex; align-items:stretch; justify-content:center; gap:6px;
       flex-wrap:wrap; margin:6px 0 26px; }
.mtl-step { flex:1; min-width:150px; max-width:230px; background:#fff;
       border:1px solid #BFDBFE; border-top:4px solid #2E5C92; border-radius:14px;
       padding:16px 14px 14px; text-align:center; position:relative;
       box-shadow:0 3px 14px rgba(13,27,62,.08); transition:transform .2s, box-shadow .2s; }
.mtl-step:hover { transform:translateY(-4px); box-shadow:0 12px 30px rgba(13,27,62,.15); }
.mtl-num { position:absolute; top:-13px; left:50%; transform:translateX(-50%);
       width:26px; height:26px; border-radius:50%; background:#FFA700; color:#1A3C66;
       font-weight:900; font-size:.85rem; display:flex; align-items:center;
       justify-content:center; box-shadow:0 2px 6px rgba(26,60,102,.25); }
.mtl-ic { font-size:1.7rem; margin:8px 0 6px; }
.mtl-t  { font-weight:800; font-size:.92rem; color:#0D1B3E; margin-bottom:5px; }
.mtl-d  { font-size:.74rem; color:#4B5563; line-height:1.5; }
.mtl-arrow { display:flex; align-items:center; color:#93C5FD; font-size:1.5rem;
       font-weight:900; }
@media (max-width:760px){ .mtl-arrow{ transform:rotate(90deg); } .mtl-step{ max-width:none; } }

@keyframes fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }

/* ── Extra motion layer (native CSS, no JS libs) ─────────────────────────── */
@keyframes fadeIn   { from{opacity:0} to{opacity:1} }
@keyframes slideInL { from{opacity:0;transform:translateX(-22px)} to{opacity:1;transform:translateX(0)} }

/* staggered entrance for the hero stat cards */
.hero-stats .hero-stat { animation: slideInL .55s ease both; }
.hero-stats .hero-stat:nth-child(1){ animation-delay:.10s }
.hero-stats .hero-stat:nth-child(2){ animation-delay:.22s }
.hero-stats .hero-stat:nth-child(3){ animation-delay:.34s }

/* staggered entrance for the 4 stat-bar cards */
.stat-bar .stc { animation: fadeUp .55s ease both; }
.stat-bar .stc:nth-child(1){ animation-delay:.05s }
.stat-bar .stc:nth-child(2){ animation-delay:.15s }
.stat-bar .stc:nth-child(3){ animation-delay:.25s }
.stat-bar .stc:nth-child(4){ animation-delay:.35s }

/* content cards rise in + lift on hover */
.card { animation: fadeUp .55s ease both; transition: transform .22s ease, box-shadow .22s ease; }
.card:hover { transform: translateY(-4px); box-shadow: 0 14px 38px rgba(13,27,62,.16); }

/* hero entrance for the big number cards already had hover; smooth it */
.hero-stat { transition: transform .22s ease, background .22s ease; }
.hero-stat:hover { background: rgba(255,255,255,.22); }

/* gold pill buttons: lift + animated sheen on hover */
.dcu-navbar .risk-btn { transition: transform .2s ease, background .2s ease, box-shadow .2s ease; }
.dcu-navbar .risk-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(255,167,0,.45); }
.dcu-navbar .navlink { transition: color .18s ease, transform .18s ease; }
.dcu-navbar .navlink:hover { transform: translateY(-1px); }

/* sample-patient "Load" buttons & links lift slightly */
.sp-cell a, .sp-cell button { transition: transform .18s ease, box-shadow .18s ease; }
.sp-cell a:hover, .sp-cell button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(26,60,102,.18); }

/* bars grow from the left on render (SHAP, sequelae, per-patient factors) */
@keyframes growBar { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.anim-bar, .lbf { transform-origin: left center;
                  animation: growBar .9s cubic-bezier(.22,.61,.36,1) both; }

/* risk-ring sweep: the coloured arc animates from empty up to the score */
@keyframes ringFill { from { stroke-dashoffset: var(--circ); } to { stroke-dashoffset: var(--off); } }
.ring-arc { stroke-dashoffset: var(--off); animation: ringFill 1.1s cubic-bezier(.22,.61,.36,1) forwards; }
/* the score number and band fade/scale in just after the ring starts */
.rrw .rn { animation: fadeUp .7s ease both .25s; }
.rrw .rb { animation: fadeIn .6s ease both .7s; }

/* respect users who prefer reduced motion (accessibility) */
@media (prefers-reduced-motion: reduce) {
  .ring-arc { stroke-dashoffset: var(--off); animation: none !important; }
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

#MainMenu,footer,header { visibility: hidden; }

/* ── Expanders styled as white cards with readable dark text ──────────────── */
[data-testid="stExpander"] details {
    background: #fff !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 14px !important;
    box-shadow: 0 3px 14px rgba(13,27,62,.08) !important;
    margin: 8px 0 18px !important;
}
[data-testid="stExpander"] summary { color: #0D1B3E !important; font-weight: 800 !important; }
[data-testid="stExpander"] summary:hover { color: #2E5C92 !important; }
/* force dark, readable text for everything inside an expander (all expander
   content sits on white, so this is always safe) */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
    color: #1F2937 !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong { color: #0D1B3E !important; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] a { color: #1D4ED8 !important; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] th { color: #0D1B3E !important; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] em { color: #4B5563 !important; }

/* ── Print / Save-as-PDF styling ─────────────────────────────────────────── */
@media print {
  /* keep DCU colours in the printout */
  * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  /* hide app chrome that shouldn't be on a clinical sheet */
  .dcu-navbar, [data-testid="stSidebar"], [data-testid="stToolbar"],
  [data-testid="stHeader"], .no-print, .stButton, iframe { display: none !important; }
  .stApp { background: #fff !important; }
  /* avoid splitting cards across pages */
  .card, .rrw, .mc, figure { break-inside: avoid; page-break-inside: avoid; }
  a[href]::after { content: ""; }   /* don't append URLs */
}

/* ════════════════════════════════════════════════════════════════════════════
   CLINICAL / ACADEMIC THEME OVERRIDE
   A restrained research-tool aesthetic: neutral paper background, flat bordered
   cards, solid (non-gradient) headers, muted accents, and no decorative motion.
   Placed last in the stylesheet so it wins over the vivid defaults above.
   ════════════════════════════════════════════════════════════════════════════ */

/* Page + containers → calm neutral paper, not bright sky-blue */
html, body, [class*="css"],
.main .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMainBlockContainer"] {
    background: #F4F6F8 !important;
    color: #1A2233 !important;
}

/* Navbar → clean white strip with a hairline rule */
[data-testid="stElementContainer"]:has(.dcu-navbar),
[data-testid="stLayoutWrapper"]:has(.dcu-navbar) {
    background: #FFFFFF !important;
    border-bottom: 1px solid #E2E8F0 !important;
    box-shadow: none !important;
}
.dcu-navbar .proj-title { border-left-color: #B8860B !important; }
.dcu-navbar .proj-title .pt2 { color: #5B6675 !important; }

/* CURATED MOTION - keep meaningful entrance/feedback, drop the gimmicks.
   We deliberately KEEP: hero/section fade-in, risk-ring fill, bar growth, and a
   subtle card-hover lift (motion that conveys cause→effect). We DISABLE only the
   decorative loops that read as "demo": the pulsing badge and the floating glow.
   (Accessibility: the prefers-reduced-motion guard further up still wins.) */

/* Subtle, professional hover lift on interactive cards only */
.card:hover, .fc:hover, .mtl-step:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(13,27,62,.08) !important;
}
/* Static blocks shouldn't react to hover */
.stc:hover, .mc:hover, .hero-stat:hover, .wu-card:hover {
    transform: none !important;
    box-shadow: none !important;
}

/* Flat, bordered cards (academic, not glossy) */
.card, .fc, .rrw, .wu, .tool-welcome,
[data-testid="stExpander"] details {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
.card-title {
    color: #1A3C66 !important;
    border-bottom: 1px solid #E2E8F0 !important;
    letter-spacing: .04em !important;
}

/* Solid restrained headers/sections - replace the vivid blue gradients */
.hero, .tool-hdr, .ethics, .footer {
    background: #1A3C66 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
.hero h1 span, .sh h2 span { color: #E8B23A !important; }

/* Sidebar → solid navy instead of a 3-stop blue gradient */
[data-testid="stSidebar"] { background: #1A3C66 !important; }

/* Metric / stat blocks → neutral surfaces with a thin accent, not saturated blue */
.mc {
    background: #F8FAFC !important;
    border-left: 3px solid #1A3C66 !important;
    border-radius: 4px !important;
}
.stc {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 6px !important;
}
.stc .n { color: #1A3C66 !important; }
.stc .l { color: #5B6675 !important; }
.chip {
    background: #F1F5F9 !important;
    border: 1px solid #D7DEE7 !important;
    color: #334155 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
.mtl-step {
    box-shadow: none !important;
    border: 1px solid #E2E8F0 !important;
    border-top: 3px solid #1A3C66 !important;
    border-radius: 6px !important;
}
.mtl-num { background: #1A3C66 !important; color: #fff !important; box-shadow: none !important; }
.mtl-arrow { color: #9AA7B5 !important; }

/* Flatten the gold pills - keep the colour, drop the heavy glow */
.dcu-navbar .risk-btn, .nb-start button,
[data-testid="stColumn"]:has(.hero-cta) [data-testid="stButton"] button,
[data-testid="stColumn"]:has(.proj-logo) [data-testid="stButton"] button {
    box-shadow: none !important;
}

/* "Live Result" → quiet label, not a pulsing badge */
.live-badge {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,.4) !important;
    font-weight: 600 !important;
}

/* Risk band pills + dividers: flatter radius, thinner rules */
.rb { border-radius: 4px !important; }
.div { border-top: 1px solid #E2E8F0 !important; }

/* Hide the large decorative emoji icons (feature cards, who-can-use cards,
   methodology steps) - they read as clip-art in an academic tool. */
.fc-icon, .wu-icon, .mtl-ic { display: none !important; }
.fc-title, .mtl-t { margin-top: 0 !important; }

/* ── Typographic identity ────────────────────────────────────────────────────
   Serif headlines (journal/thesis feel) over a clean sans body. This is the
   single strongest cue that a human designed the page, not a template. */
.hero h1, .sh h2, .tool-hdr h2, .tool-welcome h3,
.fc-title, .wu-title, .mtl-t,
h1, h2, h3,
[data-testid="stExpander"] summary {
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important;
    letter-spacing: -0.01em !important;
    font-weight: 600 !important;
}
.hero h1 { font-weight: 700 !important; letter-spacing: -0.02em !important; line-height: 1.08 !important; }
/* Body, labels, data and small caps stay sans for clarity */
.card-title, .ml, .l, .fch, .hero-badge, .navlink,
.sp-th, .mtl-num, .rb, .chip, .live-badge {
    font-family: 'Inter', Arial, sans-serif !important;
}
/* Numeric displays use tabular figures so columns/scores don't jitter */
.rn, .mv, .stc .n, .hero-stat .n, .fv {
    font-variant-numeric: tabular-nums !important;
}

/* Neutral scrollbar */
::-webkit-scrollbar-track { background: #ECEFF3 !important; }
::-webkit-scrollbar-thumb { background: #C2CAD4 !important; }
::-webkit-scrollbar-thumb:hover { background: #9AA7B5 !important; }
</style>
""", unsafe_allow_html=True)

# Hide sidebar CSS (for landing page)
HIDE_SIDEBAR = """
<style>
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important;}
.main{margin-left:0!important;}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _find_data():
    # Resolve the dataset path without hardcoding it: env var override first,
    # then alongside this script, then the working directory.
    candidates = [
        os.environ.get("COVID_DATA_PATH", ""),          # env var override
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "covid.csv"),
        "covid.csv",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return "covid.csv"   # will surface a clear st.error below if missing

DATA_PATH = _find_data()
# Deployed model uses 9 acute-phase features (intubed & icu dropped - they were
# severity proxies; the deployed calibrated LR in analysis_output/models/ uses these 9).
FEAT_COLS  = ["age","sex","diabetes","hypertension","cardiovascular",
              "pneumonia","obesity","asthma","copd"]

LC_META = {
    "Respiratory":  {"icon":"","syms":["Shortness of breath","Persistent cough","Reduced lung capacity","Chest tightness"],          "specs":("Consultant Pulmonologist","Respiratory Physiotherapist")},
    "Cardiac":      {"icon":"","syms":["Heart palpitations","Chest pain","Elevated heart rate","Blood pressure changes"],            "specs":("Consultant Cardiologist","Cardiac Rehabilitation Specialist")},
    "Neurological": {"icon":"","syms":["Brain fog","Memory difficulties","Headaches","Concentration problems"],                        "specs":("Consultant Neurologist","Cognitive Rehabilitation Specialist")},
    "Systemic":     {"icon":"","syms":["Chronic fatigue","Post-exertional malaise","Sleep disturbances","Muscle weakness"],            "specs":("Consultant in Internal Medicine","Rehabilitation Medicine Specialist")},
    "Metabolic":    {"icon":"","syms":["Blood sugar dysregulation","Weight changes","Hormonal imbalances","Digestive issues"],        "specs":("Consultant Endocrinologist","Clinical Dietitian")},
}
TIMELINES = {
    "LOW":     [("#059669","W1-2","Initial Monitoring","GP review, baseline blood work, confirm COVID recovery."),("#2E5C92","W3-4","Lifestyle Optimisation","Balanced nutrition, graded exercise reintroduction."),("#0891B2","M2-3","Follow-up Assessment","Re-evaluate symptoms, confirm no new organ involvement."),("#7C3AED","M4+","Maintenance","Annual check-up, continued healthy lifestyle.")],
    "MEDIUM":  [("#D97706","W1-2","Urgent GP Review","Full blood panel, ECG baseline, spirometry if respiratory."),("#2E5C92","W3-4","Specialist Referral","Refer to appropriate consultant."),("#DC2626","M2-3","Structured Rehabilitation","Symptom-paced rehabilitation, cognitive support."),("#7C3AED","M4+","Long-term Monitoring","Quarterly reviews for 12 months.")],
    "HIGH":    [("#DC2626","W1","Immediate Clinical Review","Same-week GP. ECG, chest X-ray, full metabolic panel."),("#D97706","W2-3","Multi-Specialist Referral","Cardiology, pulmonology, neurology referrals."),("#7C3AED","W4-8","Intensive Rehabilitation","Multidisciplinary team, medication management."),("#2E5C92","M3+","Ongoing Care Plan","Monthly reviews, sustained follow-up.")],
    "CRITICAL":[("#7C3AED","D1-3","Emergency Review","Urgent hospital admission. Cardiopulmonary evaluation."),("#DC2626","W1-2","Intensive Specialist Care","Cardiology, respiratory, neurology - simultaneous."),("#D97706","W3-8","Comprehensive Rehabilitation","Inpatient or intensive outpatient rehabilitation."),("#2E5C92","M3+","Sustained Long-term Plan","Quarterly reviews, carer support programme.")],
}

def rlevel(s): return "CRITICAL" if s>=70 else "HIGH" if s>=50 else "MEDIUM" if s>=30 else "LOW"
def urgency(l): return {"CRITICAL":"Within 24-48 hours","HIGH":"Within 1 week","MEDIUM":"Within 2 weeks","LOW":"As scheduled"}[l]
def recovery(s, a, pt=None):
    """
    Estimate a recovery-time band from the risk score, age, and comorbidity load.
    `pt` is the full patient dict; if omitted, only score and age are used.
    """
    b = "3-4 months" if s<30 else "4-6 months" if s<50 else "6-9 months" if s<70 else "9-12 months"
    # Older patients with substantial risk tend to take longer.
    if a >= 65 and s >= 50:
        b = b.replace("months", "months (extended)")
    # A heavy comorbidity load also extends the expected timeline.
    if pt is not None:
        comorb_keys = ["diabetes","hypertension","cardiovascular","obesity","asthma","copd","pneumonia"]
        comorb_count = sum(pt.get(k, 0) for k in comorb_keys)
        if comorb_count >= 3 and "(extended)" not in b:
            b += " (extended - high comorbidity burden)"
    return b
def bclr(p): return "#DC2626" if p>=70 else "#EA580C" if p>=50 else "#D97706" if p>=30 else "#059669"

def lc_risks(pt, base):
    """
    Estimate risk for the five Long COVID sequelae categories.

    These are clinically-informed heuristics drawn from the Long COVID risk-factor
    literature, NOT direct outputs of the trained mortality model. Each formula
    blends the base mortality risk with condition-specific factors (e.g. pneumonia
    weights respiratory, cardiovascular weights cardiac). The UI states this clearly
    so the heuristic is never mistaken for a model prediction.
    """
    # Uses only the 9 deployed-model features (no icu/intubed) so the heuristic
    # layer is consistent with the model's actual inputs.
    a,pn,cv,db,hy,ob,as_,cp = (pt["age"],pt["pneumonia"],pt["cardiovascular"],
        pt["diabetes"],pt["hypertension"],pt["obesity"],pt["asthma"],pt["copd"])
    return {
        "Respiratory":  round(min(100, base*.6  + pn*20 + as_*10 + cp*12 + (a>60)*8), 1),
        "Cardiac":      round(min(100, base*.55 + cv*18 + hy*12  + (a>55)*8 + db*8),  1),
        "Neurological": round(min(100, base*.5  + (a>50)*12 + cv*8 + hy*8 + db*6),    1),
        "Systemic":     round(min(100, base*.65 + ob*12 + (a>45)*8 + pn*10 + db*6),   1),
        "Metabolic":    round(min(100, base*.45 + db*20 + ob*14 + hy*10 + (a>50)*6),  1),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")   # cache the dataframe across reruns
def load_dataset(path: str) -> pd.DataFrame:
    """Load and preprocess the COVID dataset; raises a clear error if missing."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(
            f"**Dataset not found:** `{path}`\n\n"
            "Place `covid.csv` next to this script, or set the "
            "`COVID_DATA_PATH` environment variable to the full file path."
        )
        st.stop()
    df["mortality"] = (df["date_died"] != "9999-99-99").astype(int)
    # Match covid_analysis_full.py Phase 4: model is trained on the CONFIRMED
    # COVID subset (covid_res==1 → 220,218 records), not the full dataset.
    # Also drop physiologically invalid ages (matches Phase 2 cleaning).
    df = df[df["covid_res"] == 1].copy()
    df = df[(df["age"] >= 0) & (df["age"] <= 120)].copy()
    for c in FEAT_COLS:
        if c != "age":
            df[c] = (df[c] == 1).astype(int)
    df = df.dropna(subset=FEAT_COLS + ["mortality"])
    return df


@st.cache_resource(show_spinner="Loading prediction model...")
def load_models():
    """
    Load the DEPLOYED, pre-trained pipeline (no live training):
      imputer (median) → scaler (StandardScaler) → calibrated tuned LR.
    Artifacts live in analysis_output/models/ (see STREAMLIT_HANDOFF.md).
    Metrics are the verified values from the advanced analysis / MODEL_CARD.md.
    """
    import joblib
    _base = os.path.dirname(os.path.abspath(__file__))
    _mdir_candidates = [
        os.path.join(_base, "analysis_output", "models"),
        os.path.join("analysis_output", "models"),
    ]
    _mdir = next((d for d in _mdir_candidates if os.path.isdir(d)), _mdir_candidates[0])

    imputer = joblib.load(os.path.join(_mdir, "imputer.joblib"))
    scaler  = joblib.load(os.path.join(_mdir, "scaler.joblib"))
    model   = joblib.load(os.path.join(_mdir, "model_primary_deployed.joblib"))

    # Verified metrics from analysis_output/advanced/. These hardcoded values are
    # FALLBACKS; the block below overrides them by reading the final CSVs so the
    # app always mirrors analysis_output/advanced/ exactly.
    eval_metrics = {
        "lr":  {"auc": 0.8877},
        "rf":  {"auc": 0.8657},
        "gb":  {"auc": 0.8881},
        "ens": {
            "auc":       0.8878,  # primary deployed model AUC
            "ci_lower":  0.8838,
            "ci_upper":  0.8916,
            "accuracy":  85.6,    # implied by sens/spec at threshold 0.26
            "precision": 44.7,    # threshold-0.26 operating point
            "recall":    70.2,    # = sensitivity
            "specificity": 87.8,
            "f1":        54.6,
            "brier":     0.078,   # after isotonic calibration
            "ece":       0.0022,  # after isotonic calibration
        },
        "threshold":  0.26,
        "temporal_auc": 0.8909,
        "n_train":    176174,
        "n_test":     44044,
        "n_total":    220218,
    }

    # ── Override from the FINAL advanced CSVs (source of truth) ───────────────
    _adv = next((d for d in [os.path.join(_base, "analysis_output", "advanced"),
                             os.path.join("analysis_output", "advanced")]
                 if os.path.isdir(d)), None)

    def _rd(_name):
        try:
            return pd.read_csv(os.path.join(_adv, _name)) if _adv else None
        except Exception:
            return None

    _auc = _rd("covid_auc_confidence_intervals.csv")
    if _auc is not None:
        _m = {str(r["model"]).upper(): r for _, r in _auc.iterrows()}
        for _k, _key in [("LR", "lr"), ("RF", "rf"), ("GB", "gb")]:
            if _k in _m:
                eval_metrics[_key]["auc"] = round(float(_m[_k]["auc"]), 4)
        if "PRIMARY" in _m:
            _p = _m["PRIMARY"]
            eval_metrics["ens"]["auc"]      = round(float(_p["auc"]), 4)
            eval_metrics["ens"]["ci_lower"] = round(float(_p["ci_lower"]), 4)
            eval_metrics["ens"]["ci_upper"] = round(float(_p["ci_upper"]), 4)

    _thr = _rd("covid_threshold_optimization.csv")
    if _thr is not None:
        _f1 = _thr[_thr["strategy"].astype(str).str.contains("F1", case=False, na=False)]
        if len(_f1):
            _r = _f1.iloc[0]
            eval_metrics["threshold"]             = round(float(_r["threshold"]), 2)
            eval_metrics["ens"]["recall"]         = round(float(_r["sensitivity"]) * 100, 1)
            eval_metrics["ens"]["specificity"]    = round(float(_r["specificity"]) * 100, 1)
            eval_metrics["ens"]["precision"]      = round(float(_r["precision"]) * 100, 1)
            eval_metrics["ens"]["f1"]             = round(float(_r["f1"]) * 100, 1)
            # accuracy implied at this threshold (prevalence 12.31% in confirmed cohort)
            eval_metrics["ens"]["accuracy"]       = round(
                (float(_r["sensitivity"]) * 0.1231 + float(_r["specificity"]) * 0.8769) * 100, 1)

    _cal = _rd("covid_calibration_improvement.csv")
    if _cal is not None and len(_cal):
        _r = _cal.iloc[0]
        eval_metrics["ens"]["brier"] = round(float(_r["brier_after"]), 3)
        eval_metrics["ens"]["ece"]   = round(float(_r["ece_after"]), 4)

    _tmp = _rd("covid_temporal_validation.csv")
    if _tmp is not None and len(_tmp):
        eval_metrics["temporal_auc"] = round(float(_tmp.iloc[0]["temporal_auc"]), 4)

    return imputer, scaler, model, eval_metrics


@st.cache_data(show_spinner=False)
def load_shap_values():
    """
    Load Tree-SHAP feature importance for the deployed 9-feature model from
    analysis_output/advanced/covid_tree_shap_importance.csv. Returns a list of
    (pretty_label, importance_percent) tuples sorted descending.
    Falls back to the recorded values if the CSV is missing.
    """
    _pretty = {
        "pneumonia": "Pneumonia", "age": "Age", "sex": "Sex",
        "diabetes": "Diabetes", "hypertension": "Hypertension",
        "obesity": "Obesity", "copd": "COPD", "asthma": "Asthma",
        "cardiovascular": "Cardiovascular",
    }
    _fallback = [
        ("Pneumonia", 44.2), ("Age", 32.9), ("Sex", 8.5), ("Diabetes", 5.6),
        ("Hypertension", 4.5), ("Obesity", 3.0), ("COPD", 0.6),
        ("Asthma", 0.3), ("Cardiovascular", 0.3),
    ]
    _candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "analysis_output", "advanced", "covid_tree_shap_importance.csv"),
        os.path.join("analysis_output", "advanced", "covid_tree_shap_importance.csv"),
    ]
    for _p in _candidates:
        if os.path.isfile(_p):
            try:
                _df = pd.read_csv(_p)
                _tot = _df["tree_shap_importance"].sum()
                _df = _df.sort_values("tree_shap_importance", ascending=False)
                return [
                    (_pretty.get(str(r["feature_name"]).lower(),
                                 str(r["feature_name"]).title()),
                     round(float(r["tree_shap_importance"]) / _tot * 100, 1))
                    for _, r in _df.iterrows()
                ]
            except Exception:
                break
    return _fallback


@st.cache_data(show_spinner=False)
def load_odds_ratios():
    """
    Load per-feature Odds Ratios (per +1 SD, 95% CI) from
    analysis_output/advanced/covid_odds_ratios_ci.csv. Returns
    {feature_name: {'or','lo','hi','p','sig'}}. Falls back to MODEL_CARD values.
    """
    _fallback = {
        "pneumonia": {"or": 2.65, "lo": 2.61, "hi": 2.69, "p": 0.0,    "sig": True},
        "age":       {"or": 2.33, "lo": 2.28, "hi": 2.37, "p": 0.0,    "sig": True},
        "diabetes":  {"or": 1.15, "lo": 1.13, "hi": 1.17, "p": 1e-80,  "sig": True},
        "obesity":   {"or": 1.10, "lo": 1.08, "hi": 1.12, "p": 6e-30,  "sig": True},
        "hypertension":{"or":1.09,"lo": 1.07, "hi": 1.10, "p": 2e-24,  "sig": True},
        "copd":      {"or": 1.03, "lo": 1.02, "hi": 1.04, "p": 2e-07,  "sig": True},
        "cardiovascular":{"or":1.00,"lo":0.99,"hi": 1.01, "p": 0.95,   "sig": False},
        "asthma":    {"or": 0.98, "lo": 0.97, "hi": 1.00, "p": 0.09,   "sig": False},
        "sex":       {"or": 0.80, "lo": 0.79, "hi": 0.82, "p": 6e-128, "sig": True},
    }
    _candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "analysis_output", "advanced", "covid_odds_ratios_ci.csv"),
        os.path.join("analysis_output", "advanced", "covid_odds_ratios_ci.csv"),
    ]
    for _p in _candidates:
        if os.path.isfile(_p):
            try:
                _df = pd.read_csv(_p)
                out = {}
                for _, r in _df.iterrows():
                    out[str(r["feature_name"]).lower()] = {
                        "or": float(r["odds_ratio"]),
                        "lo": float(r["or_ci_lower"]),
                        "hi": float(r["or_ci_upper"]),
                        "p":  float(r["p_value"]),
                        "sig": float(r["or_ci_lower"]) > 1.0 or float(r["or_ci_upper"]) < 1.0,
                    }
                return out
            except Exception:
                break
    return _fallback


def predict_risk(imputer, scaler, model, pt):
    """Deployed pipeline: imputer → scaler → calibrated LR. Returns risk % [0,100].
    pt['sex'] must already be encoded Female=1, Male=0 (matches training)."""
    row = [[pt["age"], pt["sex"], pt["diabetes"], pt["hypertension"],
            pt["cardiovascular"], pt["pneumonia"], pt["obesity"],
            pt["asthma"], pt["copd"]]]
    X = pd.DataFrame(row, columns=FEAT_COLS)
    X = imputer.transform(X)
    X = scaler.transform(X)
    return float(model.predict_proba(X)[0, 1]) * 100


def patient_factors(imputer, scaler, pt, top=4):
    """
    Per-patient explanation of the prediction (a SHAP-style local attribution).

    Method (fully defensible, derived from the deployed model's own statistics):
      log-odds contribution_i = ln(OR_i per +1 SD) × z_i
    where z_i is this patient's standardised feature value (from the model's own
    StandardScaler) and OR_i is the per-+1-SD odds ratio from
    covid_odds_ratios_ci.csv. Positive → pushes risk UP, negative → pushes it DOWN.

    Returns (raises, lowers): two lists of (label, signed_contribution) sorted by
    magnitude. This is an interpretable approximation of the model's logit, not a
    Kernel/Tree-SHAP run, and is labelled as such in the UI.
    """
    import math
    _pretty = {"age":"Age","sex":"Sex","diabetes":"Diabetes","hypertension":"Hypertension",
               "cardiovascular":"Cardiovascular","pneumonia":"Pneumonia","obesity":"Obesity",
               "asthma":"Asthma","copd":"COPD"}
    ors = load_odds_ratios()
    row = [[pt["age"], pt["sex"], pt["diabetes"], pt["hypertension"],
            pt["cardiovascular"], pt["pneumonia"], pt["obesity"],
            pt["asthma"], pt["copd"]]]
    X  = pd.DataFrame(row, columns=FEAT_COLS)
    Xz = scaler.transform(imputer.transform(X))[0]   # standardised values (z-scores)

    contribs = []
    for feat, z in zip(FEAT_COLS, Xz):
        o = ors.get(feat)
        if not o:
            continue
        contribs.append((_pretty[feat], float(math.log(o["or"]) * z)))

    raises = sorted([c for c in contribs if c[1] > 0.01], key=lambda x: -x[1])[:top]
    lowers = sorted([c for c in contribs if c[1] < -0.01], key=lambda x: x[1])[:top]
    return raises, lowers


@st.cache_data(show_spinner=False)
def load_nomogram():
    """
    Load the points-based nomogram tables from analysis_output/advanced/:
      covid_nomogram_points.csv      → feature, unit, points
      covid_nomogram_risk_lookup.csv → total_points, predicted_risk_pct
    Returns (points_df, lookup_df) or (None, None) if missing.
    """
    _adv = next((d for d in [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "analysis_output", "advanced"),
                             os.path.join("analysis_output", "advanced")]
                 if os.path.isdir(d)), None)
    if not _adv:
        return None, None
    try:
        nomo   = pd.read_csv(os.path.join(_adv, "covid_nomogram_points.csv"))
        lookup = pd.read_csv(os.path.join(_adv, "covid_nomogram_risk_lookup.csv")
                             ).sort_values("total_points")
        return nomo, lookup
    except Exception:
        return None, None


def nomogram_score(pt):
    """
    Illustrative manual points score for a patient dict (sex encoded Female=1).
    Returns (total_points, risk_pct, points_breakdown) or None if tables missing.
    This is a 'show-your-work' transparency panel - NOT the official prediction.
    """
    nomo, lookup = load_nomogram()
    if nomo is None or lookup is None:
        return None
    pts, breakdown = 0, []
    for r in nomo.itertuples():
        f = str(r.feature)
        if f == "age":
            p = int(round(pt["age"] / 10) * r.points)
        elif f == "sex":
            p = int(r.points) if pt.get("sex") == 1 else 0   # sex=1 → Female
        else:
            p = int(r.points) if pt.get(f) else 0
        if p != 0:
            breakdown.append((f, p))
        pts += p
    risk = float(np.interp(pts, lookup["total_points"], lookup["predicted_risk_pct"]))
    return int(pts), round(risk, 1), breakdown


@st.cache_data(show_spinner=False)
def find_sample_patients(_imp, _sc, _model):
    """
    Automatically select real patients from the training dataset whose
    ensemble risk score falls closest to each target band centre:
      CRITICAL → 82%   HIGH → 60%   MEDIUM → 38%   LOW → 12%
    Underscored model args tell st.cache_data to skip hashing them.
    Cached for the lifetime of the session - runs only once.
    """
    df  = load_dataset(DATA_PATH)
    X   = df[FEAT_COLS]
    Xs  = _sc.transform(_imp.transform(X))

    # Vectorised calibrated probability from the deployed model
    probs = _model.predict_proba(Xs)[:, 1] * 100

    # Targets matched to the calibrated bands (rlevel: ≥70 crit, ≥50 high, ≥30 med)
    targets = {"crit": 80, "high": 58, "med": 38, "low": 10}
    samples, scores = {}, {}
    for key, target in targets.items():
        idx = int(np.argmin(np.abs(probs - target)))
        row = df.iloc[idx]
        samples[key] = {f: int(row[f]) for f in FEAT_COLS}
        scores[key]  = float(probs[idx])
    return samples, scores


# Load the models at startup; a missing artifact shows a friendly UI error
# instead of a raw traceback.
try:
    imp_m, sc_m, model_m, mdl_metrics = load_models()
except Exception as _startup_err:
    st.error(
        f"**Model initialisation failed:** {_startup_err}\n\n"
        "Ensure `analysis_output/models/` (imputer/scaler/model_primary_deployed) "
        "and `covid.csv` are accessible, then rerun the app."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {"page":"landing","patient":None,"score":None}.items():
    if k not in st.session_state: st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# NAV BAR  - everything in ONE row of st.columns
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _dcu_logo_uri(fname="dcu_logo.png"):
    """Return a DCU logo asset as a base64 data URI.
    fname='dcu_logo.png' (slate, for light bg) or 'dcu_logo_white.png' (for dark bg)."""
    import base64
    for _p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", fname),
               os.path.join("assets", fname)]:
        if os.path.isfile(_p):
            with open(_p, "rb") as _f:
                return "data:image/png;base64," + base64.b64encode(_f.read()).decode()
    return ""


@st.cache_data(show_spinner=False)
def _plot_uri(relpath):
    """Return a PNG from analysis_output/visualizations/<relpath> as a base64 data URI
    (empty string if missing). Lets us embed the real analysis plots inline."""
    import base64
    for _b in [os.path.dirname(os.path.abspath(__file__)), "."]:
        _p = os.path.join(_b, "analysis_output", "visualizations", *relpath.split("/"))
        if os.path.isfile(_p):
            with open(_p, "rb") as _f:
                return "data:image/png;base64," + base64.b64encode(_f.read()).decode()
    return ""


def render_nav():
    import streamlit.components.v1 as components
    page = st.session_state.page
    _logo = _dcu_logo_uri()
    _logo_img = (f'<img src="{_logo}" alt="Dublin City University - Home">'
                 if _logo else '<span style="font-size:.85rem;font-weight:800;color:#1A3C66;letter-spacing:.03em;">DCU</span>')

    # Nav links are native anchors → fully responsive, no st.columns.
    # Landing page: links to the landing sections (in-page #id scroll).
    # Tool page: links to the tool's own sections (Sample Patients / Form / Results).
    if page == "landing":
        _links = [
            ("Overview", "section-overview"), ("Performance", "section-performance"),
            ("SHAP", "section-shap"), ("Methodology", "section-methodology"),
            ("Who Can Use", "section-research"), ("Ethics", "section-ethics"),
            ("Contact us", "section-contact"),
        ]
    else:
        _links = [
            ("Overview", "section-overview"),
            ("Sample Patients", "section-samples"),
            ("Patient Assessment Form", "section-form"),
            ("Results", "section-results"),
            ("Contact us", "section-contact"),
        ]
    _links_html = "".join(
        f'<a class="navlink" href="#{_sid}">{_lbl}</a>' for _lbl, _sid in _links)

    # Right-side action: Risk Tool on the landing page, ← Home on the tool page
    # (same gold pill - size/shape/colour identical).
    if page == "tool":
        _action = ('<a class="risk-btn" href="?" target="_self" title="Back to home">← Home</a>')
    else:
        _action = ('<a class="risk-btn" href="?page=tool" target="_self" '
                   'title="Open the assessment tool">Risk Tool</a>')

    # Two-tier navbar (like dcu.ie): row 1 = DCU logo + project title + action;
    # row 2 = the nav links. One flex column → adapts to any screen via clamp()/wrap.
    # The project title is also a clickable link → the assessment tool.
    st.markdown(
        '<div class="dcu-navbar">'
        '<div class="nav-row1">'
        f'<a class="dcu-logo" href="?" target="_self" title="Home">{_logo_img}</a>'
        '<a class="proj-title" href="?page=tool" target="_self" title="Open the assessment tool">'
        '<div class="pt1">Long COVID Risk System</div>'
        '<div class="pt2">COVID-19 mortality-proxy model</div>'
        '</a>'
        f'{_action}'
        '</div>'
        f'<nav class="nav-row2 dcu-links">{_links_html}</nav>'
        '</div>',
        unsafe_allow_html=True)

    # Scroll to the URL hash, re-snapping until the layout settles (the tool page
    # loads progressively, so a one-shot scroll lands too early). Handles both the
    # reload case and in-page #section clicks.
    components.html("""<script>
(function(){
  var win=window.parent, doc=win.document;
  function tgt(){ var h=win.location.hash; return (h&&h.length>1)?doc.getElementById(h.slice(1)):null; }
  // On first load (reload to a #hash): re-snap until the layout settles.
  var last=-1, stable=0, n=0;
  var iv=setInterval(function(){
    n++; var el=tgt();
    if(el){ var top=Math.round(el.getBoundingClientRect().top); el.scrollIntoView({block:'start'});
      if(top===last){stable++;} else {stable=0;last=top;}
      if(stable>=3 || n>30){ clearInterval(iv); }
    }
    if(n>32) clearInterval(iv);
  },150);
  // Click delegation for in-page nav links - always scrolls (even on repeat
  // clicks of the same link) and avoids the native-jump quirk. Bound once.
  if(!win._navClickBound){ win._navClickBound = 1;
    doc.addEventListener('click', function(e){
      var a = e.target.closest ? e.target.closest('.navlink') : null;
      if(!a) return;
      var href = a.getAttribute('href') || '';
      if(href.charAt(0) === '#'){
        e.preventDefault();
        var el = doc.getElementById(href.slice(1));
        if(el) el.scrollIntoView({behavior:'smooth', block:'start'});
      }
    }, true);
  }
})();
</script>""", height=0, scrolling=False)




# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
def render_contact():
    """Contact & Research Team section (shown on both the landing and tool pages)."""
    st.markdown(
        '<div id="section-contact" style="'
        'background:#1A3C66;'
        'border-radius:6px 6px 0 0;padding:48px 40px 40px;margin:40px 0 0;color:#fff;text-align:center;">'
        '<h2 style="font-size:1.65rem;font-weight:800;margin:0 0 8px;color:#fff;">Contact &amp; Research Team</h2>'
        '<p style="font-size:.95rem;color:#BFDBFE;margin:0 auto 32px;max-width:560px;">'
        'Have questions about this tool, the research, or potential collaboration? We\'d love to hear from you.'
        '</p>'
        '<div style="display:flex;flex-wrap:wrap;gap:22px;justify-content:center;">'
        # Card 1: Researcher
        '<div style="background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.22);'
        'border-radius:18px;padding:26px 28px;min-width:230px;flex:1;max-width:290px;text-align:left;">'
        '<div style="font-weight:800;font-size:1.05rem;color:#fff;margin-bottom:3px;">Durga Prasad Narsing</div>'
        '<div style="font-size:.8rem;color:#BFDBFE;margin-bottom:2px;">MSc Data Analytics · A00050350</div>'
        '<div style="font-size:.8rem;color:#BFDBFE;margin-bottom:14px;">Dublin City University</div>'
        '<div style="display:flex;flex-direction:column;gap:7px;">'
        '<a href="mailto:durgaprasad.narsing2@mail.dcu.ie" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">durgaprasad.narsing2@mail.dcu.ie</a>'
        '<a href="https://github.com/D-Durga1728" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">github.com/D-Durga1728</a>'
        '<a href="https://www.linkedin.com/in/durga-prasad-narsing-155394281/" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">LinkedIn Profile</a>'
        '</div></div>'
        # Card 2: Supervisor
        '<div style="background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.22);'
        'border-radius:18px;padding:26px 28px;min-width:230px;flex:1;max-width:320px;text-align:left;">'
        '<div style="font-weight:800;font-size:1.05rem;color:#fff;margin-bottom:3px;">Dr Martin Crane</div>'
        '<div style="font-size:.8rem;color:#FFA700;font-weight:700;margin-bottom:2px;">Research Supervisor</div>'
        '<div style="font-size:.78rem;color:#BFDBFE;margin-bottom:2px;">Associate Professor</div>'
        '<div style="font-size:.78rem;color:#BFDBFE;margin-bottom:10px;">School of Computing, Dublin City University</div>'
        '<div style="font-size:.75rem;color:rgba(255,255,255,.7);margin-bottom:12px;line-height:1.55;">'
        'Research interests: complex systems, computational finance, bioinformatics, '
        'machine learning, and statistical modelling of large-scale data.'
        '</div>'
        '<div style="display:flex;flex-direction:column;gap:7px;">'
        '<a href="mailto:martin.crane@dcu.ie" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">martin.crane@dcu.ie</a>'
        '<a href="https://www.dcu.ie/computing/people/martin-crane" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">DCU Faculty Page</a>'
        '<a href="https://scholar.google.com/citations?user=VwafmG0AAAAJ&hl=en" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">Google Scholar</a>'
        '<a href="https://www.linkedin.com/in/martin-crane-a985a1a/" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">LinkedIn Profile</a>'
        '</div></div>'
        # Card 2b: Co-Supervisor
        '<div style="background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.22);'
        'border-radius:18px;padding:26px 28px;min-width:230px;flex:1;max-width:320px;text-align:left;">'
        '<div style="font-weight:800;font-size:1.05rem;color:#fff;margin-bottom:3px;">Dr Tai Tan Mai</div>'
        '<div style="font-size:.8rem;color:#FFA700;font-weight:700;margin-bottom:2px;">Co-Supervisor</div>'
        '<div style="font-size:.78rem;color:#BFDBFE;margin-bottom:2px;">Assistant Professor</div>'
        '<div style="font-size:.78rem;color:#BFDBFE;margin-bottom:10px;">School of Computing, Dublin City University</div>'
        '<div style="font-size:.75rem;color:rgba(255,255,255,.7);margin-bottom:12px;line-height:1.55;">'
        'Research interests: artificial intelligence, machine learning, healthcare informatics, '
        'natural language processing, and data-driven clinical decision support.'
        '</div>'
        '<div style="display:flex;flex-direction:column;gap:7px;">'
        '<a href="mailto:tai.tanmai@dcu.ie" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">tai.tanmai@dcu.ie</a>'
        '<a href="https://www.dcu.ie/computing/people/tai-tan-mai" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">DCU Faculty Page</a>'
        '<a href="https://scholar.google.com/citations?user=UDh0sOUAAAAJ&hl=en" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">Google Scholar</a>'
        '<a href="https://www.linkedin.com/in/taitanmai/" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">LinkedIn Profile</a>'
        '</div></div>'
        # Card 3: Institution
        '<div style="background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.22);'
        'border-radius:18px;padding:26px 28px;min-width:230px;flex:1;max-width:290px;text-align:left;">'
        '<div style="font-weight:800;font-size:1.05rem;color:#fff;margin-bottom:3px;">Dublin City University</div>'
        '<div style="font-size:.8rem;color:#BFDBFE;margin-bottom:2px;">School of Computing</div>'
        '<div style="font-size:.8rem;color:#BFDBFE;margin-bottom:14px;">DCU F-REC Ethics Approved</div>'
        '<div style="display:flex;flex-direction:column;gap:7px;">'
        '<a href="https://www.dcu.ie" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">www.dcu.ie</a>'
        '<a href="https://www.dcu.ie/computing" target="_blank" '
        'style="color:#93C5FD;font-size:.8rem;text-decoration:none;">School of Computing</a>'
        '</div></div>'
        '</div></div>',
        unsafe_allow_html=True)


def render_contact_footer():
    """Combined Contact + Footer block - used on every page."""
    render_contact()
    render_footer()


def render_footer():
    _flogo = _dcu_logo_uri("dcu_logo_white.png")
    _flogo_html = (f'<img src="{_flogo}" alt="Dublin City University" '
                   f'style="height:34px;width:auto;display:block;margin-bottom:10px;">'
                   if _flogo else '')
    st.markdown(f"""
    <div class="footer">
        <div class="fd">
            Disclaimer: Research prototype at Dublin City University (DCU F-REC Notification-Only
            ethics approval). The model predicts COVID-19 <strong>in-hospital mortality as a proxy</strong>
            for severe outcomes - the source dataset contains no Long-COVID / PASC follow-up labels.
            User testing with dummy data only. Not validated for clinical use, not a certified medical
            device, and not a substitute for professional judgement - always consult a qualified clinician.
        </div>
        <div class="fg2">
            <div>
                {_flogo_html}
                <div class="fb">Long COVID Risk Assessment</div>
                <div class="ft2">COVID-19 mortality-proxy model with calibrated ML and SHAP
                explainability. DCU MSc research prototype - educational use only.</div>
            </div>
            <div>
                <div class="fch">Research</div>
                <span class="fi">MSc Data Analytics - DCU</span>
                <span class="fi">Supervisors: Dr Martin Crane &amp; Dr Tai Tan Mai</span>
                <span class="fi">Durga Prasad Narsing</span>
                <span class="fi">ID: A00050350</span>
            </div>
            <div>
                <div class="fch">Compliance</div>
                <span class="fi">DCU F-REC Ethics Approved</span>
                <span class="fi">GDPR Compliant</span>
                <span class="fi">No Personal Data Collected</span>
                <span class="fi">Dummy Data Only</span>
            </div>
        </div>
        <div class="fbot">
            <span>2026 Durga Prasad Narsing - Dublin City University</span>
            <span>Data: Mexican Government COVID-19 Dataset (Kaggle) - Educational Use Only</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_landing():
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    render_nav()

    # ── Scroll-to-section: triggered when user clicks a nav link on the tool page ──
    if "scroll_to" in st.session_state:
        _scroll_target = st.session_state.pop("scroll_to")
        import streamlit.components.v1 as _nav_comp
        _nav_comp.html(f"""<script>
(function(){{
  var _t = '{_scroll_target}';
  var _last = -1, _stable = 0;
  function _scroll(smooth){{
    try{{
      var el = window.parent.document.getElementById(_t);
      if(!el) return false;
      /* scrollIntoView auto-targets the real scroll container (section[stMain]);
         scroll-margin-top on the anchor (CSS) leaves room for the sticky nav. */
      el.scrollIntoView({{behavior: smooth ? 'smooth' : 'auto', block:'start'}});
      return true;
    }}catch(e){{ return false; }}
  }}
  /* Re-scroll repeatedly until the target's position stops moving (layout has
     settled after late content/images load). Then one final smooth scroll. */
  var _n = 0;
  var _iv = setInterval(function(){{
    _n++;
    var el = window.parent.document.getElementById(_t);
    if(el){{
      var top = Math.round(el.getBoundingClientRect().top);
      _scroll(false);                 /* snap precisely each tick */
      if(top === _last){{ _stable++; }} else {{ _stable = 0; _last = top; }}
      if(_stable >= 2 || _n > 18){{    /* stable for ~2 ticks, or give up */
        clearInterval(_iv);
        _scroll(true);                /* final smooth settle */
      }}
    }}
    if(_n > 20) clearInterval(_iv);
  }}, 150);
}})();
</script>""", height=0, scrolling=False)

    # Hero
    _hero_auc_pct = f"{mdl_metrics['ens']['auc']*100:.1f}%"
    st.markdown(f"""
    <div id="section-overview" class="hero">
        <div>
            <div class="hero-badge">DCU MSc Research Prototype · 2026</div>
            <h1 style="margin-bottom:6px;"><span>Long COVID</span><br>Risk Assessment
                <span style="display:block;font-size:.42em;font-weight:700;color:#FFA700;
                    margin-top:6px;letter-spacing:.01em;">COVID-19 mortality-proxy model</span>
            </h1>
            <p style="margin-top:0;">Predictive modelling for personalised Long COVID risk
               stratification, built on 566,602 patient records with a calibrated
               logistic-regression model, SHAP-based explanation, and the supporting
               clinical evidence.</p>
            <div style="background:rgba(255,255,255,.10);border-left:3px solid #FFA700;
                border-radius:8px;padding:10px 14px;margin:14px 0;font-size:.74rem;
                color:rgba(255,255,255,.82);line-height:1.55;">
                Disclaimer: Research prototype at Dublin City University. Not validated for
                clinical use, not a medical device, and does not constitute medical advice.
                Medications must only be prescribed by qualified clinicians.
            </div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="n countup">566,602</div>
                <div class="l">Patient Records Analysed</div>
            </div>
            <div class="hero-stat">
                <div class="n countup">{_hero_auc_pct} AUC</div>
                <div class="l">Calibrated LR · Test AUC</div>
            </div>
            <div class="hero-stat">
                <div class="n countup">5</div>
                <div class="l">Long COVID Sequelae Assessed</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # (Start Assessment removed from the landing page - use the navbar "🧬 Risk Tool".)

    # Stat bar - AUC from computed model metrics
    _stat_auc_pct = f"{mdl_metrics['ens']['auc']*100:.0f}%"
    st.markdown(f"""
    <div class="stat-bar">
        <div class="stc"><div class="n countup">566K+</div><div class="l">Training Patients</div></div>
        <div class="stc"><div class="n countup">{_stat_auc_pct}</div><div class="l">Model AUC Score</div></div>
        <div class="stc"><div class="n countup">5 Models</div><div class="l">Benchmarked (LR best)</div></div>
        <div class="stc"><div class="n countup">~2.8%</div><div class="l">Gender AUC Gap (within 5% threshold)</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Count-up animator for hero/stat numbers (a polished-product touch). Parses
    # each .countup element's text, preserving prefix/suffix/decimals/commas, and
    # eases from 0 to the value once. Respects prefers-reduced-motion.
    import streamlit.components.v1 as _cu
    _cu.html("""
<script>
(function(){
  try{
    var doc = window.parent.document;
    if(window.parent.matchMedia && window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    function animate(el){
      var full = el.textContent.trim();
      var m = full.match(/^([^0-9]*)([0-9][0-9.,]*)(.*)$/);
      if(!m) return;
      var pre=m[1], numStr=m[2], suf=m[3];
      var dec=(numStr.split('.')[1]||'').length;
      var hasComma=numStr.indexOf(',')>-1;
      var target=parseFloat(numStr.replace(/,/g,''));
      if(isNaN(target)) return;
      var start=null, dur=1000;
      function fmt(v){
        var s = hasComma ? Math.round(v).toLocaleString('en-US')
                         : (dec ? v.toFixed(dec) : Math.round(v).toString());
        return pre+s+suf;
      }
      function step(ts){
        if(!start) start=ts;
        var p=Math.min((ts-start)/dur,1);
        var e=1-Math.pow(1-p,3);
        el.textContent=fmt(target*e);
        if(p<1){ requestAnimationFrame(step); } else { el.textContent=full; }
      }
      requestAnimationFrame(step);
    }
    function run(){
      doc.querySelectorAll('.countup').forEach(function(el){
        if(el.getAttribute('data-cu')) return;
        el.setAttribute('data-cu','1');
        animate(el);
      });
    }
    setTimeout(run, 250);
  }catch(e){}
})();
</script>
""", height=0)

    # ── Data at a glance (EDA) - real figures from the analysis ───────────────
    def _fig_grid(items):
        """items: list of (relpath, caption). Returns an HTML flex grid of figures."""
        _h = ""
        for _rel, _cap in items:
            _u = _plot_uri(_rel)
            if not _u:
                continue
            _h += (
                '<figure style="flex:1;min-width:240px;margin:0;background:#fff;'
                'border:1px solid #BFDBFE;border-radius:14px;padding:12px;'
                'box-shadow:0 3px 14px rgba(13,27,62,.08);">'
                f'<img src="{_u}" style="width:100%;height:auto;border-radius:8px;display:block;">'
                f'<figcaption style="font-size:.75rem;color:#4B5563;margin-top:8px;text-align:center;">{_cap}</figcaption>'
                '</figure>')
        return f'<div style="display:flex;gap:14px;flex-wrap:wrap;">{_h}</div>' if _h else ""

    _eda = _fig_grid([
        ("eda_plots/01_age_distribution.png",   "<strong>Age distribution</strong> of the confirmed-case cohort."),
        ("eda_plots/05_mortality_by_age.png",   "<strong>Mortality by age</strong> - risk rises sharply with age."),
        ("eda_plots/03_condition_prevalence.png","<strong>Comorbidity prevalence</strong> across the cohort."),
    ])
    if _eda:
        with st.expander("Data at a glance - exploratory analysis"):
            st.markdown(_eda, unsafe_allow_html=True)

    # Model Performance Section
    st.markdown('<div id="section-performance">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sh">'
        '<h2>Model <span>Performance</span></h2>'
        f'<p>Deployed model: calibrated, tuned Logistic Regression on '
        f'{mdl_metrics["n_train"]:,} confirmed COVID-19 cases, validated on '
        f'{mdl_metrics["n_test"]:,} held-out records (stratified 80/20). LR/RF/GB/XGBoost/'
        f'Stacking benchmarked - no model significantly beats LR (DeLong), so the '
        f'interpretable LR is deployed.</p>'
        '</div>',
        unsafe_allow_html=True)

    # AUC cards - values computed from stratified hold-out test set
    col1, col2, col3, col4 = st.columns(4)
    models_data = [
        (col1, "Logistic Regression", f"{mdl_metrics['lr']['auc']:.3f}", "#1D4ED8", "#DBEAFE", mdl_metrics['lr']['auc']*100),
        (col2, "Random Forest",       f"{mdl_metrics['rf']['auc']:.3f}", "#059669", "#D1FAE5", mdl_metrics['rf']['auc']*100),
        (col3, "Gradient Boosting",   f"{mdl_metrics['gb']['auc']:.3f}", "#7C3AED", "#EDE9FE", mdl_metrics['gb']['auc']*100),
        (col4, "Primary (Deployed LR)", f"{mdl_metrics['ens']['auc']:.3f}", None,     None,      mdl_metrics['ens']['auc']*100),
    ]
    for col, name, auc, clr, bg, pct in models_data:
        with col:
            if clr is None:
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#1A3C66,#2E5C92);border-radius:14px;padding:20px 16px;text-align:center;box-shadow:0 6px 22px rgba(26,60,102,.3);border:2px solid #2E5C92;">'
                    f'<div style="font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:rgba(255,255,255,.8);margin-bottom:9px;">{name}</div>'
                    f'<div style="font-size:2rem;font-weight:900;color:#fff;line-height:1;">{auc}</div>'
                    f'<div style="font-size:.73rem;color:rgba(255,255,255,.8);margin-top:4px;font-weight:500;">AUC-ROC</div>'
                    f'<div style="background:rgba(255,255,255,.2);border-radius:5px;height:7px;margin-top:10px;overflow:hidden;">'
                    f'<div style="width:{pct}%;height:100%;background:rgba(255,255,255,.8);border-radius:5px;"></div></div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:#fff;border-radius:14px;padding:20px 16px;text-align:center;box-shadow:0 4px 18px rgba(13,27,62,.09);border:2px solid {bg};">'
                    f'<div style="font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{clr};margin-bottom:9px;">{name}</div>'
                    f'<div style="font-size:2rem;font-weight:900;color:{clr};line-height:1;">{auc}</div>'
                    f'<div style="font-size:.73rem;color:#374151;margin-top:4px;font-weight:500;">AUC-ROC</div>'
                    f'<div style="background:{bg};border-radius:5px;height:7px;margin-top:10px;overflow:hidden;">'
                    f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{clr},{clr}88);border-radius:5px;"></div></div></div>',
                    unsafe_allow_html=True)

    # Metric cards - values computed from stratified 80/20 hold-out test set
    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
    # Operating point at F1-optimal threshold 0.26 (covid_threshold_optimization.csv)
    metrics = [
        (mc1, f"{mdl_metrics['ens']['recall']:.1f}%",      "Sensitivity",  "#1A3C66","#DBEAFE"),
        (mc2, f"{mdl_metrics['ens']['specificity']:.1f}%", "Specificity",  "#1A3C66","#DBEAFE"),
        (mc3, f"{mdl_metrics['ens']['precision']:.1f}%",   "Precision",    "#1A3C66","#DBEAFE"),
        (mc4, f"{mdl_metrics['ens']['f1']:.1f}%",          "F1-Score",     "#1A3C66","#DBEAFE"),
        (mc5, f"{mdl_metrics['ens']['brier']:.3f}",        "Brier Score",  "#059669","#D1FAE5"),
        (mc6, f"{mdl_metrics['ens']['ece']:.4f}",          "ECE (count-wtd)", "#059669","#D1FAE5"),
    ]
    for col, val, lbl, clr, bg in metrics:
        with col:
            st.markdown(
                f'<div style="background:#fff;border-radius:11px;padding:15px 10px;text-align:center;box-shadow:0 3px 12px rgba(13,27,62,.08);border:2px solid {bg};">'
                f'<div style="font-size:1.4rem;font-weight:900;color:{clr};line-height:1;">{val}</div>'
                f'<div style="font-size:.71rem;color:#374151;margin-top:3px;font-weight:600;">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True)

    # Validation note - deployed model: calibration + temporal validation
    _ens_auc_str  = f"{mdl_metrics['ens']['auc']:.4f}"
    _ci_lo        = f"{mdl_metrics['ens']['ci_lower']:.3f}"
    _ci_hi        = f"{mdl_metrics['ens']['ci_upper']:.3f}"
    _ens_ece_str  = f"{mdl_metrics['ens']['ece']:.4f}"
    _temporal_str = f"{mdl_metrics['temporal_auc']:.3f}"
    st.markdown(
        f'<div style="background:#fff;border-radius:11px;padding:13px 18px;'
        f'border-left:4px solid #2E5C92;box-shadow:0 3px 12px rgba(13,27,62,.07);'
        f'margin-top:14px;display:flex;gap:18px;flex-wrap:wrap;align-items:center;">'
        f'<div style="font-size:.81rem;color:#0D1B3E;font-weight:700;">Validation:</div>'
        f'<span style="font-size:.79rem;color:#374151;">Stratified 80/20 hold-out</span>'
        f'<span style="background:#DBEAFE;color:#1D4ED8;border-radius:4px;padding:2px 9px;'
        f'font-size:.75rem;font-weight:700;">AUC {_ens_auc_str} (95% CI {_ci_lo}–{_ci_hi})</span>'
        f'<span style="background:#EDE9FE;color:#5B21B6;border-radius:4px;padding:2px 9px;'
        f'font-size:.75rem;font-weight:700;">Temporal AUC {_temporal_str}</span>'
        f'<span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:2px 9px;'
        f'font-size:.75rem;font-weight:700;">Isotonic-calibrated · ECE {_ens_ece_str} (count-weighted)</span>'
        f'</div>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model benchmark comparison table ──────────────────────────────────────
    # Every model is re-trained on the same split and isotonic-calibrated identically,
    # then evaluated at its OWN F1-optimal threshold (analysis_output/advanced/
    # covid_model_benchmark_full.csv). Deployed-LR operating point is the production
    # value from the main pipeline (covid_threshold_optimization.csv, cv=5 calibration);
    # AUCs are the canonical single-model values (covid_auc_confidence_intervals.csv).
    with st.expander("Model benchmark comparison — all 5 models"):
        _bench = pd.DataFrame({
            "Model": [
                "Logistic Regression (deployed)",
                "Gradient Boosting",
                "XGBoost",
                "Stacking Ensemble",
                "Random Forest",
            ],
            "AUC":         [mdl_metrics["ens"]["auc"], mdl_metrics["gb"]["auc"], 0.8879, 0.8876, mdl_metrics["rf"]["auc"]],
            "Sensitivity": [round(mdl_metrics["ens"]["recall"] / 100, 3), 0.716, 0.656, 0.677, 0.673],
            "Specificity": [round(mdl_metrics["ens"]["specificity"] / 100, 3), 0.873, 0.895, 0.887, 0.876],
            "Precision":   [round(mdl_metrics["ens"]["precision"] / 100, 3), 0.441, 0.466, 0.457, 0.432],
            "F1":          [round(mdl_metrics["ens"]["f1"] / 100, 3), 0.546, 0.545, 0.546, 0.526],
            "Brier":       [mdl_metrics["ens"]["brier"], 0.078, 0.078, 0.078, 0.081],
            "ECE":         [mdl_metrics["ens"]["ece"], 0.004, 0.004, 0.003, 0.006],
            "Deployed":    ["Yes", "No", "No", "No", "No"],
        })
        st.dataframe(
            _bench,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Every model re-trained on the same stratified 80/20 split and isotonic-calibrated "
            "identically, then evaluated at its own F1-optimal threshold. All five are "
            "well-calibrated (ECE < 0.01). DeLong test: no model significantly outperforms the "
            "Logistic Regression, so the interpretable, calibrated LR is deployed."
        )

    # ── Evidence plots - the real figures from covid_analysis_full.py ─────────
    _roc_uri = _plot_uri("model_comparison_plots/02_roc_curves.png")
    _cal_uri = _plot_uri("calibration_plots/02_reliability_diagram.png")
    if _roc_uri or _cal_uri:
        _imgs = ""
        if _roc_uri:
            _imgs += (
                '<figure style="flex:1;min-width:280px;margin:0;background:#fff;'
                'border:1px solid #BFDBFE;border-radius:14px;padding:14px;'
                'box-shadow:0 3px 14px rgba(13,27,62,.08);">'
                f'<img src="{_roc_uri}" alt="ROC curves" style="width:100%;height:auto;border-radius:8px;">'
                '<figcaption style="font-size:.76rem;color:#4B5563;margin-top:8px;text-align:center;">'
                '<strong style="color:#0D1B3E;">ROC curves</strong> - all benchmarked models on the hold-out set.</figcaption>'
                '</figure>')
        if _cal_uri:
            _imgs += (
                '<figure style="flex:1;min-width:280px;margin:0;background:#fff;'
                'border:1px solid #BFDBFE;border-radius:14px;padding:14px;'
                'box-shadow:0 3px 14px rgba(13,27,62,.08);">'
                f'<img src="{_cal_uri}" alt="Reliability diagram" style="width:100%;height:auto;border-radius:8px;">'
                '<figcaption style="font-size:.76rem;color:#4B5563;margin-top:8px;text-align:center;">'
                '<strong style="color:#0D1B3E;">Reliability diagram</strong> - predicted vs. observed after isotonic calibration.</figcaption>'
                '</figure>')
        st.markdown(
            '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:16px;">'
            + _imgs + '</div>', unsafe_allow_html=True)

    # ── Model Card (collapsible) - the single most defensible artefact ────────
    with st.expander("Model Card - full technical summary"):
        st.markdown(f"""
**Model name:** Long COVID Risk Assessment - COVID-19 mortality-proxy model
**Owner:** Durga Prasad Narsing · MSc Data Analytics, Dublin City University (2026)

| Field | Detail |
|---|---|
| **Intended use** | Research/education prototype for risk *stratification* - **not** a medical device, **not** for clinical decisions. |
| **Target** | In-hospital mortality (used as a *proxy* for severe outcome / Long COVID risk). |
| **Data** | 566,602 real records → {mdl_metrics['n_total']:,} confirmed COVID-19 cases (Mexican open COVID dataset). |
| **Split** | Stratified 80/20 - {mdl_metrics['n_train']:,} train / {mdl_metrics['n_test']:,} test. |
| **Algorithm** | Tuned **Logistic Regression**, isotonic-**calibrated** (selected over RF, GB, XGBoost & Stacking; no model significantly beats LR by DeLong). |
| **Pipeline** | median imputation → standard scaling → calibrated LR. |
| **Features (9)** | Age, Sex, Diabetes, Hypertension, Cardiovascular, Pneumonia, Obesity, Asthma, COPD. |
| **Discrimination** | AUC **{mdl_metrics['ens']['auc']:.4f}** (95% CI {mdl_metrics['ens']['ci_lower']:.3f}–{mdl_metrics['ens']['ci_upper']:.3f}); temporal AUC {mdl_metrics['temporal_auc']:.3f}. |
| **Operating point** | Threshold **{mdl_metrics['threshold']:.2f}** (F1-optimal): sensitivity {mdl_metrics['ens']['recall']:.1f}%, specificity {mdl_metrics['ens']['specificity']:.1f}%, precision {mdl_metrics['ens']['precision']:.1f}%, F1 {mdl_metrics['ens']['f1']:.1f}%. |
| **Calibration** | Brier {mdl_metrics['ens']['brier']:.3f}, ECE {mdl_metrics['ens']['ece']:.4f} (count-weighted). |
| **Fairness** | Audited across age, sex, comorbidity - gender AUC gap ~2.8% (within 5% threshold); older subgroups (50–65, 65+) and diabetes subgroup flagged for monitoring. |
| **Ethics** | DCU F-REC compliant; fully de-identified open data. |

**Known limitations**
- Mortality is a **proxy**, not a direct Long COVID diagnosis - interpret accordingly.
- Single-country dataset; external/geographic generalisation is unverified.
- **Low precision at the chosen threshold** (≈{mdl_metrics['ens']['precision']:.0f}%) - suitable for *screening/triage*, not confirmation.
- Long COVID sequelae scores are clinically-informed **heuristics**, not direct model outputs.
""")

    st.markdown("<hr class='div' style='margin:28px 0 20px;'>", unsafe_allow_html=True)

    # Feature Importance Section - LR-based SHAP values from covid_analysis_full.py
    # (analysis_output/analysis/covid_shap_values.csv). Top 6 shown.
    _shap_vals = load_shap_values()[:6]
    fi_colors = ["#DC2626","#D97706","#7C3AED","#1D4ED8","#059669","#0891B2"]
    fi_bg     = ["#FEE2E2","#FEF3C7","#EDE9FE","#DBEAFE","#D1FAE5","#CFFAFE"]

    fi_bars_html = ""
    for (_lbl, _pct), clr, bg in zip(_shap_vals, fi_colors, fi_bg):
        bar_w = min(round(_pct * 2, 0), 100)
        # No indentation - indented HTML inside st.markdown triggers CommonMark code blocks
        fi_bars_html += (
            f'<div style="margin-bottom:8px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;font-size:.81rem;font-weight:700;">'
            f'<span style="color:#0D1B3E;">{_lbl}</span>'
            f'<span style="color:{clr};">{_pct}%</span>'
            f'</div>'
            f'<div style="background:{bg};border-radius:5px;height:12px;overflow:hidden;">'
            f'<div class="anim-bar" style="width:{bar_w:.0f}%;height:100%;background:linear-gradient(90deg,{clr},{clr}88);border-radius:5px;"></div>'
            f'</div>'
            f'</div>'
        )

    top2_names = _shap_vals[0][0], _shap_vals[1][0]
    top2_pct   = round(_shap_vals[0][1] + _shap_vals[1][1], 1)

    # SHAP section - ONE HTML block (flex 2-col) so the content lives INSIDE the
    # card (no empty white box) and the #section-shap anchor lands on the heading.
    _shap_html = (
        '<div id="section-shap" style="background:#fff;border-radius:18px;padding:24px 28px;'
        'box-shadow:0 4px 20px rgba(26,60,102,.09);border:1px solid #BFDBFE;margin-bottom:26px;">'
        '<div style="display:flex;gap:34px;flex-wrap:wrap;align-items:flex-start;">'
        # ── LEFT: Tree-SHAP bars ──
        '<div style="flex:1;min-width:300px;">'
        '<div style="font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#2E5C92;margin-bottom:5px;">Tree SHAP (XGBoost) - Feature Importance</div>'
        '<div style="font-size:1.35rem;font-weight:800;color:#0D1B3E;margin-bottom:4px;">What Drives <span style="color:#2E5C92;">Predictions?</span></div>'
        '<div style="font-size:.83rem;color:#374151;margin-bottom:14px;line-height:1.6;">Tree SHAP values (from the XGBoost benchmark model) show which clinical features most impact mortality risk. <em style="color:#6B7280;font-size:.76rem;">(Global mean |SHAP|, normalised to 100%)</em></div>'
        f'{fi_bars_html}'
        '</div>'
        # ── RIGHT: Clinical interpretation ──
        '<div style="flex:1;min-width:300px;">'
        '<div style="font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#2E5C92;margin-bottom:5px;">Clinical Interpretation</div>'
        '<div style="font-size:1.35rem;font-weight:800;color:#0D1B3E;margin-bottom:14px;">Why <span style="color:#2E5C92;">Explainability</span> Matters</div>'
        '<div style="background:#FFF5F5;border-radius:9px;padding:13px 15px;border-left:4px solid #DC2626;margin-bottom:10px;">'
        f'<div style="font-weight:700;font-size:.84rem;color:#991B1B;margin-bottom:3px;">{top2_names[0]} - Strongest Predictor</div>'
        '<div style="font-size:.78rem;color:#374151;line-height:1.55;">Highest single-feature impact on mortality - consistent with clinical literature on COVID severity.</div></div>'
        '<div style="background:#FFFBEB;border-radius:9px;padding:13px 15px;border-left:4px solid #D97706;margin-bottom:10px;">'
        f'<div style="font-weight:700;font-size:.84rem;color:#92400E;margin-bottom:3px;">{top2_names[1]} - Second Strongest</div>'
        '<div style="font-size:.78rem;color:#374151;line-height:1.55;">Older age is a leading driver of severe-outcome risk - consistent with the COVID-19 mortality literature.</div></div>'
        '<div style="background:#EFF6FF;border-radius:9px;padding:13px 15px;border-left:4px solid #1D4ED8;margin-bottom:10px;">'
        '<div style="font-weight:700;font-size:.84rem;color:#1E40AF;margin-bottom:3px;">Transparency Builds Clinical Trust</div>'
        '<div style="font-size:.78rem;color:#374151;line-height:1.55;">Every prediction is explainable - clinicians see exactly which factors drove the risk score for each individual patient.</div></div>'
        '<div style="background:linear-gradient(135deg,#1A3C66,#2E5C92);border-radius:9px;padding:11px 15px;">'
        f'<div style="font-size:.79rem;color:rgba(255,255,255,.9);line-height:1.6;">Key insight: {top2_names[0]} + {top2_names[1]} account for <strong style="color:#93C5FD;">{top2_pct}%</strong> of the prediction - the primary clinical intervention targets.</div></div>'
        '</div>'   # end right
        '</div>'   # end flex
        '</div>'   # end card
    )
    st.markdown(_shap_html, unsafe_allow_html=True)

    # Real SHAP summary plot from the analysis (covid_analysis_full.py)
    _shap_uri = _plot_uri("shap_plots/01_shap_feature_importance.png")
    if _shap_uri:
        st.markdown(
            '<figure style="margin:0 0 26px;background:#fff;border:1px solid #BFDBFE;'
            'border-radius:18px;padding:18px 22px;box-shadow:0 4px 20px rgba(26,60,102,.09);">'
            '<figcaption style="font-size:.71rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:.08em;color:#2E5C92;margin-bottom:10px;">SHAP summary - generated figure</figcaption>'
            f'<img src="{_shap_uri}" alt="SHAP feature importance" '
            'style="width:100%;max-width:760px;height:auto;display:block;margin:0 auto;border-radius:8px;">'
            '</figure>', unsafe_allow_html=True)

    # Methodology
    st.markdown("""
    <div id="section-methodology" class="sh">
        <h2>Built for <span>Clinical Intelligence</span></h2>
        <p>A research-grade tool combining interpretable ML with evidence-based
           clinical guidelines for Long COVID risk stratification.</p>
    </div>
    <div class="mtl">
        <div class="mtl-step"><div class="mtl-num">1</div><div class="mtl-ic">🗄️</div>
            <div class="mtl-t">Data</div>
            <div class="mtl-d">566,602 real patient records; 220,218 confirmed COVID-19 cases selected.</div></div>
        <div class="mtl-arrow">→</div>
        <div class="mtl-step"><div class="mtl-num">2</div><div class="mtl-ic">🧹</div>
            <div class="mtl-t">Clean &amp; Encode</div>
            <div class="mtl-d">Validate ages, encode 9 acute-phase features, median-impute, standard-scale.</div></div>
        <div class="mtl-arrow">→</div>
        <div class="mtl-step"><div class="mtl-num">3</div><div class="mtl-ic">🤖</div>
            <div class="mtl-t">Train 5 Models</div>
            <div class="mtl-d">LR, RF, GB, XGBoost &amp; Stacking benchmarked on a stratified 80/20 split.</div></div>
        <div class="mtl-arrow">→</div>
        <div class="mtl-step"><div class="mtl-num">4</div><div class="mtl-ic">🎯</div>
            <div class="mtl-t">Calibrate</div>
            <div class="mtl-d">Isotonic calibration of the deployed LR (Brier 0.078, ECE 0.0022).</div></div>
        <div class="mtl-arrow">→</div>
        <div class="mtl-step"><div class="mtl-num">5</div><div class="mtl-ic">✅</div>
            <div class="mtl-t">Validate</div>
            <div class="mtl-d">Hold-out AUC 0.888, temporal AUC 0.891, fairness audit across subgroups.</div></div>
    </div>
    <div class="fg">
        <div class="fc"><div class="fc-icon">🤖🧬</div>
            <div class="fc-title">Calibrated ML Prediction</div>
            <div class="fc-text">Calibrated, tuned Logistic Regression - selected over RF, GB,
            XGBoost &amp; Stacking by DeLong test. 0.89 AUC on 44,000+ held-out records.</div>
        </div>
        <div class="fc"><div class="fc-icon">🔍📊</div>
            <div class="fc-title">SHAP Explainability</div>
            <div class="fc-text">Every prediction explained through feature impact analysis.
            Clinicians see why a risk score was generated, not just the number.</div>
        </div>
        <div class="fc"><div class="fc-icon">🫁❤️🧠🔋⚗️</div>
            <div class="fc-title">5 Long COVID Sequelae</div>
            <div class="fc-text">Respiratory, Cardiac, Neurological (brain fog),
            Systemic (fatigue) and Metabolic - each assessed independently.</div>
        </div>
        <div class="fc"><div class="fc-icon">⚖️🩺</div>
            <div class="fc-title">Fairness Audited</div>
            <div class="fc-text">Tested across age, sex &amp; comorbidity. Gender AUC gap ~2.8% (within the 5% threshold); older groups (50–65, 65+) and diabetes subgroup flagged for monitoring.</div>
        </div>
        <div class="fc"><div class="fc-icon">📅🏥</div>
            <div class="fc-title">Personalised Care Timeline</div>
            <div class="fc-text">Phase-based clinical recommendations from Week 1 to Month 4+,
            tailored to risk level and patient age.</div>
        </div>
        <div class="fc"><div class="fc-icon">💊💰</div>
            <div class="fc-title">Cost-Benefit (Illustrative)</div>
            <div class="fc-text">Hypothetical scenario only: earlier triage of high-risk patients
            could yield substantial savings - figures depend on assumed intervention costs and are
            not empirically validated.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Who Can Use
    st.markdown("""
    <div id="section-research" class="wu">
        <div class="sh">
            <h2>Who Can Use <span>This Tool</span></h2>
            <p>Designed for healthcare professionals, researchers, and educators.
            Not approved for direct clinical decision-making.</p>
        </div>
        <div class="wu-grid">
            <div class="wu-card"><div class="wu-icon">🏥</div>
                <div class="wu-title">Consultant Pulmonologist</div>
                <div class="wu-text">Respiratory specialists evaluating post-COVID lung function
                and long-term sequelae risk.</div>
            </div>
            <div class="wu-card"><div class="wu-icon">❤️</div>
                <div class="wu-title">Consultant Cardiologist</div>
                <div class="wu-text">Cardiac specialists assessing cardiovascular complications
                in post-acute COVID-19 syndrome.</div>
            </div>
            <div class="wu-card"><div class="wu-icon">🧠</div>
                <div class="wu-title">Consultant Neurologist</div>
                <div class="wu-text">Neurologists investigating cognitive symptoms including
                brain fog and memory impairment.</div>
            </div>
            <div class="wu-card"><div class="wu-icon">📊</div>
                <div class="wu-title">Healthcare Data Scientist</div>
                <div class="wu-text">Clinical analytics professionals building predictive
                models for risk stratification.</div>
            </div>
            <div class="wu-card"><div class="wu-icon">🔬</div>
                <div class="wu-title">Clinical Research Scientist</div>
                <div class="wu-text">Researchers investigating Long COVID epidemiology,
                risk factors, and intervention outcomes.</div>
            </div>
            <div class="wu-card"><div class="wu-icon">🎓</div>
                <div class="wu-title">Health Informatics Educator</div>
                <div class="wu-text">Lecturers in medical informatics using real-world
                case studies for teaching clinical AI.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ethics
    st.markdown("""
    <div id="section-ethics" class="ethics">
        <h3>Research Ethics and Compliance - DCU F-REC Approved</h3>
        <div class="eg">
            <div class="ep"><span class="epc">✓</span>Approved under DCU F-REC Notification Only ethics review</div>
            <div class="ep"><span class="epc">✓</span>All interactions use dummy patient data - no personal data stored</div>
            <div class="ep"><span class="epc">✓</span>AI system does not replace clinical judgment or decision-making</div>
            <div class="ep"><span class="epc">✓</span>Target is in-hospital mortality used as a proxy - no Long-COVID/PASC labels exist in the dataset</div>
            <div class="ep"><span class="epc">✓</span>Fairness audited across age, sex &amp; comorbidity; age-group disparities transparently flagged</div>
            <div class="ep"><span class="epc">✓</span>Trained on anonymised Mexican Government COVID-19 Dataset</div>
            <div class="ep"><span class="epc">✓</span>GDPR compliant - zero personally identifiable information</div>
            <div class="ep"><span class="epc">✓</span>Supervised by Dr Martin Crane &amp; Dr Tai Tan Mai, School of Computing, DCU</div>
            <div class="ep"><span class="epc">✓</span>For research and user testing only - not clinical deployment</div>
        </div>
        <div class="disc">Disclaimer: Research prototype at Dublin City University. Not validated for
        clinical use, not a medical device, and does not constitute medical advice. Medications must
        only be prescribed by qualified clinicians.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fairness audit plots (real figures) ──────────────────────────────────
    _fair = _fig_grid([
        ("fairness_plots/03_fairness_gender.png", "<strong>Fairness by sex</strong> - AUC gap ~2.8% (within the 5% threshold)."),
        ("fairness_plots/01_fairness_age.png",    "<strong>Fairness by age group</strong> - 50+ flagged for monitoring."),
    ])
    if _fair:
        with st.expander("Fairness audit - subgroup performance"):
            st.markdown(_fair, unsafe_allow_html=True)

    # ── References & evidence base ───────────────────────────────────────────
    # Curated subset of the project's 27-paper bibliography - each entry maps to a
    # method/feature visible in this app. Full list is in the written report.
    with st.expander("References & evidence base"):
        st.markdown(
            '<div style="font-size:.84rem;color:#374151;line-height:1.6;margin-bottom:10px;">'
            'Each method in this prototype is grounded in published work. The papers below '
            'map directly to features shown in the app; the full 27-reference bibliography '
            '(11 clinical + 16 methodology) is in the written report.</div>'
            '<ol style="font-size:.83rem;color:#1F2937;line-height:1.65;padding-left:22px;margin:0;">'
            '<li><strong>Guzman-Esquivel, J. et al. (2023).</strong> Acute-phase predictors of '
            'Long COVID. <em>Healthcare</em>, 11(2):197. '
            '<a href="https://doi.org/10.3390/healthcare11020197" target="_blank" rel="noopener">doi.org/10.3390/healthcare11020197</a> '
            '- feature selection (acute-phase risk factors).</li>'
            '<li><strong>DeLong, E.R. et al. (1988).</strong> Comparing areas under correlated ROC '
            'curves. <em>Biometrics</em>, 44(3):837–845. '
            '<a href="https://doi.org/10.2307/2531595" target="_blank" rel="noopener">doi.org/10.2307/2531595</a> '
            '- model comparison (LR vs RF/GB/XGBoost).</li>'
            '<li><strong>Guo, C. et al. (2017).</strong> On calibration of modern neural networks. '
            '<em>ICML</em>. '
            '<a href="https://proceedings.mlr.press/v70/guo17a.html" target="_blank" rel="noopener">proceedings.mlr.press/v70/guo17a.html</a> '
            '- Expected Calibration Error (ECE) metric.</li>'
            '<li><strong>Niculescu-Mizil, A. &amp; Caruana, R. (2005).</strong> Predicting good '
            'probabilities with supervised learning. <em>ICML</em>. '
            '<a href="https://doi.org/10.1145/1102351.1102430" target="_blank" rel="noopener">doi.org/10.1145/1102351.1102430</a> '
            '- probability calibration (isotonic).</li>'
            '<li><strong>Lundberg, S.M. &amp; Lee, S.-I. (2017).</strong> A unified approach to '
            'interpreting model predictions (SHAP). <em>NeurIPS</em>. '
            '<a href="https://arxiv.org/abs/1705.07874" target="_blank" rel="noopener">arxiv.org/abs/1705.07874</a> '
            '- SHAP global + per-patient explanations.</li>'
            '<li><strong>Lundberg, S.M. et al. (2020).</strong> Local explanations to global '
            'understanding with trees. <em>Nature Machine Intelligence</em>, 2:56–67. '
            '<a href="https://doi.org/10.1038/s42256-019-0138-9" target="_blank" rel="noopener">doi.org/10.1038/s42256-019-0138-9</a> '
            '- Tree SHAP (XGBoost) importance.</li>'
            '<li><strong>Hardt, M. et al. (2016).</strong> Equality of opportunity in supervised '
            'learning. <em>NeurIPS</em>. '
            '<a href="https://arxiv.org/abs/1610.02413" target="_blank" rel="noopener">arxiv.org/abs/1610.02413</a> '
            '- fairness audit &amp; mitigation.</li>'
            '<li><strong>Mitchell, M. et al. (2019).</strong> Model cards for model reporting. '
            '<em>FAT*</em>. '
            '<a href="https://doi.org/10.1145/3287560.3287596" target="_blank" rel="noopener">doi.org/10.1145/3287560.3287596</a> '
            '- the Model Card above.</li>'
            '</ol>'
            '<div style="font-size:.76rem;color:#6B7280;margin-top:12px;border-top:1px solid #EFF2F7;'
            'padding-top:8px;line-height:1.55;">Dataset: Mexican Government open COVID-19 dataset '
            '(anonymised, public). These sources justify the methods implemented here; this '
            'prototype is not a clinical guideline.</div>',
            unsafe_allow_html=True)

    render_contact_footer()


# ─────────────────────────────────────────────────────────────────────────────
# TOOL PAGE  (sidebar + results)
# ─────────────────────────────────────────────────────────────────────────────
def page_tool():
    # Hide sidebar completely - inputs are now inline
    st.markdown("""
    <style>
    section[data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] { display:none !important; }
    </style>
    """, unsafe_allow_html=True)
    render_nav()

    # ── Sample patients: auto-selected from real training data ─────────────────
    # find_sample_patients() scans the dataset and picks the real patients whose
    # calibrated risk is closest to 80 / 58 / 38 / 10 %, so each Load button lands
    # in its intended risk band (Critical / High / Medium / Low).
    SAMPLES, _sample_scores = find_sample_patients(imp_m, sc_m, model_m)

    # ── Detect sample patient clicked in HTML component (via query param) ─────
    try:
        _sp_key = st.query_params.get("_sp", "")
        if _sp_key and _sp_key in SAMPLES:
            st.session_state["_pending_sample"] = SAMPLES[_sp_key]
            st.session_state.patient = SAMPLES[_sp_key]
            st.session_state.score   = _sample_scores[_sp_key]
            st.query_params.pop("_sp", None)
            st.rerun()
    except Exception:
        pass  # st.query_params not available in older Streamlit builds

    pt    = st.session_state.patient
    score = st.session_state.score

    # ── Overview section (assessment-page intro + info banner, combined) ───────
    st.markdown(
        '<div id="section-overview" style="background:#fff;border-radius:16px;'
        'padding:20px 26px;margin:2px 0 14px;box-shadow:0 2px 14px rgba(26,60,102,.08);'
        'border:1px solid #BFDBFE;border-left:5px solid #FFA700;">'
        # header row: title block (left) + model/compliance chips (right)
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'flex-wrap:wrap;gap:12px;">'
        '<div style="flex:1;min-width:240px;">'
        '<div style="font-size:.72rem;font-weight:800;color:#2E5C92;text-transform:uppercase;'
        'letter-spacing:.07em;margin-bottom:4px;">Overview</div>'
        '<div style="font-size:1.15rem;font-weight:800;color:#0D1B3E;margin-bottom:6px;">'
        'Long COVID Risk Assessment Tool</div></div>'
        '<div style="display:flex;gap:10px;flex-shrink:0;flex-wrap:wrap;">'
        '<div style="background:#EFF6FF;border-radius:8px;padding:6px 14px;'
        'font-size:.74rem;font-weight:700;color:#1E40AF;border:1px solid #BFDBFE;">'
        'Calibrated logistic regression · 9 features</div>'
        '<div style="background:#F0FDF4;border-radius:8px;padding:6px 14px;'
        'font-size:.74rem;font-weight:700;color:#166534;border:1px solid #BBF7D0;">'
        'DCU F-REC Compliant</div>'
        '</div></div>'
        # description
        '<div style="font-size:.84rem;color:#374151;line-height:1.6;margin-top:8px;">'
        'Enter a patient profile below to get a calibrated COVID-19 mortality-proxy risk score '
        '(0–100%) with a risk band and the factors driving it. Use a '
        '<strong style="color:#1A3C66;">Sample Patient</strong> to autofill the form, or complete it '
        'manually, then click <strong style="color:#1A3C66;">Assess Risk</strong>. '
        'Trained on 220,218 confirmed COVID-19 cases · AUC 0.888 · isotonic-calibrated.</div>'
        # conditions assessed
        '<div style="font-size:.76rem;color:#6B7280;margin-top:8px;border-top:1px solid #EFF2F7;'
        'padding-top:8px;">Conditions assessed: Respiratory &nbsp;·&nbsp; Cardiac &nbsp;·&nbsp; '
        'Neurological &nbsp;·&nbsp; Systemic &nbsp;·&nbsp; Metabolic</div>'
        '</div>',
        unsafe_allow_html=True)

    # ── Stat bar (matches landing page - same CSS class) ──────────────────────
    _tool_auc_pct = f"{mdl_metrics['ens']['auc']*100:.0f}%"
    st.markdown(f"""
<div class="stat-bar">
<div class="stc"><div class="n">566K+</div><div class="l">Training Patients</div></div>
<div class="stc"><div class="n">{_tool_auc_pct}</div><div class="l">Model AUC Score</div></div>
<div class="stc"><div class="n">5 Models</div><div class="l">Benchmarked (LR best)</div></div>
<div class="stc"><div class="n">~2.8%</div><div class="l">Gender AUC Gap (within 5% threshold)</div></div>
</div>
""", unsafe_allow_html=True)

    # ── Batch prediction — upload CSV, download results ──────────────────────
    with st.expander("Batch prediction — upload a CSV of multiple patients"):
        st.markdown(
            '<div style="font-size:.8rem;color:#374151;margin-bottom:10px;">'
            'Upload a CSV with columns: <code>age, sex, diabetes, hypertension, '
            'cardiovascular, pneumonia, obesity, asthma, copd</code>. '
            'Sex: 1 = Female, 0 = Male. Conditions: 1 = Yes, 0 = No.'
            '</div>',
            unsafe_allow_html=True)
        _batch_file = st.file_uploader("Choose CSV file", type="csv", key="batch_upload",
                                        label_visibility="collapsed")
        if _batch_file is not None:
            try:
                _REQUIRED = ["age","sex","diabetes","hypertension","cardiovascular",
                             "pneumonia","obesity","asthma","copd"]
                _bdf = pd.read_csv(_batch_file)
                _missing = [c for c in _REQUIRED if c not in _bdf.columns]
                if _missing:
                    st.error(f"Missing columns: {', '.join(_missing)}")
                else:
                    _bdf = _bdf.copy()
                    _scores = []
                    for _, _row in _bdf.iterrows():
                        _pt = {c: _row[c] for c in _REQUIRED}
                        _scores.append(round(predict_risk(imp_m, sc_m, model_m, _pt), 1))
                    _bdf["risk_score"] = _scores
                    _bdf["risk_band"] = _bdf["risk_score"].apply(rlevel)
                    st.dataframe(_bdf[_REQUIRED + ["risk_score", "risk_band"]],
                                 use_container_width=True, hide_index=True)
                    _csv_out = _bdf.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download results ({len(_bdf)} patients)",
                        data=_csv_out,
                        file_name="longcovid_risk_results.csv",
                        mime="text/csv",
                        key="batch_download",
                    )
            except Exception as _e:
                st.error(f"Could not process file: {_e}")

    # ── SAMPLE PATIENTS - ONE clean, complete table ──────────────────────────
    # Columns: Risk Level | Threshold | Sample Score | Load
    # .sp-patient-section anchors the CSS (gap removal + uniform Load button style).
    _SP_DATA = [
        ("crit", "Critical", "#DC2626", "#FEF2F2", "≥ 70%"),
        ("high", "High",     "#EA580C", "#FFF7ED", "50–69%"),
        ("med",  "Medium",   "#D97706", "#FFFBEB", "30–49%"),
        ("low",  "Low",      "#059669", "#F0FDF4", "< 30%"),
    ]
    _SP_W = [2.4, 1.8, 1.8, 1.8]   # column widths: Level | Threshold | Score | Load

    # Wrap everything in ONE full-width column so the .sp-patient-section CSS scope
    # (gap removal + uniform Load-button styling) reliably reaches the nested rows.
    st.markdown('<div id="section-samples"></div>', unsafe_allow_html=True)  # nav anchor
    _sp_wrap = st.columns([1])[0]
    with _sp_wrap:
        st.markdown(
            '<div class="sp-patient-section" '
            'style="font-size:.78rem;font-weight:800;color:#1A3C66;'
            'text-transform:uppercase;letter-spacing:.07em;margin:4px 0 6px 0;">'
            'Sample Patients '
            '<span style="font-size:.64rem;font-weight:500;color:#6B7280;'
            'text-transform:none;letter-spacing:0;">'
            '- real training-data examples · click ▶ Load to autofill the form</span></div>',
            unsafe_allow_html=True)

        # Header row
        _hc = st.columns(_SP_W, gap="small")
        for _i, _htxt in enumerate(["Risk Level", "Threshold", "Sample Score", "Action"]):
            with _hc[_i]:
                st.markdown(f'<div class="sp-th">{_htxt}</div>', unsafe_allow_html=True)

        # Data rows
        for key, name, clr, bg, rng in _SP_DATA:
            _c1, _c2, _c3, _c4 = st.columns(_SP_W, gap="small")

            with _c1:   # Risk level badge
                st.markdown(
                    f'<div class="sp-cell" style="background:{bg};border-left:5px solid {clr};'
                    f'justify-content:flex-start;font-weight:800;color:{clr};font-size:.84rem;">'
                    f'{name}</div>',
                    unsafe_allow_html=True)

            with _c2:   # Threshold range
                st.markdown(
                    f'<div class="sp-cell" style="background:{bg};">'
                    f'<span style="color:{clr};font-weight:700;font-size:.78rem;">{rng}</span></div>',
                    unsafe_allow_html=True)

            with _c3:   # Predicted sample score
                st.markdown(
                    f'<div class="sp-cell" style="background:{bg};">'
                    f'<span style="background:{clr};color:#fff;border-radius:5px;'
                    f'padding:2px 11px;font-size:.78rem;font-weight:800;">'
                    f'{_sample_scores[key]:.0f}%</span></div>',
                    unsafe_allow_html=True)

            with _c4:   # Load button (uniform style via .sp-patient-section CSS)
                if st.button("▶ Load", key=f"sp_{key}", use_container_width=True,
                             help=f"Load {name} sample patient ({rng}) into the form"):
                    st.session_state["_pending_sample"] = SAMPLES[key]
                    st.session_state.patient            = SAMPLES[key]
                    st.session_state.score              = _sample_scores[key]
                    st.rerun()

    # ── SECTION 2 : Assessment form (horizontal card) ─────────────────────────
    st.markdown('<div id="section-form" style="height:18px;"></div>', unsafe_allow_html=True)  # nav anchor + spacing
    st.markdown(
        '<div style="background:#1A3C66;'
        'border-radius:6px 6px 0 0;padding:13px 22px;">'
        '<span style="color:#fff;font-weight:700;font-size:.9rem;">'
        'Patient Assessment Form</span></div>',
        unsafe_allow_html=True)

    # Form area: white background + dark navy text on all widget labels
    st.markdown("""
    <style>
    /* ── Assessment form: widget label text → dark navy ── */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stCheckboxLabel"] p,
    [data-testid="stCheckboxLabel"],
    .stSlider label, .stSlider [data-testid="stWidgetLabel"] p,
    .stSelectbox label, .stSelectbox [data-testid="stWidgetLabel"] p,
    .stCheckbox label, .stCheckbox span,
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
        color: #0D1B3E !important;
        font-weight: 600 !important;
    }
    /* slider value / tick labels */
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"],
    [data-testid="stSlider"] span { color: #0D1B3E !important; }
    /* selectbox selected text */
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] div { color: #0D1B3E !important; }
    /* checkbox label text */
    [data-testid="stCheckbox"] label p,
    [data-testid="stCheckbox"] label span { color: #0D1B3E !important; font-weight: 600 !important; }
    /* button text colours are handled in global CSS block above */
    </style>
    <span id="form-body"></span>
    """, unsafe_allow_html=True)

    # ── Apply any pending sample BEFORE widgets are created ─────────────────────
    if "_pending_sample" in st.session_state:
        _s = st.session_state.pop("_pending_sample")
        st.session_state.il_age            = _s["age"]
        # Data encodes Female=1, Male=0 (matches training)
        st.session_state.il_sex            = "Female" if _s["sex"] == 1 else "Male"
        st.session_state.il_diabetes       = bool(_s["diabetes"])
        st.session_state.il_hypertension   = bool(_s["hypertension"])
        st.session_state.il_cardiovascular = bool(_s["cardiovascular"])
        st.session_state.il_pneumonia      = bool(_s["pneumonia"])
        st.session_state.il_obesity        = bool(_s["obesity"])
        st.session_state.il_asthma         = bool(_s["asthma"])
        st.session_state.il_copd           = bool(_s["copd"])

    # ── 3 columns: Demographics | Conditions (comorbidities + clinical) | Actions ─
    fa, fb, fd = st.columns([1.1, 2.1, 1.6], gap="medium")

    with fa:
        st.markdown(
            '<p style="font-size:.74rem;font-weight:800;color:#1A3C66;'
            'text-transform:uppercase;letter-spacing:.07em;margin:0 0 6px;">Demographics</p>',
            unsafe_allow_html=True)
        age = st.slider("Age (years)", 1, 100, 50, key="il_age")
        sex = st.selectbox("Biological Sex", ["Female", "Male"], key="il_sex")

    with fb:
        # ── Comorbidities ──────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:.74rem;font-weight:800;color:#1A3C66;'
            'text-transform:uppercase;letter-spacing:.07em;margin:0 0 4px;">Comorbidities</p>',
            unsafe_allow_html=True)
        cb1, cb2 = st.columns(2)
        with cb1:
            diabetes       = st.checkbox("Diabetes",       key="il_diabetes")
            cardiovascular = st.checkbox("Cardiovascular", key="il_cardiovascular")
            asthma         = st.checkbox("Asthma",         key="il_asthma")
        with cb2:
            hypertension   = st.checkbox("Hypertension",   key="il_hypertension")
            obesity        = st.checkbox("Obesity",         key="il_obesity")
            copd           = st.checkbox("COPD",            key="il_copd")

        # ── Clinical Status ────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:.74rem;font-weight:800;color:#1A3C66;'
            'text-transform:uppercase;letter-spacing:.07em;margin:10px 0 4px;">Clinical Status</p>',
            unsafe_allow_html=True)
        pneumonia = st.checkbox("Pneumonia Diagnosed", key="il_pneumonia")
        st.markdown(
            '<div style="background:#FEF3C7;border-radius:8px;padding:6px 10px;'
            'font-size:.68rem;color:#78350F;border-left:3px solid #F59E0B;margin-top:8px;">'
            'Note: Research prototype only - not for clinical use.</div>',
            unsafe_allow_html=True)

    with fd:
        # ── Action buttons ─────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:.74rem;font-weight:800;color:#1A3C66;'
            'text-transform:uppercase;letter-spacing:.07em;margin:0 0 6px;">Actions</p>',
            unsafe_allow_html=True)
        st.markdown('<div class="sb-run">', unsafe_allow_html=True)
        run_clicked = st.button("Assess Risk", use_container_width=True, key="il_run")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-reset">', unsafe_allow_html=True)
        reset_clicked = st.button("↺ Reset", use_container_width=True, key="il_reset")
        st.markdown('</div>', unsafe_allow_html=True)

    # Handle button clicks
    if run_clicked:
        # sex encoded Female=1, Male=0 to match the deployed model's training
        pt_new = dict(age=age, sex=1 if sex=="Female" else 0,
                      diabetes=int(diabetes), hypertension=int(hypertension),
                      cardiovascular=int(cardiovascular), pneumonia=int(pneumonia),
                      obesity=int(obesity), asthma=int(asthma), copd=int(copd))
        st.session_state.patient = pt_new
        st.session_state.score   = predict_risk(imp_m, sc_m, model_m, pt_new)
        st.rerun()

    if reset_clicked:
        st.session_state.patient = None
        st.session_state.score   = None
        st.rerun()

    # ── SECTION 3 : Results (shown only after assessment) ─────────────────────
    st.markdown('<div id="section-results"></div>', unsafe_allow_html=True)  # nav anchor
    if pt is None:
        st.markdown(
            '<div style="background:#fff;border-radius:14px;padding:32px;'
            'margin-top:20px;box-shadow:0 2px 12px rgba(26,60,102,.08);'
            'border:1px solid #BFDBFE;text-align:center;">'
            '<div style="font-size:1rem;font-weight:700;color:#0D1B3E;margin-bottom:8px;">'
            'Ready for Long COVID Assessment</div>'
            '<div style="font-size:.85rem;color:#4B6CB7;max-width:520px;margin:0 auto;">'
            'Fill in the patient profile above, then click '
            '<strong>Assess Risk</strong> to generate a personalised '
            'Long COVID risk prediction.<br><br>'
            'Or click a <strong>Sample Patient</strong> button for an instant demo.'
            '</div></div>',
            unsafe_allow_html=True)
    else:
        level = rlevel(score)
        RC    = {"CRITICAL":("#DC2626","rb-critical"),"HIGH":("#EA580C","rb-high"),
                 "MEDIUM":("#D97706","rb-medium"),"LOW":("#059669","rb-low")}
        clr, rbc = RC[level]
        lc   = lc_risks(pt, score)
        urge = urgency(level)
        rec  = recovery(score, pt["age"], pt)

        # marker: start of the printable results region (used by the PDF export)
        st.markdown('<div id="print-start"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="tool-hdr" style="margin-top:20px;">'
            '<div><h2>Patient Risk Assessment Results</h2>'
            '<p>Calibrated logistic-regression prediction · synthetic inputs · '
            'DCU F-REC research-compliant</p>'
            f'</div><span class="live-badge">Assessed · {datetime.now().strftime("%H:%M")}</span></div>',
            unsafe_allow_html=True)

        # Save-as-PDF button - clones ONLY the results region into a clean print
        # window (so the overview/sample/form at the top of the tool are excluded).
        import streamlit.components.v1 as _components
        _components.html("""
<button id="pdfbtn" style="cursor:pointer;background:#FFA700;color:#1A3C66;border:none;
  border-radius:50px;padding:9px 20px;font-size:.82rem;font-weight:800;
  box-shadow:0 3px 10px rgba(255,167,0,.4);font-family:Arial,sans-serif;">
  Save as PDF</button>
<script>
(function(){
  var btn = document.getElementById('pdfbtn');
  btn.addEventListener('click', function(){
    var pdoc = window.parent.document;
    var start = pdoc.getElementById('print-start');
    var end   = pdoc.getElementById('print-end');
    if(!start || !end){ window.parent.print(); return; }
    var vb = pdoc.querySelector('section[data-testid="stMain"] [data-testid="stVerticalBlock"]');
    function topChild(node){ var n = node; while(n && n.parentElement !== vb){ n = n.parentElement; } return n; }
    var s = topChild(start), e = topChild(end);
    if(!s || !e){ window.parent.print(); return; }
    var html = '';
    for(var n = s; n && n !== e; n = n.nextElementSibling){ html += n.outerHTML; }
    // Force any collapsed expanders (e.g. the nomogram) to render open in the PDF.
    html = html.replace(/<details(?![^>]*\bopen\b)/g, '<details open');
    var styles = '';
    pdoc.querySelectorAll('style, link[rel="stylesheet"]').forEach(function(el){ styles += el.outerHTML; });
    var w = window.open('', '_blank');
    w.document.write('<html><head><title>Long COVID Risk Assessment - Result</title>'
      + styles
      + '<style>body{background:#fff;padding:24px;margin:0;}'
      + '.dcu-navbar,.stButton,button,iframe,[data-testid="stToolbar"],[data-testid="stHeader"]{display:none!important;}'
      + '*{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important;}'
      + '.card,.rrw,.mc,figure{break-inside:avoid;}</style></head><body>'
      + html + '</body></html>');
    w.document.close();
    setTimeout(function(){ w.focus(); w.print(); }, 500);
  });
})();
</script>""", height=52)

        # Honest framing of what the score is / isn't (defensibility)
        st.markdown(
            '<div style="background:#FFFBEB;border:1px solid #FDE68A;border-left:4px solid #FFA700;'
            'border-radius:10px;padding:11px 16px;margin:4px 0 14px;display:flex;gap:14px;'
            'flex-wrap:wrap;font-size:.78rem;line-height:1.55;color:#374151;background:#FFFFFF;">'
            '<div style="flex:1;min-width:240px;color:#374151;"><strong style="color:#0D1B3E;">What this is:</strong> '
            'a calibrated, population-level estimate of <em>severe-outcome (mortality-proxy)</em> risk '
            'for screening and triage support.</div>'
            '<div style="flex:1;min-width:240px;color:#374151;"><strong style="color:#0D1B3E;">What this isn\'t:</strong> '
            'a Long COVID diagnosis, an individual certainty, or a substitute for clinical judgement - '
            'a 30% score means ~30 of 100 similar patients experienced the outcome.</div>'
            '</div>',
            unsafe_allow_html=True)

        # ── Patient Profile (entered inputs) - clean summary so the PDF/record
        #    shows exactly what was submitted to the form ─────────────────────
        _profile_fields = [
            ("Age",            f"{pt['age']} yrs"),
            ("Biological Sex", "Female" if pt["sex"] == 1 else "Male"),
            ("Diabetes",       "Yes" if pt["diabetes"] else "No"),
            ("Hypertension",   "Yes" if pt["hypertension"] else "No"),
            ("Cardiovascular", "Yes" if pt["cardiovascular"] else "No"),
            ("Pneumonia",      "Yes" if pt["pneumonia"] else "No"),
            ("Obesity",        "Yes" if pt["obesity"] else "No"),
            ("Asthma",         "Yes" if pt["asthma"] else "No"),
            ("COPD",           "Yes" if pt["copd"] else "No"),
        ]
        _cells = ""
        for _lbl, _val in _profile_fields:
            _yes = _val not in ("No",)
            _vc  = "#1A3C66" if _yes else "#9CA3AF"
            _cells += (
                f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:9px;'
                f'padding:8px 12px;min-width:120px;flex:1;">'
                f'<div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;'
                f'color:#6B7280;font-weight:700;">{_lbl}</div>'
                f'<div style="font-size:.95rem;font-weight:800;color:{_vc};">{_val}</div></div>')
        st.markdown(
            '<div class="card" style="margin-bottom:14px;">'
            '<div style="font-size:.72rem;font-weight:800;color:#2E5C92;text-transform:uppercase;'
            'letter-spacing:.06em;margin-bottom:10px;">Patient Profile - entered inputs</div>'
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;">{_cells}</div></div>',
            unsafe_allow_html=True)

        # ── Tuned-threshold high-risk flag (F1-optimal cut-off 0.26) ─────────
        #    The 0/100 score is prob×100, so the operating threshold is thr×100.
        _thr_pct = mdl_metrics["threshold"] * 100          # 0.26 → 26
        _flag_hi = score >= _thr_pct
        if _flag_hi:
            _fbg, _fbd, _ftx, _fic, _fmsg = ("#FEF2F2", "#DC2626", "#7F1D1D", "(!)",
                "<strong>HIGH RISK</strong> - at or above the F1-optimal screening "
                f"threshold ({_thr_pct:.0f}/100). Recommend early clinical review.")
        else:
            _fbg, _fbd, _ftx, _fic, _fmsg = ("#ECFDF5", "#059669", "#065F46", "(✓)",
                "<strong>Lower risk</strong> - below the F1-optimal screening "
                f"threshold ({_thr_pct:.0f}/100). Routine pathway as scheduled.")
        st.markdown(
            f'<div style="background:{_fbg};border:1px solid {_fbd};border-left:5px solid {_fbd};'
            f'border-radius:11px;padding:12px 18px;margin-bottom:12px;display:flex;align-items:center;'
            f'gap:12px;color:{_ftx};font-size:.9rem;">'
            f'<span style="font-size:1.4rem;line-height:1;">{_fic}</span>'
            f'<span style="line-height:1.5;">{_fmsg}</span></div>',
            unsafe_allow_html=True)

        # ── Calibration trust note (model probabilities are calibrated) ──────
        st.markdown(
            f'<div style="font-size:.74rem;color:#6B7280;margin:-4px 0 14px;line-height:1.55;'
            f'border-left:3px solid #4AC9E3;padding:4px 0 4px 12px;">'
            f'<strong style="color:#1A3C66;">Calibrated risk</strong> '
            f'(Brier {mdl_metrics["ens"]["brier"]:.3f}, ECE {mdl_metrics["ens"]["ece"]:.4f}) - '
            f'a {score:.0f}% estimate corresponds to roughly {score:.0f} severe outcomes per 100 '
            f'similar patients, not an arbitrary index.</div>',
            unsafe_allow_html=True)

        # ROW 1 - Score ring + Key metrics
        cr, ck = st.columns([1, 2], gap="large")
        with cr:
            st.markdown(f"""
            <div class="rrw">
                <div style="font-size:.69rem;font-weight:700;letter-spacing:.1em;
                            text-transform:uppercase;color:#4B6CB7;margin-bottom:7px;">
                    Mortality Risk Score
                </div>
                <div style="position:relative;display:inline-block;margin:7px 0;">
                    <svg width="182" height="182" viewBox="0 0 182 182">
                        <circle cx="91" cy="91" r="74" fill="none" stroke="#DBEAFE" stroke-width="14"/>
                        <circle cx="91" cy="91" r="74" fill="none" stroke="{clr}" stroke-width="14"
                                stroke-linecap="round" class="ring-arc"
                                style="--circ:{2*3.14159*74:.1f};--off:{2*3.14159*74*(1-score/100):.1f};
                                       stroke-dasharray:{2*3.14159*74:.1f};
                                       transform:rotate(-90deg);transform-origin:91px 91px;"/>
                    </svg>
                    <div style="position:absolute;top:50%;left:50%;
                                transform:translate(-50%,-50%);text-align:center;">
                        <div class="rn" style="color:{clr};">{score:.0f}</div>
                        <div class="rd">/ 100</div>
                    </div>
                </div>
                <div style="margin-top:9px;">
                    <span class="rb {rbc}">{level} RISK</span>
                </div>
                <div style="margin-top:11px;font-size:.8rem;color:#374151;font-weight:500;">{urge}</div>
                <div style="margin-top:4px;font-size:.76rem;color:#6B7280;">Recovery: {rec}</div>
            </div>
            """, unsafe_allow_html=True)

        with ck:
            conds = sum([pt[k] for k in ["diabetes","hypertension","cardiovascular","obesity","asthma","copd"]])
            kc1, kc2 = st.columns(2)
            for i, (lbl, val, acc) in enumerate([
                ("Age",                  f"{pt['age']} yrs",                  "#2E5C92"),
                ("Biological Sex",       "Female" if pt["sex"]==1 else "Male", "#7C3AED"),
                ("Active Comorbidities", f"{conds}/6",                         "#D97706"),
                ("Pneumonia",            "Yes" if pt["pneumonia"] else "No",   "#DC2626"),
            ]):
                with (kc1 if i%2==0 else kc2):
                    st.markdown(
                        f'<div class="mc" style="border-left-color:{acc};">'
                        f'<div class="ml">{lbl}</div>'
                        f'<div class="mv" style="color:{acc};">{val}</div></div>',
                        unsafe_allow_html=True)

            st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
            cmap = {"diabetes":"Diabetes","hypertension":"Hypertension",
                    "cardiovascular":"Cardiovascular","obesity":"Obesity",
                    "asthma":"Asthma","copd":"COPD","pneumonia":"Pneumonia"}
            active = [v for k,v in cmap.items() if pt.get(k)]
            if active:
                chips = "".join([f'<span class="chip">{c}</span>' for c in active])
                st.markdown(
                    f'<div class="card"><div style="font-size:.72rem;font-weight:700;'
                    f'color:#1E40AF;text-transform:uppercase;margin-bottom:7px;">'
                    f'Active Conditions</div><div>{chips}</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="background:#D1FAE5;border-radius:10px;padding:12px;'
                    'font-size:.82rem;color:#064E3B;font-weight:600;border:2px solid #6EE7B7;">'
                    'No active comorbidities or clinical flags.</div>',
                    unsafe_allow_html=True)

        # ── Per-patient explanation (local attribution) ──────────────────────
        _raises, _lowers = patient_factors(imp_m, sc_m, pt)
        _allc = _raises + _lowers
        _mx   = max([abs(c) for _, c in _allc], default=1.0) or 1.0

        def _factor_rows(items, up):
            _c = "#DC2626" if up else "#059669"
            _bg = "#FEE2E2" if up else "#D1FAE5"
            if not items:
                return ('<div style="font-size:.78rem;color:#6B7280;font-style:italic;">'
                        'None notable for this patient.</div>')
            _h = ""
            for lbl, val in items:
                _w = min(100, abs(val) / _mx * 100)
                _h += (
                    f'<div style="margin-bottom:7px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:.78rem;'
                    f'font-weight:700;margin-bottom:2px;"><span style="color:#0D1B3E;">{lbl}</span>'
                    f'<span style="color:{_c};">{"+" if up else "−"}{abs(val):.2f}</span></div>'
                    f'<div style="background:{_bg};border-radius:5px;height:9px;overflow:hidden;">'
                    f'<div class="anim-bar" style="width:{_w:.0f}%;height:100%;background:{_c};border-radius:5px;"></div>'
                    f'</div></div>')
            return _h

        st.markdown(
            '<div class="card" style="margin-top:6px;">'
            '<div class="card-title" style="margin-bottom:4px;">Why this score? '
            '<span style="font-size:.68rem;color:#6B7280;font-weight:500;text-transform:none;'
            'letter-spacing:0;">(This patient\'s top factors - local log-odds attribution)</span></div>'
            '<div style="display:flex;gap:26px;flex-wrap:wrap;margin-top:10px;">'
            '<div style="flex:1;min-width:240px;">'
            '<div style="font-size:.74rem;font-weight:800;color:#991B1B;margin-bottom:8px;">'
            'Risk-increasing factors</div>' + _factor_rows(_raises, True) + '</div>'
            '<div style="flex:1;min-width:240px;">'
            '<div style="font-size:.74rem;font-weight:800;color:#065F46;margin-bottom:8px;">'
            'Risk-decreasing factors</div>' + _factor_rows(_lowers, False) + '</div>'
            '</div>'
            '<div style="font-size:.71rem;color:#6B7280;margin-top:10px;border-top:1px solid #EFF2F7;'
            'padding-top:8px;line-height:1.5;">Contribution = ln(odds ratio per +1 SD) × this patient\'s '
            'standardised value. An interpretable approximation of the model\'s logit - directionally '
            'faithful, not a full Kernel/Tree-SHAP run.</div>'
            '</div>',
            unsafe_allow_html=True)

        # ── Nomogram: transparent manual "show-your-work" score (illustrative) ─
        _nomo = nomogram_score(pt)
        if _nomo is not None:
            _npts, _nrisk, _nbd = _nomo
            _names = {"pneumonia":"Pneumonia","age":"Age","sex":"Sex (Female)","diabetes":"Diabetes",
                      "copd":"COPD","obesity":"Obesity","hypertension":"Hypertension",
                      "cardiovascular":"Cardiovascular","asthma":"Asthma"}
            with st.expander("Nomogram - transparent manual score (illustrative, not the official prediction)"):
                _chips = ""
                for _f, _p in sorted(_nbd, key=lambda x: -abs(x[1])):
                    _pc = "#DC2626" if _p > 0 else "#059669"
                    _chips += (f'<span style="display:inline-block;background:#F8FAFC;border:1px solid #E2E8F0;'
                               f'border-radius:7px;padding:3px 10px;margin:0 6px 6px 0;font-size:.78rem;'
                               f'color:#0D1B3E;">{_names.get(_f,_f)} '
                               f'<strong style="color:{_pc};">{"+" if _p>0 else ""}{_p}</strong></span>')
                st.markdown(
                    '<div style="font-size:.82rem;color:#374151;line-height:1.6;margin-bottom:12px;">'
                    'A classic points-based <strong>nomogram</strong> built from the logistic-regression '
                    'coefficients - it shows <em>how</em> a risk estimate is assembled by hand. '
                    'The app\'s <strong>official</strong> risk is always the calibrated-model number above; '
                    'this manual score is for transparency only and may differ slightly.</div>'
                    '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px;">'
                    '<div style="flex:1;min-width:160px;background:linear-gradient(135deg,#1A3C66,#2E5C92);'
                    'border-radius:12px;padding:14px 18px;color:#fff;">'
                    '<div style="font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;'
                    'color:rgba(255,255,255,.8);">Total nomogram points</div>'
                    f'<div style="font-size:1.9rem;font-weight:900;line-height:1.1;">{_npts}</div></div>'
                    '<div style="flex:1;min-width:160px;background:#FFFBEB;border:1px solid #FDE68A;'
                    'border-radius:12px;padding:14px 18px;">'
                    '<div style="font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:#92400E;">'
                    'Nomogram risk (illustrative)</div>'
                    f'<div style="font-size:1.9rem;font-weight:900;color:#B45309;line-height:1.1;">{_nrisk}%</div>'
                    f'<div style="font-size:.72rem;color:#6B7280;margin-top:2px;">Model (official): '
                    f'<strong style="color:#0D1B3E;">{score:.1f}%</strong></div></div>'
                    '</div>'
                    f'<div style="font-size:.74rem;font-weight:700;color:#2E5C92;text-transform:uppercase;'
                    f'letter-spacing:.05em;margin-bottom:6px;">This patient\'s points</div>'
                    f'<div>{_chips if _chips else "<em style=\'color:#6B7280;font-size:.8rem;\'>No points contributed.</em>"}</div>',
                    unsafe_allow_html=True)
                # full points chart so clinicians can see the scoring rule.
                # Rendered as an HTML table (not st.dataframe) so it is captured
                # by the Save-as-PDF clone - the dataframe grid is a canvas that
                # does not clone into the print window.
                _nomo_tbl, _ = load_nomogram()
                if _nomo_tbl is not None:
                    _rows_html = ""
                    for _, _tr in _nomo_tbl.iterrows():
                        _fname = _names.get(str(_tr["feature"]), str(_tr["feature"]))
                        _pv    = int(_tr["points"])
                        _pc    = "#DC2626" if _pv > 0 else ("#059669" if _pv < 0 else "#6B7280")
                        _rows_html += (
                            f'<tr>'
                            f'<td style="padding:6px 12px;border-bottom:1px solid #EFF2F7;color:#0D1B3E;">{_fname}</td>'
                            f'<td style="padding:6px 12px;border-bottom:1px solid #EFF2F7;color:#6B7280;">{_tr["unit"]}</td>'
                            f'<td style="padding:6px 12px;border-bottom:1px solid #EFF2F7;text-align:right;'
                            f'font-weight:800;color:{_pc};">{"+" if _pv>0 else ""}{_pv}</td>'
                            f'</tr>')
                    st.markdown(
                        '<div style="font-size:.74rem;font-weight:700;color:#2E5C92;'
                        'text-transform:uppercase;letter-spacing:.05em;margin:14px 0 4px;">'
                        'Nomogram points chart</div>'
                        '<table style="width:100%;border-collapse:collapse;font-size:.8rem;'
                        'background:#fff;border:1px solid #E2E8F0;border-radius:8px;overflow:hidden;">'
                        '<thead><tr style="background:#F1F5F9;">'
                        '<th style="padding:7px 12px;text-align:left;color:#2E5C92;font-weight:800;">Feature</th>'
                        '<th style="padding:7px 12px;text-align:left;color:#2E5C92;font-weight:800;">Unit</th>'
                        '<th style="padding:7px 12px;text-align:right;color:#2E5C92;font-weight:800;">Points</th>'
                        '</tr></thead><tbody>' + _rows_html + '</tbody></table>',
                        unsafe_allow_html=True)

        st.markdown("<hr class='div'>", unsafe_allow_html=True)

        # ROW 2 - Sequelae + Risk factor impact
        cl, cf = st.columns([3, 2], gap="large")
        with cl:
            st.markdown(
                '<div class="card-title">Long COVID Sequelae Risk Assessment '
                '<span style="font-size:.68rem;color:#6B7280;font-weight:500;'
                'text-transform:none;letter-spacing:0;">'
                '(Heuristic estimates - not direct model outputs)</span></div>',
                unsafe_allow_html=True)
            for cat, pct in lc.items():
                m   = LC_META[cat]; a = bclr(pct)
                sev = ("Very Low" if pct<20 else "Low" if pct<40 else
                       "Moderate" if pct<60 else "High" if pct<80 else "Critical")
                syms = ""
                if pct >= 50:
                    syms = ('<br><div style="margin-top:5px;display:flex;flex-wrap:wrap;gap:4px;">'
                            + "".join([f'<span style="background:#fff;border:2px solid {a}55;'
                                       f'color:{a};border-radius:7px;padding:2px 9px;'
                                       f'font-size:.71rem;font-weight:600;">{s}</span>'
                                       for s in m["syms"]]) + "</div>")
                st.markdown(
                    f'<div style="background:#fff;border-radius:11px;padding:12px 14px;'
                    f'margin-bottom:10px;border:2px solid #BFDBFE;'
                    f'box-shadow:0 1px 6px rgba(26,60,102,.06);">'
                    f'<div class="lbl"><span>{m["icon"]} <b>{cat}</b></span>'
                    f'<span style="color:{a};font-weight:800;">{pct:.0f}% - {sev}</span></div>'
                    f'<div class="lbb"><div class="lbf" style="width:{pct}%;'
                    f'background:linear-gradient(90deg,{a}aa,{a});"></div></div>'
                    f'{syms}</div>',
                    unsafe_allow_html=True)

        with cf:
            st.markdown('<div class="card-title">Risk Factor Impact (Odds Ratios)</div>',
                        unsafe_allow_html=True)
            # This patient's PRESENT risk factors, ranked by odds ratio (per +1 SD).
            _ors = load_odds_ratios()
            _present = [("Age", _ors.get("age", {}).get("or", 2.33))]
            if pt["sex"] == 0:   # Male is the higher-risk direction (female OR 0.80)
                _present.append(("Male sex", 1.0 / _ors.get("sex", {}).get("or", 0.80)))
            for _f, _lab in [("pneumonia","Pneumonia"),("diabetes","Diabetes"),
                             ("hypertension","Hypertension"),("obesity","Obesity"),
                             ("copd","COPD"),("cardiovascular","Cardiovascular"),
                             ("asthma","Asthma")]:
                if pt.get(_f):
                    _present.append((_lab, _ors.get(_f, {}).get("or", 1.0)))
            _present.sort(key=lambda t: t[1], reverse=True)
            _maxor = max(o for _, o in _present) if _present else 1.0
            for _lab, _or in _present:
                _w = min(100, _or / _maxor * 100)
                st.markdown(
                    f'<div class="fr"><div class="fn">{_lab}</div>'
                    f'<div class="fbb"><div class="fbf" style="width:{_w:.0f}%;"></div></div>'
                    f'<div class="fv">{_or:.2f}×</div></div>',
                    unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:.68rem;color:#6B7280;margin-top:6px;line-height:1.5;">'
                'Odds ratio = multiplicative effect on mortality odds per +1 SD '
                '(from logistic regression). Only this patient’s present factors shown.</div>',
                unsafe_allow_html=True)

        st.markdown("<hr class='div'>", unsafe_allow_html=True)

        # ROW 3+4 merged - LEFT: Timeline + Clinical Guidance | RIGHT: Referrals + Red Flags + Model
        cleft, cright = st.columns([3, 2], gap="large")

        with cleft:
            # Care Timeline
            st.markdown('<div class="card-title">Personalised Care Timeline</div>',
                        unsafe_allow_html=True)
            steps = TIMELINES.get(level, TIMELINES["MEDIUM"])
            tl = "".join([
                f'<div class="ti"><div class="td" style="background:{dc};">{ph}</div>'
                f'<div><div class="tt">{t}</div><div class="tx">{x}</div></div></div>'
                for dc,ph,t,x in steps])
            st.markdown(
                f'<div class="card">'
                f'<div style="font-size:.8rem;color:#374151;font-weight:500;margin-bottom:12px;">'
                f'⏱️ Estimated recovery: <strong>{rec}</strong></div>'
                f'{tl}'
                f'<div style="margin-top:8px;font-size:.72rem;color:#6B7280;font-style:italic;">'
                f'Guidance only. Medications must be prescribed by a qualified clinician.</div>'
                f'</div>',
                unsafe_allow_html=True)

            # Evidence-Based Clinical Guidance (below timeline - balances right column height)
            st.markdown('<div class="card-title" style="margin-top:14px;">Evidence-Based Clinical Guidance</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="card" style="font-size:.83rem;line-height:1.75;color:#374151;">'
                '<strong style="color:#0D1B3E;">WHO Post-COVID (2022):</strong> Typically within '
                '3 months of acute infection, lasting 2 months or more.<br><br>'
                '<strong style="color:#0D1B3E;">CDC PASC:</strong> Assess ongoing symptoms and '
                'provide specialist referrals guided by risk stratification.<br><br>'
                '<strong style="color:#0D1B3E;">Modifiable Risk Factors:</strong>'
                '<div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:9px;">'
                '<span class="chip">Vaccination</span>'
                '<span class="chip">Diabetes control</span>'
                '<span class="chip">Blood pressure</span>'
                '<span class="chip">Weight management</span>'
                '<span class="chip">Smoking cessation</span>'
                '<span class="chip">Rehabilitation</span>'
                '</div></div>',
                unsafe_allow_html=True)

        with cright:
            # Specialist Referrals
            st.markdown('<div class="card-title">Specialist Referrals</div>',
                        unsafe_allow_html=True)
            hcats = ([c for c,p in lc.items() if p>=40]
                     or sorted(lc, key=lambda k: lc[k], reverse=True)[:2])
            # Cap at top 3 to avoid duplication overload for CRITICAL patients
            hcats = hcats[:3]
            surg  = {"CRITICAL":"Within 24-48 hrs","HIGH":"Within 1 week",
                     "MEDIUM":"Within 2 weeks","LOW":"Within 1 month"}[level]
            for cat in hcats:
                m  = LC_META[cat]; pct = lc[cat]; a = bclr(pct)
                sc2 = "".join([f'<span class="chip">{s}</span>' for s in m["specs"]])
                st.markdown(
                    f'<div style="background:#fff;border-radius:11px;padding:11px 14px;'
                    f'margin-bottom:9px;border:2px solid #BFDBFE;'
                    f'box-shadow:0 1px 6px rgba(26,60,102,.06);">'
                    f'<div style="font-weight:700;font-size:.84rem;color:{a};margin-bottom:5px;">'
                    f'{cat} ({pct:.0f}%)</div>'
                    f'<div>{sc2}</div>'
                    f'<div style="font-size:.72rem;color:#6B7280;margin-top:4px;font-weight:500;">'
                    f'{surg}</div></div>',
                    unsafe_allow_html=True)

            # Emergency Red Flags
            st.markdown('<div class="card-title" style="margin-top:14px;">Emergency Red Flags</div>',
                        unsafe_allow_html=True)
            rf_html = "".join([f"<p style='margin:4px 0;font-size:.81rem;color:#7F1D1D;'>{f}</p>" for f in [
                "Sudden chest pain or pressure",
                "Severe shortness of breath at rest",
                "Confusion, slurred speech, facial drooping",
                "Heart rate above 120 bpm at rest",
                "Oxygen saturation below 94%",
                "Sudden severe headache",
            ]])
            st.markdown(
                f'<div class="rfb"><div style="font-weight:700;font-size:.84rem;'
                f'color:#7F1D1D;margin-bottom:6px;">Seek emergency care immediately:</div>'
                f'{rf_html}</div>',
                unsafe_allow_html=True)

            # Model Information
            st.markdown('<div class="card-title" style="margin-top:14px;">Model Information</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="card" style="font-size:.82rem;line-height:1.75;color:#374151;">'
                '<strong style="color:#0D1B3E;">Model:</strong> Calibrated tuned Logistic Regression (9 features)<br>'
                f'<strong style="color:#0D1B3E;">Training:</strong> {mdl_metrics["n_train"]:,} confirmed COVID cases<br>'
                f'<strong style="color:#0D1B3E;">AUC:</strong> {mdl_metrics["ens"]["auc"]:.3f} · Brier: {mdl_metrics["ens"]["brier"]:.3f} · ECE: {mdl_metrics["ens"]["ece"]:.4f}<br>'
                '<strong style="color:#0D1B3E;">Fairness:</strong> Gender gap ~2.8% (within threshold); age 50+, diabetes subgroup flagged<br>'
                '<strong style="color:#0D1B3E;">Supervisors:</strong> Dr Martin Crane &amp; Dr Tai Tan Mai, DCU<br><br>'
                '<div style="background:#FEF3C7;border-radius:7px;padding:9px 12px;'
                'font-size:.76rem;color:#78350F;border-left:3px solid #F59E0B;font-weight:600;">'
                'Note: Research prototype only. Not for clinical use.</div>'
                '</div>',
                unsafe_allow_html=True)

        # marker: end of the printable results region (footer below is excluded)
        st.markdown('<div id="print-end"></div>', unsafe_allow_html=True)

    render_contact_footer()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER - URL-driven (?page=tool) so the HTML nav links work on any device
# ─────────────────────────────────────────────────────────────────────────────
_qp_page = st.query_params.get("page", "")
st.session_state.page = "tool" if _qp_page == "tool" else "landing"
if st.session_state.page == "tool":
    page_tool()
else:
    page_landing()
