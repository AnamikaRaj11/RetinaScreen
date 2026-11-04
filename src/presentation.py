"""Visual system for the presentation workspace; no inference logic."""
import base64
from html import escape

from src.config import DATA_DIR, CLASS_LABELS


CSS = """
:root {--ink:#172e45; --muted:#677b8d; --teal:#087d78; --line:#e0e7ed; --paper:#f3f6f9;}
body {background:var(--paper) !important;}
.gradio-container {width:100% !important; max-width:1500px !important; box-sizing:border-box !important; padding:24px clamp(14px,2.8vw,40px) 36px !important; margin:auto !important; font-family:'Aptos','Segoe UI',sans-serif !important;}
.gradio-container .main {padding:0 !important;}
.gradio-container .prose {line-height:1.65; color:var(--ink);}
.gradio-container .prose p {max-width:78ch;}
.gradio-container .prose h2 {font-weight:650; letter-spacing:-.8px;}
.brand-bar {display:flex; align-items:center; justify-content:space-between; padding:4px 0 23px; border-bottom:1px solid #dbe4eb; gap:16px;}
.brand {display:flex; align-items:center; gap:12px; color:var(--ink);}
.brand-symbol {width:43px; height:43px; background:var(--ink); color:#fff; border-radius:13px; display:grid; place-items:center;}
.brand-symbol svg {width:29px; height:29px;}
.brand-symbol path,.brand-symbol circle {stroke:#fff !important;}
.brand-name {font-size:23px; letter-spacing:-.8px; font-weight:750; line-height:1.05;}
.brand-caption {font-size:12px; color:var(--muted); margin-top:5px;}
.research-tag {border:1px solid #d5e0e7; background:#fff; color:#496075; border-radius:30px; padding:8px 13px; font-size:12px; display:flex; align-items:center; gap:8px;}
.status-dot {width:7px; height:7px; display:inline-block; background:#209688; border-radius:50%;}
.page-title {display:flex; justify-content:space-between; align-items:end; margin:21px 0 16px; gap:20px;}
.page-title h1 {font-size:30px; font-weight:650; line-height:1.15; color:var(--ink); margin:0 0 8px; letter-spacing:-1px;}
.page-title p {margin:0; font-size:14px; color:var(--muted);}
.page-meta {color:#718495; font-size:12px; text-align:right; padding-bottom:3px;}
#section-navigation {gap:8px; padding:0 0 13px; border-bottom:1px solid #dde5eb; margin-bottom:16px;}
#section-navigation button {flex:0 1 auto; min-width:105px; border:0; background:transparent; color:#486075; box-shadow:none; font-size:13px; font-weight:600; padding:11px 18px; border-radius:8px;}
#section-navigation button:hover {background:#e5eeef; color:#056d69;}
#input-panel {background:#fff; border:1px solid var(--line); border-radius:14px; padding:21px; gap:13px; box-shadow:0 4px 18px #213c5410;}
#input-panel .prose h3 {font-size:19px; margin:0 0 5px; color:var(--ink); letter-spacing:-.35px;}
#input-panel .prose p {font-size:13px; color:var(--muted); margin:0; line-height:1.55;}
#input-panel #input-image {border:1px dashed #bdced7; border-radius:10px; background:#f7fafb;}
#sample-buttons {gap:6px;}
#sample-buttons button {min-width:0 !important; flex:1 1 0; padding:8px 4px; font-size:11px; white-space:nowrap; border-radius:7px; background:#f2f6f8; border:1px solid #e1e8ee; box-shadow:none; color:#425d71;}
#screen-button {height:48px; font-size:15px; font-weight:650; background:#087d78; color:white; border-radius:9px; box-shadow:0 3px 9px #087d7820; border:0;}
#screen-button:hover {background:#056c68;}
#input-help .prose p {font-size:11px; color:#82909c; line-height:1.5;}
#workspace {background:#fff; border:1px solid var(--line); border-radius:14px; padding:22px 26px; gap:17px; min-width:0 !important; box-shadow:0 4px 18px #213c5408;}
#workspace-selector {background:#f5f8fa; border-radius:9px; border:1px solid #e7edf1;}
#tutorial-panels > .tab-wrapper, #tutorial-panels > .tab-nav, #tutorial-panels [role="tablist"] {display:none !important;}
#tutorial-panels {border:0; padding:0;}
#tutorial-panels > .tabitem {padding:0; border:0;}
.empty-screen {display:grid; grid-template-columns:minmax(0,1fr) 235px; align-items:center; gap:18px; padding:25px 2px 28px;}
.empty-screen h2 {font-size:37px !important; font-weight:620 !important; line-height:1.12 !important; letter-spacing:-1.5px !important; margin:0 0 18px !important; color:var(--ink); max-width:400px;}
.empty-screen p {font-size:14px; color:var(--muted); line-height:1.7; max-width:330px; margin:0 0 20px;}
.input-prompt {display:inline-flex; align-items:center; gap:10px; font-size:12px; color:#087d78; background:#edf7f5; border:1px solid #d6ebe6; border-radius:7px; padding:9px 12px;}
.retina-figure {position:relative; margin:0; text-align:center; padding:14px 0;}
.retina-frame {width:210px; height:210px; border-radius:50%; background:#0c1420; border:9px solid #f0f4f6; box-shadow:0 0 0 1px #dbe4eb; margin:auto; overflow:hidden;}
.retina-frame img {width:100%; height:100%; object-fit:cover; display:block;}
.retina-figure figcaption {font-size:10px; color:#7c8b98; margin-top:13px;}
.workflow-strip {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:19px 0; gap:15px;}
.workflow-item {display:flex; gap:10px; align-items:flex-start;}
.step-index {font-size:11px; color:#087d78; background:#ecf5f3; padding:4px 7px; border-radius:5px; font-weight:650;}
.workflow-item strong {display:block; font-size:12px; color:#304e63; font-weight:600; margin-bottom:3px;}
.workflow-item small {display:block; font-size:11px; color:#7c8d9c; line-height:1.4;}
.scope-note {display:flex; gap:10px; align-items:baseline; justify-content:space-between; margin:19px 0 9px; font-size:11px; color:#758898;}
.scope-note strong {font-weight:600; color:#4a6579;}
.tutorial-intro {padding:3px 0 17px; margin-bottom:16px; border-bottom:1px solid var(--line);}
.tutorial-intro h2 {font-size:26px; font-weight:650; letter-spacing:-.7px; color:var(--ink); margin:0 0 8px;}
.tutorial-intro .tutorial-number {font-size:12px; font-weight:650; color:#13827c; margin-bottom:8px;}
.tutorial-intro p {font-size:14px; color:var(--muted); margin:0; max-width:72ch;}
.result-header {display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); padding:13px 0 20px; gap:20px;}
.result-header .result-label {font-size:12px; color:#61798c; margin-bottom:10px;}
.result-header h2 {font-size:31px !important; margin:0 0 8px !important; line-height:1.15; letter-spacing:-1px;}
.result-header p {font-size:13px; color:var(--muted); margin:0;}
.grade-badge {border:1px solid #d9e4e7; background:#f4f8f9; border-radius:12px; width:85px; height:85px; flex-shrink:0; display:flex; flex-direction:column; align-items:center; justify-content:center;}
.grade-badge b {font-size:35px; line-height:1.1; font-weight:650; color:var(--ink);}
.grade-badge span {font-size:10px; color:var(--muted);}
.result-metrics {display:grid; grid-template-columns:1fr 1fr; gap:0; border-bottom:1px solid var(--line); padding:18px 0;}
.result-metrics > div {padding:0 15px; border-left:1px solid var(--line);}
.result-metrics > div:first-child {padding-left:0; border:0;}
.result-metrics span {display:block; font-size:12px; color:var(--muted); margin-bottom:7px;}
.result-metrics strong {font-size:23px; font-weight:620; color:var(--ink);}
.result-action {background:#f0f7f6; border-left:3px solid #168c7e; border-radius:0 8px 8px 0; padding:15px 18px; margin:19px 0 14px;}
.result-action span {font-size:11px; color:#4e7c75; display:block; margin-bottom:5px;}
.result-action strong {font-size:15px; color:#185d56; font-weight:650;}
.review-callout {background:#fff8e9; border:1px solid #f0e2bd; padding:11px 14px; border-radius:7px; color:#826020; font-size:12px; margin-bottom:13px;}
.result-footnote {color:#80909c; font-size:11px; line-height:1.6; margin:14px 0;}
.presentation-footer {display:flex; justify-content:space-between; border-top:1px solid #dbe4eb; padding:17px 0 0; color:#8a98a4; font-size:11px; margin-top:24px; gap:15px;}
button:focus-visible, input:focus-visible {outline:3px solid #68b8ae !important; outline-offset:3px;}
footer {display:none !important;}
@media(min-width:1100px) {#input-panel {position:sticky; top:18px; align-self:flex-start;}}
@media(max-width:1050px) {.gradio-container {padding:20px !important;} .empty-screen {grid-template-columns:1fr 170px;} .retina-frame {width:160px; height:160px;} .empty-screen h2 {font-size:30px !important;} #workspace {padding:20px;}}
@media(max-width:760px) {#analysis-layout {flex-direction:column !important;} #input-panel,#workspace {flex-basis:auto !important; flex-grow:0 !important; box-sizing:border-box !important;} .gradio-container {padding:14px !important;} .gradio-container .main,.gradio-container .wrap,.gradio-container main {min-width:0 !important; max-width:100% !important;} #input-panel,#workspace {min-width:0 !important; width:100% !important; position:static; padding:18px;} .brand-bar {padding-bottom:16px;} .brand-name {font-size:21px;} .research-tag {font-size:10px; padding:6px 9px;} .brand-caption {font-size:10px;} .page-title h1 {font-size:25px;} .page-meta {display:none;} .page-title {margin:16px 0;} #section-navigation {flex-wrap:wrap; gap:3px;} #section-navigation button {min-width:85px; padding:8px 10px; font-size:12px;} .empty-screen {grid-template-columns:1fr; padding:18px 0; gap:5px;} .empty-screen h2 {font-size:30px !important; max-width:100%;} .empty-screen p {max-width:100%;} .retina-figure {padding:16px 0;} .retina-frame {width:180px; height:180px;} .workflow-strip {grid-template-columns:1fr; gap:14px;} .scope-note {flex-direction:column; gap:5px;} .presentation-footer {flex-direction:column; gap:7px;} .result-header h2 {font-size:25px !important;} .grade-badge {width:68px; height:68px;} .result-metrics strong {font-size:19px;}}
@media(prefers-reduced-motion:reduce) {* {animation:none !important; transition:none !important; scroll-behavior:auto !important;}}
"""


def masthead():
    return '''<div class="brand-bar"><div class="brand"><div class="brand-symbol"><svg viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="M3 16s4.5-9 13-9 13 9 13 9-4.5 9-13 9S3 16 3 16Z" stroke="currentColor" stroke-width="1.7"/><circle cx="16" cy="16" r="5" stroke="currentColor" stroke-width="1.7"/><circle cx="16" cy="16" r="1.8" fill="#78d5be"/></svg></div><div><div class="brand-name">RetinaScreen</div><div class="brand-caption">Retinal imaging &amp; decision support</div></div></div><div class="research-tag"><span class="status-dot"></span>Research edition</div></div><div class="page-title"><div><h1>Retinal screening workspace</h1><p>Inspect the image. Understand the prediction. Explore the evidence.</p></div><div class="page-meta">APTOS 2019<br>15 connected tutorials</div></div>'''


def ready_screen():
    sample = DATA_DIR / 'processed' / 'train_224' / '002c21358ce6.png'
    uri = 'data:image/png;base64,' + base64.b64encode(sample.read_bytes()).decode('ascii')
    return f'''<section class="empty-screen"><div><h2>Every retinal image<br>deserves a closer look.</h2><p>Review diabetic retinopathy severity with the trained image model, then explore preprocessing, visual explanations and the supporting analyses.</p><div class="input-prompt">Select a sample or upload a fundus photograph</div></div><figure class="retina-figure"><div class="retina-frame"><img src="{uri}" alt="Example retinal fundus photograph from the APTOS dataset"/></div><figcaption>APTOS sample / illustrative preview</figcaption></figure></section><div class="workflow-strip"><div class="workflow-item"><span class="step-index">01</span><div><strong>Prepare the image</strong><small>Quality checks &amp; normalization</small></div></div><div class="workflow-item"><span class="step-index">02</span><div><strong>Review the prediction</strong><small>Severity &amp; model confidence</small></div></div><div class="workflow-item"><span class="step-index">03</span><div><strong>Explore the evidence</strong><small>Attention maps &amp; tutorial results</small></div></div></div><div class="scope-note"><strong>One workflow, fifteen tutorials</strong><span>Images, clinical text, planning &amp; behavioral analysis</span></div>'''


def render_screening_summary(state):
    grade = state['pred_class']
    rule = state['rule']
    quality = 'Passed' if state['passed_quality'] else 'Review required'
    review = '<div class="review-callout">Low confidence or a close runner-up. A second-reader review is advised.</div>' if state['is_borderline'] else ''
    return f'''<div class="result-header"><div><div class="result-label">Image model result</div><h2>{escape(CLASS_LABELS[grade])}</h2><p>Five-class diabetic retinopathy screening</p></div><div class="grade-badge"><b>{grade}</b><span>ICDR grade</span></div></div><div class="result-metrics"><div><span>Model confidence</span><strong>{state['confidence_pct']:.1f}%</strong></div><div><span>Image quality</span><strong>{quality}</strong></div></div><div class="result-action"><span>Template referral recommendation</span><strong>{escape(rule['action'])}</strong><div style="font-size:12px;color:#547d77;margin-top:7px">Target window: {escape(rule['timeframe'])}</div></div>{review}<div class="result-footnote">{escape(state['quality_status'])}<br>Educational screening result. Recommendations require clinician review; classification does not confirm individual lesions.</div>'''
