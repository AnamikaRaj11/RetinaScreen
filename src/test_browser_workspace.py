"""Optional browser smoke check; start app.py first and install playwright + Chromium.
Run from the project root: .venv/Scripts/python.exe src/test_browser_workspace.py
"""
from playwright.sync_api import sync_playwright, expect, TimeoutError as PlaywrightTimeout
import re
from pathlib import Path
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    page.goto('http://127.0.0.1:7860',wait_until='domcontentloaded',timeout=120000)
    page.get_by_role('button',name='Run screening',exact=True).wait_for(timeout=120000)
    try:
        page.wait_for_load_state('networkidle', timeout=10000)
    except PlaywrightTimeout:
        page.locator('#screen-button').wait_for(state='visible')
    page.screenshot(path='outputs/figures/ui_tests/workspace_desktop.png',full_page=True)
    print('TABS',page.get_by_role('tab').count(),'ERRORS',errors)
    page.get_by_role('button',name='Clinical text',exact=True).click()
    page.get_by_role('heading',name='Text classification',exact=True).wait_for(timeout=30000)
    page.get_by_role('button',name='Classify & cluster text',exact=True).click()
    page.get_by_text('Predicted text severity:',exact=False).wait_for(timeout=60000)
    print('TEXT_CLASSIFICATION_VISIBLE',page.get_by_text('Predicted text severity:',exact=False).inner_text())
    page.screenshot(path='outputs/figures/ui_tests/workspace_text.png',full_page=True)
    for number in range(1, 16):
        page.locator('#workspace-selector').get_by_role('combobox').click()
        page.get_by_role('option').filter(has_text=re.compile(rf'\bT{number:02d}\b')).click()
        page.locator('.tutorial-number').filter(has_text=f'Tutorial {number:02d} of 15').wait_for(state='visible', timeout=30000)
    print('ALL_15_TUTORIALS_NAVIGATE', True)
    page.get_by_role('button',name='Clinical text',exact=True).click()
    page.get_by_role('heading',name='Text classification',exact=True).wait_for()
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path='outputs/figures/ui_tests/workspace_mobile.png',full_page=True)
    overflow = page.evaluate('document.documentElement.scrollWidth > innerWidth')
    print('MOBILE_OVERFLOW', overflow)
    assert not overflow, 'Mobile layout overflows the viewport'
    assert not errors, errors
    bounds = page.locator('#workspace').bounding_box()
    assert bounds['width'] >= 300, bounds
    assert page.locator('#workspace').bounding_box()['y'] > page.locator('#input-panel').bounding_box()['y']
    page.set_viewport_size({'width':1440,'height':1000})
    page.get_by_role('button',name='Screening',exact=True).click()
    page.get_by_role('button',name='Moderate sample',exact=True).click()
    page.get_by_role('button',name=re.compile('^Image settings')).click()
    expect(page.get_by_role('checkbox',name=re.compile('^Image is already T2 normalized'))).to_be_checked(timeout=30000)
    page.get_by_role('button',name=re.compile('^Image settings')).click()
    page.get_by_role('button',name='Run screening',exact=True).click()
    page.locator('#screening-summary .result-header').wait_for(state='visible',timeout=120000)
    assert page.locator('#screening-summary .grade-badge').is_visible()
    page.screenshot(path='outputs/figures/ui_tests/workspace_screening.png',full_page=True)
    print('SCREENING_RESULT',page.locator('#screening-summary .result-header').inner_text())
    page.get_by_role('button',name='Report',exact=True).click()
    page.get_by_text('Screening report',exact=False).last.wait_for(timeout=30000)
    assert page.locator('a[href*=".pdf"]').count() > 0
    assert page.locator('a[href*=".json"]').count() > 0
    assert not errors, errors
    print('REPORT_EXPORTS', True, 'JS_ERRORS', errors)
    browser.close()

