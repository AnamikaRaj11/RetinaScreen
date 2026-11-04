"""Verify the T6 reading order and current-image comparison in Chromium."""
import re
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width':1440,'height':1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:7860', wait_until='domcontentloaded')
    page.locator('#workspace-selector').get_by_role('combobox').click()
    page.get_by_role('option').filter(has_text='T06').click()
    page.locator('#t6-reading-guide').wait_for()
    labels = [page.locator(selector).get_by_role('button').first.inner_text() for selector in
              ['#t6-dataset','#t6-preparation','#t6-evaluation','#t6-current-image']]
    assert all(label.startswith(f'{i:02d}') for i,label in enumerate(labels,1)), labels
    page.screenshot(path='outputs/figures/ui_tests/t6_structured_overview.png',full_page=True)
    page.locator('#t6-evaluation').get_by_role('button').first.click()
    page.get_by_text('Saved T5 evaluation metrics by severity grade',exact=True).wait_for()
    page.locator('#t6-evaluation').get_by_role('button').first.click()
    page.get_by_role('button',name='Moderate sample',exact=True).click()
    page.get_by_role('button',name='Remove Image',exact=True).wait_for(timeout=30000)
    page.get_by_role('button',name='Run screening',exact=True).click()
    page.locator('#screening-summary .result-header').wait_for(timeout=120000)
    page.locator('#workspace-selector').get_by_role('combobox').click()
    page.get_by_role('option').filter(has_text='T06').click()
    page.locator('#t6-dataset').get_by_role('button').first.click()
    page.locator('#t6-current-image').get_by_role('button').first.click()
    page.locator('#t6-case-input img').first.wait_for()
    page.locator('#t6-case-attention img').first.wait_for()
    assert 'Grade' in page.locator('#t6-case-note').inner_text()
    assert 'Confidence' in page.locator('#t6-case-note').inner_text()
    page.get_by_role('slider',name='Attention overlay opacity').focus()
    page.get_by_role('slider',name='Attention overlay opacity').press('ArrowLeft')
    page.screenshot(path='outputs/figures/ui_tests/t6_structured_current_image.png',full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
    assert not errors, errors
    print('T6 sequence, grouped figures, evaluation table, live input/Grad-CAM comparison and mobile layout verified.')
    browser.close()
