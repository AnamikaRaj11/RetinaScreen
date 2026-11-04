"""Browser regression for visible SIFT, RANSAC and trained CNN results in T5."""
import re
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width':1440, 'height':1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:7860', wait_until='domcontentloaded')
    page.get_by_role('button', name='Moderate sample', exact=True).click()
    page.get_by_role('button', name='Remove Image', exact=True).wait_for(timeout=30000)
    page.get_by_role('button', name='Run screening', exact=True).click()
    page.locator('#screening-summary .result-header').wait_for(timeout=120000)
    page.locator('#workspace-selector').get_by_role('combobox').click()
    page.get_by_role('option').filter(has_text='T05').click()
    for heading in ['SIFT feature detection', 'RANSAC geometric registration', 'CNN severity classification']:
        page.get_by_role('heading', name=heading, exact=True).wait_for(timeout=30000)
    page.locator('#t5-ransac-image img').first.wait_for()
    assert 'RANSAC inliers' in page.locator('#t5-ransac-metrics').inner_text()
    assert 'reprojection RMSE' in page.locator('#t5-ransac-metrics').inner_text()
    assert 'Grade' in page.locator('#t5-cnn-probabilities').inner_text()
    assert page.get_by_text('best_retinascreen_cnn.pth', exact=False).count() > 0
    page.screenshot(path='outputs/figures/ui_tests/t5_sift_ransac_cnn.png', full_page=True)
    assert not errors, errors
    print('T5 SIFT, RANSAC visualization/metrics/homography and trained CNN results verified; no JavaScript errors.')
    browser.close()
