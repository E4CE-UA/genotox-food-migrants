"""Exercise the actual interactive notebook (optional Playwright dependency)."""
import argparse,hashlib,json,re,subprocess,sys,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'docs/ui_checks')
args=parser.parse_args()
HERE=args.output.resolve();HERE.mkdir(parents=True,exist_ok=True)
PYTHON=Path(sys.executable)
NOTEBOOK=ROOT/'notebooks/what_if_wrong_compound.py'
CASES=json.loads((ROOT/'docs/case_results.json').read_text())['results']
log=(HERE/'browser_server.log').open('w')
server=subprocess.Popen([str(PYTHON),'-m','marimo','run',str(NOTEBOOK),'--headless','--no-sandbox','--no-token','--port','2724'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
report={'tested_notebook_sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()}
try:
    for _ in range(50):
        try:
            urllib.request.urlopen('http://127.0.0.1:2724',timeout=1);break
        except Exception:time.sleep(.3)
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
        page=context.new_page();page.set_default_timeout(20000)
        console_errors=[];page.on('pageerror',lambda e:console_errors.append(str(e)))
        page.goto('http://127.0.0.1:2724',wait_until='domcontentloaded')
        summary=page.locator('#scenario-comparison');summary.wait_for(timeout=90000)
        expect(summary).to_have_attribute('data-assumption','published')
        assert abs(float(summary.get_attribute('data-average'))-CASES['published_0.4']['conditional_score'])<1e-9
        report['assigned']=summary.inner_text()
        comparison=page.locator('#molecule-comparison')
        expect(comparison.locator('section').nth(0)).to_have_attribute('data-comparison-key','FCM2018-0014:1')
        expect(comparison.locator('section').nth(1)).to_have_attribute('data-comparison-key','FCM2018-0014:2')
        page.locator('button[aria-haspopup="dialog"]').nth(1).click()
        page.get_by_role('option').filter(has_text='#07').click()
        expect(comparison.locator('section').nth(0)).to_have_attribute('data-comparison-key','FCM2018-0014:7')
        page.locator('button[aria-haspopup="dialog"]').nth(2).click()
        page.get_by_role('option').filter(has_text='#20').click()
        expect(comparison.locator('section').nth(1)).to_have_attribute('data-comparison-key','FCM2018-0014:20')
        expect(summary).to_have_attribute('data-assumption','published')
        assert abs(float(summary.get_attribute('data-average'))-CASES['published_0.4']['conditional_score'])<1e-9
        comparison.scroll_into_view_if_needed();page.screenshot(path=str(HERE/'comparison_AB.png'))
        report['independent_AB_in_assigned_mode']=True

        radios=page.get_by_role('radio');radios.nth(1).click()
        expect(summary).to_have_attribute('data-assumption','same_formula')
        expect(summary).to_have_attribute('data-n-evaluable','14')
        assert abs(float(summary.get_attribute('data-average'))-CASES['same_formula_0.4']['conditional_score'])<1e-9
        expect(page.locator('.gx-mol-button')).to_have_count(20)
        assert f"{100*CASES['published_0.4']['conditional_score']:.2f}%" in page.locator('.gx-inspector').inner_text()
        summary.scroll_into_view_if_needed()
        page.screenshot(path=str(HERE/'scenario_040.png'))
        report['equal_weight_040']=summary.inner_text()
        page.locator('.gx-mol-button[data-key="FCM2018-0014:7"]').click()
        expect(page.locator('.gx-inspector')).to_have_attribute('data-selection-key','FCM2018-0014:7')
        assert 'Michael acceptor' in page.locator('.gx-alert-live').inner_text()
        assert abs(float(summary.get_attribute('data-average'))-CASES['same_formula_0.4']['conditional_score'])<1e-9
        report['structure7']=page.locator('.gx-inspector').inner_text()
        page.locator('.gx-mol-button[data-key="FCM2018-0014:20"]').click()
        expect(page.locator('.gx-inspector')).to_have_attribute('data-selection-key','FCM2018-0014:20')
        assert 'No match in the 10-rule set' in page.locator('.gx-alert-live').inner_text()
        slider=page.get_by_role('slider');slider.focus()
        for _ in range(10):slider.press('ArrowRight')
        expect(slider).to_have_attribute('aria-valuenow','0.5')
        expect(summary).to_have_attribute('data-n-evaluable','12')
        assert abs(float(summary.get_attribute('data-average'))-CASES['same_formula_0.5']['conditional_score'])<1e-9
        report['equal_weight_050']=summary.inner_text()
        summary.scroll_into_view_if_needed();page.screenshot(path=str(HERE/'scenario_050.png'))
        slider.focus();slider.press('End')
        expect(slider).to_have_attribute('aria-valuenow','0.9')
        expect(summary).to_have_attribute('data-n-evaluable','0')
        expect(summary).to_have_attribute('data-average','None')
        assert 'Unavailable' in summary.inner_text()
        report['zero_coverage']=summary.inner_text()
        with page.expect_download(timeout=20000) as download_info:
            page.get_by_text('Download the current results and source identifiers',exact=True).click()
        destination=HERE/'export_zero_coverage.json';download_info.value.save_as(destination)
        payload=json.loads(destination.read_text())
        assert payload['result']['conditional_score'] is None
        assert payload['result']['listed_formula_pool_size']==2484
        assert payload['inspected_structure']['rank']==20
        assert payload['provenance']['efsa_release']['status']=='content_matched_reference_export'
        report['json_zero_verified']=True
        report['model_fit_token']=payload['provenance']['model_fit_token']
        slider.focus();slider.press('Home')
        for _ in range(30):slider.press('ArrowRight')
        expect(slider).to_have_attribute('aria-valuenow','0.4')
        expect(summary).to_have_attribute('data-n-evaluable','14')
        page.locator('button[aria-haspopup="dialog"]').first.click()
        options=page.get_by_role('option')
        report['dropdown_options']=options.all_inner_texts()
        if options.count():
            options.filter(has_text='FCM2018-0001').click()
        else:
            page.get_by_text(re.compile(r'^FCM2018-0001 ·')).click()
        expect(summary).to_contain_text('FCM2018-0001')
        expect(page.locator('.gx-inspector')).to_contain_text('TRAINING IDENTITY')
        report['training_identity_notice_verified']=True
        page.locator('button[aria-haspopup="dialog"]').first.click()
        page.get_by_role('option').filter(has_text='FCM2018-0002').click()
        expect(summary).to_contain_text('FCM2018-0002')
        expect(summary).to_contain_text('Standard confirmation is preserved')
        expect(page.locator('.gx-inspector')).to_have_attribute('data-selection-key','FCM2018-0002:1')
        report['confirmed']=summary.inner_text()
        expect(comparison.locator('section').nth(0)).to_have_attribute('data-comparison-key','FCM2018-0002:1')
        with page.expect_download(timeout=20000) as download_info:
            page.get_by_text('Download the current results and source identifiers',exact=True).click()
        destination=HERE/'export_confirmed.json';download_info.value.save_as(destination)
        payload=json.loads(destination.read_text())
        assert [row['weight'] for row in payload['candidates']]==[1]+[0]*19
        assert payload['provenance']['model_fit_token']==report['model_fit_token']
        report['model_unchanged_by_controls']=True
        page.locator('button[aria-haspopup="dialog"]').first.click()
        page.get_by_role('option').filter(has_text='FCM2018-0003').click()
        expect(summary).to_contain_text('FCM2018-0003')
        slider.focus();slider.press('End')
        for _ in range(10):slider.press('ArrowLeft')
        expect(slider).to_have_attribute('aria-valuenow','0.8')
        with page.expect_download(timeout=20000) as download_info:
            page.get_by_text('Download the current results and source identifiers',exact=True).click()
        destination=HERE/'export_FCM2018_0003.json';download_info.value.save_as(destination)
        payload=json.loads(destination.read_text())
        assert payload['setting']['feature_id']=='FCM2018-0003'
        assert payload['provenance']['model_fit_token']==report['model_fit_token']
        assert payload['comparison']['changes_scenario_weights'] is False
        report['reported_nan_crash_case_exported']=True

        page.get_by_text('Methods, provenance & sensitivity details',exact=True).click()
        expect(page.get_by_role('heading',name='Reproduced internal validation · five outer seeds',exact=True)).to_be_visible()
        assert 'lgbm_raw_ensemble' in page.locator('.gx-details').last.inner_text()
        report['generated_methods_table_visible']=True
        report['page_errors']=console_errors
        assert not console_errors,console_errors
        report['passed']=True
        context.close();browser.close()
except Exception as e:
    report['passed']=False;report['error']=repr(e)
    try:
        report['page_text']=page.locator('body').inner_text()[-12000:]
        page.screenshot(path=str(HERE/'browser_failure.png'))
    except Exception:pass
    raise
finally:
    (HERE/'browser_check.json').write_text(json.dumps(report,indent=2))
    server.terminate()
    try:server.wait(timeout=8)
    except subprocess.TimeoutExpired:server.kill()
    log.close()
    print(json.dumps({k:v for k,v in report.items() if k in ['passed','error','json_zero_verified','model_unchanged_by_controls','page_errors']},indent=2),flush=True)
