"""RetinaScreen presentation: one workspace for each of the 15 tutorials."""
from pathlib import Path

import gradio as gr
import pandas as pd

from src.config import DATA_DIR, FIGURES_DIR, REPORTS_DIR
from src.tutorial_services import classify_text, extract_entities, search_notes, disambiguate


TUTORIALS = [
    ('EHR overview', 'Understand the dataset and its record schema.', 't1_ehr_understanding.md', []),
    ('Standardization', 'Audit quality, crop borders and normalize the CNN input.', 't2_ehr_preprocessing.md', ['t2_preprocessing_comparison.png']),
    ('Duplicates & missing data', 'Inspect real image hashes, duplicate exclusions and imputation results.', 't3_redundant_missing_data.md', ['t3_deduplication_imputation.png']),
    ('Image processing', 'Enhance, restore, segment and compress retinal images.', 't4_biomedical_image_processing.md', ['t4_enhancement_comparison.png', 't4_restoration_benchmark.png', 't4_segmentation_vessels_exudates.png', 't4_compression_tradeoffs.png']),
    ('Image analysis', 'Run the trained ResNet-101 and inspect SIFT / RANSAC registration.', 't5_biomedical_image_analysis.md', ['t5_sift_ransac_matching.png', 't5_cnn_training_performance.png']),
    ('Visualization', 'Inspect model attention, class distributions and validation figures.', 't6_biomedical_image_visualization.md', ['t6_class_distribution.png', 't6_severity_sample_grid.png', 't6_preprocessing_panels.png', 't6_training_curves.png', 't6_confusion_matrix_heatmap.png', 't6_gradcam_validation_cases.png']),
    ('Demand & staffing', 'Explore simulated weekly demand, held-out forecasts and workforce capacity.', 't7_patient_prediction_staffing.md', ['sim_t7_staffing_forecast.png']),
    ('Strategic planning I', 'Describe cohort severity, diagnostic gaps and workload scenarios.', 't8_strategic_planning_part1.md', ['t8_strategic_planning_descriptive.png']),
    ('Strategic planning II', 'Compare referral policies and explicit cost assumptions.', 't9_strategic_planning_part2.md', ['t9_strategic_dashboard.png']),
    ('Text classification', 'Classify and cluster clinical text independently of the image model.', 't10_clinical_text_classification.md', ['t10_text_classification_clustering.png']),
    ('Information extraction', 'Extract entity spans from entered text with regex and spaCy Matcher.', 't11_clinical_information_extraction.md', ['t11_information_extraction_ner.png']),
    ('Ranking & word senses', 'Search synthetic notes with BM25 / TF-IDF and compare sense disambiguation.', 't12_ranking_disambiguation.md', ['t12_ranking_and_wsd.png']),
    ('Dimensionality reduction', 'Explore cached CNN embeddings with PCA, t-SNE and UMAP.', 't13_dimensionality_reduction.md', ['t13_dimensionality_reduction.png']),
    ('Behavioral exploration', 'Inspect cleaning, distributions, correlations and statistical comparisons.', 't14_behavioral_part1.md', ['t14_behavioral_factors_eda.png']),
    ('Behavioral modeling', 'Use the saved BMI, clustering and behavioral risk models.', 't15_behavioral_part2.md', ['t15_behavioral_modeling_shap.png']),
]

from src.presentation import CSS, masthead, ready_screen



def table(path, rows=None):
    frame = pd.read_csv(path)
    return frame.head(rows) if rows else frame


def build_workspace(api):
    theme = gr.themes.Soft(primary_hue='teal', secondary_hue='slate', neutral_hue='slate',
                           font=['Aptos', 'Segoe UI', 'sans-serif']).set(
        body_background_fill='#f3f6f9', body_background_fill_dark='#f3f6f9',
        body_text_color='#172e45', body_text_color_dark='#172e45',
        block_label_background_fill='transparent', block_label_background_fill_dark='transparent',
        block_label_text_color='#677b8d', block_label_text_color_dark='#677b8d',
        block_label_shadow='none', block_label_padding='3px 5px',
        block_background_fill='white', block_background_fill_dark='white',
        button_primary_background_fill='#087d78', button_primary_background_fill_dark='#087d78',
        button_secondary_background_fill='#f2f6f8', button_secondary_background_fill_dark='#f2f6f8',
        input_background_fill='white', input_background_fill_dark='white',
        background_fill_primary='white', background_fill_primary_dark='white',
        background_fill_secondary='#f5f8fa', background_fill_secondary_dark='#f5f8fa',
        border_color_primary='#e0e7ed', border_color_primary_dark='#e0e7ed')
    with gr.Blocks(title='RetinaScreen | Retinal screening & tutorial workspace') as demo:
        state = gr.State()
        gr.HTML(masthead())
        with gr.Row(elem_id='section-navigation'):
            section_buttons = [(gr.Button(label, size='sm'), index) for label, index in
                               [('Screening', 0), ('Image analysis', 4), ('Clinical text', 10),
                                ('Capacity planning', 7), ('Behavioral factors', 14), ('Report', 16)]]
        with gr.Row(equal_height=False, elem_id='analysis-layout'):
            with gr.Column(scale=1, min_width=300, elem_id='input-panel'):
                gr.Markdown('### New screening\nStart with a retinal image to generate a screening result.')
                image = gr.Image(type='numpy', label='Retinal fundus image', height=220, elem_id='input-image', sources=['upload', 'clipboard'])
                with gr.Accordion('Image settings', open=False):
                    processed = gr.Checkbox(False, label='Image is already T2 normalized', info='Set automatically for samples. Leave off for raw photographs.')
                with gr.Row(elem_id='sample-buttons'):
                    example0 = gr.Button('Normal sample', size='sm')
                    example2 = gr.Button('Moderate sample', size='sm')
                    example4 = gr.Button('PDR sample', size='sm')
                lifestyle = gr.Checkbox(False, label='Include behavioral experiment')
                with gr.Accordion('Lifestyle inputs · simulated models', open=False):
                    age = gr.Number(55, label='Age (years)')
                    sex = gr.Radio(['Male', 'Female'], value='Male', label='Sex (model encoding)')
                    with gr.Row():
                        weight = gr.Number(75, label='Weight (kg)')
                        height = gr.Number(170, label='Height (cm)')
                    activity = gr.Number(150, label='Activity (minutes/week)')
                    sleep = gr.Number(7, label='Sleep (hours/night)')
                    smoking = gr.Dropdown(['Never', 'Former', 'Current'], value='Never', label='Smoking history')
                    duration = gr.Number(5, label='Diabetes duration (years)')
                    hba1c = gr.Number(7, label='HbA1c (%)')
                    calories = gr.Number(2100, label='Daily calories (kcal)')
                    stress = gr.Slider(0, 10, value=5, label='Stress score')
                    alcohol = gr.Number(2, label='Alcohol (drinks/week)')
                run = gr.Button('Run screening', variant='primary', elem_id='screen-button')
                gr.Markdown('For education and research. Screening results require clinician review.', elem_id='input-help')
            with gr.Column(scale=3, min_width=500, elem_id='workspace'):
                choices = ['Screening summary'] + [f'T{i:02d} · {v[0]}' for i, v in enumerate(TUTORIALS, 1)] + ['Export report']
                nav = gr.Dropdown(choices, value=choices[0], label='Explore workspace', filterable=False, elem_id='workspace-selector')
                with gr.Tabs(elem_id='tutorial-panels') as tabs:
                    with gr.Tab('Summary', id='summary'):
                        summary = gr.Markdown(ready_screen(), elem_id='screening-summary')
                        probabilities = gr.Label(label='Probability across severity grades', num_top_classes=5, visible=False)
                    live = {}
                    for i, (title, description, report, figures) in enumerate(TUTORIALS, 1):
                        with gr.Tab(f'T{i:02d}', id=f't{i}'):
                            gr.HTML(f'<div class="tutorial-intro"><div class="tutorial-number">Tutorial {i:02d} of 15</div><h2>{title}</h2><p>{description}</p></div>')
                            if i in [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 14, 15]:
                                live[i] = gr.Markdown('Results for your image will appear here after screening.' if i not in [7,8,9,12] else description)
                            if i == 1:
                                frame = table(DATA_DIR / 'ehr_records.csv')
                                gr.Markdown(f'**{len(frame):,} records** · **5 severity grades** · APTOS 2019 with simulated encounter metadata')
                                gr.Dataframe(frame.head(15), label='EHR schema and sample records', interactive=False)
                            if i == 2:
                                normalized = gr.Image(label='Standardized image input (224 x 224)', height=300)
                            if i == 3:
                                gr.Dataframe(table(REPORTS_DIR / 'removal_log.csv', 25), label='Saved duplicate audit (first 25 rows)', interactive=False)
                                gr.Dataframe(table(DATA_DIR / 'clean_records.csv').isna().sum().rename('missing_values').reset_index(), label='Cohort missing-value audit', interactive=False)
                            if i == 4:
                                with gr.Row():
                                    green = gr.Image(label='Green channel')
                                    clahe = gr.Image(label='CLAHE enhancement')
                                    denoise = gr.Image(label='Bilateral restoration')
                                with gr.Row():
                                    vessels = gr.Image(label='Morphological vessel candidates')
                                    lesions = gr.Image(label='Bright-region candidates (not confirmed lesions)')
                                quality = gr.Slider(5, 100, value=80, step=5, label='JPEG quality')
                                compressed = gr.Image(label='JPEG reconstruction', height=280)
                                compression_info = gr.Markdown()
                                quality.change(api.update_jpeg_compression, [quality, state], [compressed, compression_info])
                            if i == 5:
                                gr.Markdown('### SIFT feature detection\nLocal image features detected in the uploaded photograph.')
                                sift = gr.Image(label='SIFT keypoints', height=320)
                                gr.Markdown('### RANSAC geometric registration\nMatch the standardized image to a rotated and scaled copy, then reject inconsistent feature correspondences. This tests registration, not disease severity.')
                                ransac_image = gr.Image(label='RANSAC matches: source and transformed copy', height=340, elem_id='t5-ransac-image')
                                ransac_metrics = gr.Markdown('Run screening to calculate matches, inliers and reprojection error.', elem_id='t5-ransac-metrics')
                                ransac_matrix = gr.Dataframe(headers=['h1', 'h2', 'h3'], datatype='number', interactive=False, label='Estimated 3 x 3 homography')
                                gr.Markdown('### CNN severity classification\nThe existing trained ResNet-101 predicts the five ICDR severity grades. These probabilities come from the CNN, independently of SIFT and RANSAC.')
                                cnn_info = gr.Markdown('Run screening to view the trained CNN prediction.')
                                with gr.Row():
                                    cnn_input = gr.Image(label='Standardized CNN input', height=260)
                                    cnn_probabilities = gr.Label(label='ResNet-101 grade probabilities', num_top_classes=5, elem_id='t5-cnn-probabilities')
                            if i == 6:
                                from src.t6_workspace import build_visualization_workspace
                                cam, t6_input, t6_case_note = build_visualization_workspace(api, state)
                            if i == 7:
                                annual = gr.Slider(1000, 100000, value=20000, step=1000, label='Annual screening volume')
                                staffing = gr.Markdown(api.update_staffing_calculation(20000))
                                annual.change(api.update_staffing_calculation, annual, staffing)
                                gr.Dataframe(table(REPORTS_DIR / 'sim_staffing_forecast_metrics.csv'), label='Saved held-out weekly forecasts', interactive=False)
                            if i == 8:
                                cohort = table(DATA_DIR / 'dedup_records.csv')
                                counts = cohort.diagnosis.value_counts().sort_index()
                                gr.Dataframe(pd.DataFrame({'grade': counts.index, 'records': counts.values, 'share_pct': (counts.values / len(cohort) * 100).round(2)}), interactive=False, label='Observed cohort distribution')
                            if i == 9:
                                threshold = gr.Dropdown(['Grade 1+', 'Grade 2+', 'Grade 3+', 'Grade 4+'], value='Grade 2+', label='Policy threshold')
                                policy = gr.Markdown(api.update_threshold_tradeoff('Grade 2+'))
                                threshold.change(api.update_threshold_tradeoff, threshold, policy)
                            if i in [10, 11]:
                                note = gr.Textbox(value='OS: Moderate nonproliferative diabetic retinopathy. Hard exudates noted. Refer to ophthalmology.', lines=4, label='Independent clinical text experiment')
                                action = gr.Button('Classify & cluster text' if i == 10 else 'Extract entity spans')
                                if i == 10:
                                    text_result = gr.Markdown()
                                    action.click(classify_text, note, text_result)
                                else:
                                    entity_result = gr.Dataframe(headers=['Category', 'Value', 'Tool', 'Start', 'End'], interactive=False)
                                    action.click(lambda text: pd.DataFrame(extract_entities(text), columns=['Category','Value','Tool','Start','End']), note, entity_result)
                            if i == 12:
                                query = gr.Textbox(label='Search query', value='severe hemorrhage refer urgently')
                                method = gr.Radio(['BM25', 'TF-IDF'], value='BM25', label='Ranking method')
                                search = gr.Button('Search notes')
                                matches = gr.Dataframe(interactive=False, label='Ranked matches from all synthetic notes')
                                search.click(search_notes, [query, method], matches)
                                sentence = gr.Textbox(value='Patient had watery discharge from the right eye.', label='Sentence for word sense disambiguation')
                                word = gr.Dropdown(['discharge','positive','stable','dense'], value='discharge', label='Ambiguous term')
                                wsd = gr.Button('Compare word senses')
                                sense = gr.Markdown()
                                wsd.click(disambiguate, [sentence, word], sense)
                            if i == 13:
                                projection = gr.Radio(['PCA', 't-SNE', 'UMAP'], value='PCA', label='Projection')
                                manifold = gr.Image(value=api.update_manifold_plot('PCA', None), label='CNN embedding map')
                                gr.Markdown('PCA and UMAP transform the uploaded embedding. t-SNE displays the cached reference cohort only; it has no out-of-sample transform.')
                                projection.change(api.update_manifold_plot, [projection, state], manifold)
                            if i == 14:
                                gr.Dataframe(table(DATA_DIR / 'clean_sim_behavioral_factors.csv').describe().round(2).reset_index(), label='Simulated cohort descriptive statistics', interactive=False)
                            if i != 6:
                                with gr.Accordion('Saved tutorial experiment · figures & report', open=False):
                                    gr.Markdown('These are saved cohort experiments, separate from the current upload.')
                                    if i == 9:
                                        gr.Markdown('**Assumptions:** the saved T9 report uses estimated recall adjustments and cost scenarios. These do not measure threshold sensitivity or prove an optimal clinical policy.')
                                    for figure in figures:
                                        path = FIGURES_DIR / figure
                                        if path.exists():
                                            gr.Image(str(path), label=Path(figure).stem.replace('_',' '), interactive=False)
                                    path = REPORTS_DIR / report
                                    if path.exists():
                                        gr.Markdown(path.read_text(encoding='utf-8'))
                    with gr.Tab('Report', id='report'):
                        final = gr.Markdown('Run screening to create a report.')
                        with gr.Row():
                            pdf = gr.File(label='Screening report · PDF')
                            json_file = gr.File(label='Structured result · JSON')
                nav.change(lambda value: gr.Tabs(selected=('summary' if value == choices[0] else 'report' if value == choices[-1] else f't{choices.index(value)}')), nav, tabs)
        for button, index in section_buttons:
            button.click(lambda target=choices[index]: target, outputs=nav)
        gr.HTML('<div class="presentation-footer"><span>RetinaScreen / Biomedical data analysis</span><span>APTOS image data. Synthetic notes and simulated planning &amp; behavioral data.</span></div>')
        inputs = [image, age, sex, weight, height, activity, sleep, smoking, duration, hba1c]
        for button, grade in [(example0,0), (example2,2), (example4,4)]:
            button.click(lambda g=grade: (*api.load_example(g), True), outputs=[*inputs, processed])
        image.upload(lambda: False, outputs=processed)
        outputs = [summary, probabilities, live[1], live[2], live[3], green, clahe, denoise, vessels, lesions,
                   live[5], sift, cam, live[7], live[8], live[9], live[10], live[11], live[12], manifold,
                   live[14], live[15], final, pdf, json_file]

        def analyze(img, a, sx, wt, ht, act, slp, smk, dur, hb, enabled, kcal, stress_value, drinks, preprocessed):
            if img is None:
                raise gr.Error('Upload a retinal image or select a sample first.')
            values = [a, sx, wt, ht, act, slp, smk, dur, hb] if enabled else [None] * 9
            try:
                result = api.run_screening_analysis(img, *values, daily_calories=kcal, stress_score=stress_value,
                                                   alcohol_drinks_week=drinks, image_is_preprocessed=preprocessed)
                rendered = list(api.dispatch_analysis_results(result))
                rendered[1] = gr.update(value=rendered[1], visible=True)
                registration = result['registration_data']
                cnn_details = (f"**Predicted grade:** {result['pred_class']} ({api.CLASS_LABELS[result['pred_class']]})\n\n"
                               f"**Model confidence:** {result['confidence_pct']:.1f}%\n\n"
                               f"**Loaded checkpoint:** `{api.MODEL_PATH.name}`. Input: 224 x 224 RGB, training-compatible normalization.")
                matrix = pd.DataFrame(registration['homography'], columns=['h1','h2','h3']) if registration['homography'] is not None else pd.DataFrame(columns=['h1','h2','h3'])
                return (result, *rendered, result['normalized_rgb'], choices[0],
                        registration['image'], registration['summary'], matrix,
                        cnn_details, result['normalized_rgb'], rendered[1]['value'],
                        result['normalized_rgb'],
                        f"**Current CNN prediction:** Grade {result['pred_class']} ({api.CLASS_LABELS[result['pred_class']]}) | **Confidence:** {result['confidence_pct']:.1f}%. The overlay explains this prediction.")
            except (ValueError, OSError) as error:
                raise gr.Error(str(error)) from error

        run.click(analyze, [*inputs, lifestyle, calories, stress, alcohol, processed], [state, *outputs, normalized, nav, ransac_image, ransac_metrics, ransac_matrix, cnn_info, cnn_input, cnn_probabilities, t6_input, t6_case_note], concurrency_limit=1)
    return demo, theme, CSS, None
