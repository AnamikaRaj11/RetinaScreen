"""T6 presentation arranged around the questions each figure answers."""
import gradio as gr
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR


def figure(filename, label, caption):
    path = FIGURES_DIR / filename
    if path.exists():
        gr.Image(str(path), label=label, interactive=False)
        gr.Markdown(caption)
    else:
        gr.Markdown(f'Saved figure unavailable: `{filename}`')


def build_visualization_workspace(api, state):
    gr.Markdown(
        '**Purpose:** understand the dataset, see how images are prepared, review model evaluation, '
        'and explain the current prediction.\n\n'
        '**Reading order:** 01 Dataset  /  02 Preprocessing  /  03 Evaluation  /  04 Current image. '
        'Open one section at a time. Sections 01-03 use saved project results; section 04 uses your screening.',
        elem_id='t6-reading-guide')
    with gr.Accordion('01 / Dataset overview - What images are we working with?', open=True, elem_id='t6-dataset'):
        gr.Markdown('### Understand the population\nCompare the number of images in each severity grade before interpreting model results.')
        figure('t6_class_distribution.png', 'Severity distribution before and after deduplication',
               '**How to read:** compare bar heights across grades and between the original and deduplicated cohorts. '
               '**Why it matters:** uneven class sizes mean overall accuracy can hide poor performance on rare grades.')
        with gr.Accordion('View example images across the five severity grades', open=False):
            figure('t6_severity_sample_grid.png', 'Labeled retinal examples: grades 0-4',
                   '**How to read:** compare the labeled examples across grades. These are dataset examples, '
                   'not predictions for your uploaded image.')
    with gr.Accordion('02 / Image preparation - What changes before analysis?', open=False, elem_id='t6-preparation'):
        gr.Markdown('### Follow the image through preparation\nView the same cases across processing stages to understand the purpose of each transformation.')
        figure('t6_preprocessing_panels.png', 'Raw image, normalization, enhancement and vessel candidates',
               '**How to read:** follow each case from left to right. Normalization reduces illumination differences; '
               'enhancement improves contrast; the vessel map visualizes candidate structures. '
               '**Keep in mind:** the CNN uses the standardized image, not the vessel overlay. T4 contains the processing controls.')
    with gr.Accordion('03 / Model evaluation - How should performance be assessed?', open=False, elem_id='t6-evaluation'):
        gr.Markdown('### Read the saved per-grade metrics\nPrecision describes how often a predicted grade is correct; '
                    'recall describes how many examples of that grade were found; F1 combines both. Support is the number of evaluated examples.')
        metrics = REPORTS_DIR / 't5_classification_metrics.csv'
        if metrics.exists():
            gr.Dataframe(pd.read_csv(metrics), interactive=False, label='Saved T5 evaluation metrics by severity grade')
        gr.Markdown('**How to read:** compare precision and recall for each grade alongside its support. '
                    'These are saved experiment metrics, not the confidence of the current image.')
        with gr.Accordion('Illustrative learning curves and confusion matrix', open=False):
            gr.Markdown('**Illustrations only:** the original T6 script hardcoded these curves and confusion counts. '
                        'Use them to learn how to read these chart types; they do not measure the trained model performance.')
            figure('t6_training_curves.png', 'Illustrative training curves - not measured logs',
                   '**Reading concept:** compare training and validation loss over epochs; divergence can suggest overfitting. '
                   'No training trend should be inferred for this model from these illustrative values.')
            figure('t6_confusion_matrix_heatmap.png', 'Illustrative confusion matrix - not held-out predictions',
                   '**Reading concept:** diagonal cells show correct classifications; off-diagonal cells show grade confusion. '
                   'These particular counts are illustrative.')
    with gr.Accordion('04 / Current image - Where did the CNN focus?', open=False, elem_id='t6-current-image'):
        gr.Markdown('### Compare the CNN input with its attention map\nRun screening first, then compare the same image side by side. '
                    'The overlay comes from the trained model gradients for its predicted class.')
        case_note = gr.Markdown('No screening result yet. Choose an image and select **Run screening**.', elem_id='t6-case-note')
        with gr.Row():
            case_input = gr.Image(label='Standardized image used by the CNN', height=320, elem_id='t6-case-input')
            cam = gr.Image(label='Grad-CAM attention overlay', height=320, elem_id='t6-case-attention')
        opacity = gr.Slider(0, 1, value=.55, label='Attention overlay opacity',
                            info='0 shows the image alone; 1 shows the attention colors alone.')
        opacity.change(api.update_gradcam_overlay, [opacity, state], cam)
        gr.Markdown('**How to read:** warmer colors show stronger relative attention for the predicted class; '
                    'cooler colors show weaker attention. Compare the overlay with the input before interpreting a highlighted region.\n\n'
                    '**Interpretation limit:** attention does not identify or confirm lesions, and it does not prove the prediction is correct.')
        with gr.Accordion('Compare with saved Grad-CAM examples', open=False):
            figure('t6_gradcam_validation_cases.png', 'Saved Grad-CAM case comparisons',
                   'Use these saved examples to compare attention patterns across cases. They are separate from the current upload.')
    with gr.Accordion('T6 reference report and figure provenance', open=False):
        gr.Markdown('The original report is retained for reference. Its claims about attention identifying lesions '
                    'or proving anatomical validity are not established by a heatmap alone. '
                    'The learning curves and confusion matrix are illustrative, as identified above.')
        report = REPORTS_DIR / 't6_biomedical_image_visualization.md'
        if report.exists():
            gr.Markdown(report.read_text(encoding='utf-8'))
    return cam, case_input, case_note
