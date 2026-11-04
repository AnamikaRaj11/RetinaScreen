"""Interactive adapters for the existing tutorial algorithms and artifacts."""
import hashlib
import re
from functools import lru_cache

import cv2
import imagehash
import joblib
import numpy as np
import pandas as pd
import spacy
from PIL import Image
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, MODELS_DIR
from src.t11_information_extraction import build_spacy_matcher, extract_rule_based


@lru_cache(maxsize=1)
def text_models():
    # The original T10 pipeline never persisted its fitted estimators. Build only
    # these missing estimators once; never refit the supplied image/behavior models.
    path = MODELS_DIR / 'clinical_text_models.joblib'
    if path.exists():
        return joblib.load(path)
    notes = pd.read_csv(DATA_DIR / 'synthetic_notes.csv')
    train, _ = train_test_split(notes, test_size=.2, stratify=notes.diagnosis, random_state=42)
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    x = vectorizer.fit_transform(train.sim_clinical_note)
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42).fit(x, train.diagnosis)
    cluster = KMeans(n_clusters=5, n_init=10, random_state=42).fit(vectorizer.transform(notes.sim_clinical_note))
    result = vectorizer, classifier, cluster
    joblib.dump(result, path)
    return result


def classify_text(text):
    if not text or not text.strip():
        return 'Enter a clinical note.'
    vectorizer, classifier, cluster = text_models()
    x = vectorizer.transform([text])
    return (f'Predicted text severity: **Grade {int(classifier.predict(x)[0])}**\n\n'
            f'Document cluster: **{int(cluster.predict(x)[0])}** (cluster IDs are independent of severity grades).\n\n'
            'TF-IDF + logistic regression and K-Means, fitted on synthetic notes; this is a text experiment.')


@lru_cache(maxsize=1)
def entity_tools():
    nlp = spacy.blank('en')
    return nlp, build_spacy_matcher(nlp)


def extract_entities(text):
    nlp, matcher = entity_tools()
    return [{'Category': e['label'], 'Value': e['text'], 'Tool': e['source'],
             'Start': e['start'], 'End': e['end']}
            for e in extract_rule_based(text or '', nlp, matcher)]


@lru_cache(maxsize=1)
def retrieval_index():
    notes = pd.read_csv(DATA_DIR / 'synthetic_notes.csv')
    tokens = [re.findall(r'\w+', str(s).lower()) for s in notes.sim_clinical_note]
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    matrix = tfidf.fit_transform(notes.sim_clinical_note)
    return notes, BM25Okapi(tokens), tfidf, matrix


def search_notes(query, method='BM25'):
    if not query or not query.strip():
        return pd.DataFrame(columns=['record_id', 'diagnosis', 'score', 'sim_clinical_note'])
    notes, bm25, tfidf, matrix = retrieval_index()
    scores = (np.asarray(bm25.get_scores(re.findall(r'\w+', query.lower()))) if method == 'BM25'
              else (matrix @ tfidf.transform([query]).T).toarray().ravel())
    indices = np.argsort(-scores, kind='stable')[:5]
    result = notes.iloc[indices][['record_id', 'diagnosis', 'sim_clinical_note']].copy()
    result.insert(2, 'score', scores[indices])
    return result[result.score > 0]


def disambiguate(sentence, word):
    from src.t12_ranking_disambiguation import SENSE_DEFINITIONS, lesk_disambiguate, context_embedding_disambiguate
    frame = pd.DataFrame([{'sentence': sentence, 'word': word}])
    a = lesk_disambiguate(sentence, word)
    b = context_embedding_disambiguate(frame)[0]
    return f'**Lesk:** {SENSE_DEFINITIONS[word][a]}\n\n**TF-IDF context similarity:** {SENSE_DEFINITIONS[word][b]}'


@lru_cache(maxsize=1)
def image_index():
    entries = []
    for path in sorted((DATA_DIR / 'processed' / 'train_224').glob('*.png')):
        try:
            with Image.open(path) as image:
                candidate = image.convert('RGB')
                entries.append((path.stem, imagehash.phash(candidate), hashlib.md5(np.asarray(candidate).tobytes()).hexdigest()))
        except (OSError, SyntaxError, ValueError):
            # An unreadable cohort file must not prevent analysis of a valid upload.
            continue
    return entries


def image_identity(rgb):
    digest = hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    phash = imagehash.phash(Image.fromarray(rgb))
    matches = [(phash - other_hash, record, other_digest == digest)
               for record, other_hash, other_digest in image_index()]
    matches.sort()
    nearest = matches[0] if matches else None
    return digest, str(phash), nearest, len(matches)


def registration_analysis(rgb):
    gray = cv2.cvtColor(cv2.resize(rgb, (224, 224)), cv2.COLOR_RGB2GRAY)
    affine = cv2.getRotationMatrix2D((112, 112), 8, .95)
    target = cv2.warpAffine(gray, affine, (224, 224))
    canvas = cv2.cvtColor(np.hstack([gray, target]), cv2.COLOR_GRAY2RGB)
    result = {'image': canvas, 'homography': None, 'matches': 0, 'inliers': 0,
              'reprojection_rmse': None, 'summary': ''}
    sift = cv2.SIFT_create(nfeatures=500)
    k1, d1 = sift.detectAndCompute(gray, None)
    k2, d2 = sift.detectAndCompute(target, None)
    if d1 is None or d2 is None or len(d2) < 2:
        result['summary'] = 'Insufficient descriptors for registration.'
        return result
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in pairs if a.distance < .75 * b.distance]
    result['matches'] = len(good)
    if len(good) < 4:
        result['summary'] = f'{len(good)} matches; at least four are needed for homography.'
        return result
    source = np.float32([k1[m.queryIdx].pt for m in good])
    destination = np.float32([k2[m.trainIdx].pt for m in good])
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5)
    if homography is None or mask is None:
        result['summary'] = f'{len(good)} matches; RANSAC could not estimate a homography.'
        return result
    inliers = mask.ravel().astype(bool)
    for match, keep in zip(good, inliers):
        start = tuple(np.round(k1[match.queryIdx].pt).astype(int))
        end = tuple(np.round(k2[match.trainIdx].pt).astype(int) + [224, 0])
        color = (20, 180, 130) if keep else (235, 85, 70)
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 2, color, 1)
        cv2.circle(canvas, end, 2, color, 1)
    projected = cv2.perspectiveTransform(source[inliers].reshape(-1, 1, 2), homography).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - destination[inliers]) ** 2, axis=1))))
    result.update(homography=homography, inliers=int(inliers.sum()), reprojection_rmse=rmse)
    result['summary'] = (f"{len(good)} ratio-test matches; {result['inliers']} RANSAC inliers "
                         f"({result['inliers']/len(good):.1%}); inlier reprojection RMSE {rmse:.2f} pixels. "
                         'Source on the left; an 8-degree rotated, 0.95-scale copy on the right. '
                         'Green lines are inliers; red lines are rejected matches. RANSAC threshold: 5 pixels.')
    return result


def registration(rgb):
    """Compatibility helper for callers that only need the measured summary."""
    return registration_analysis(rgb)['summary']
