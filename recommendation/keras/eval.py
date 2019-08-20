import contextlib
from typing import Dict, List, Tuple

import numpy as np
from collections import defaultdict

from sklearn import metrics
from keras.engine.training import Model
from keras_retinanet.utils.eval import _get_detections, _get_annotations
from keras_retinanet.utils.compute_overlap import compute_overlap

from recommendation.data import LocalizationImageGenerator, ImageIterator

COMPETITION_IOU_THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]


class EvaluationReport(object):
    def __init__(self, acc: float, precision: float, recall: float, f1_score: float,
                 confusion_matrix: List[List[float]], misclassified_inputs: Dict[str, float] = None):
        self.acc = acc
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.confusion_matrix = confusion_matrix
        self.misclassified_inputs = misclassified_inputs


def evaluate_locator(model: Model, generator: LocalizationImageGenerator, score_threshold=0.05,
                     iou_thresholds: List[float] = COMPETITION_IOU_THRESHOLDS, max_detections=100, save_path=None) \
        -> Dict[float, Dict[int, Tuple[float, int, int, int, int]]]:
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     save_path=save_path)
    all_annotations = _get_annotations(generator)
    return evaluate_detections(generator, all_detections, all_annotations, iou_thresholds)


def evaluate_detections(generator: LocalizationImageGenerator,
                        all_detections: List[List[np.ndarray]],
                        all_annotations: List[List[np.ndarray]],
                        iou_thresholds: List[float] = COMPETITION_IOU_THRESHOLDS) \
        -> Dict[float, Dict[int, Tuple[float, int, int, int, int]]]:
    average_precisions: Dict[float, Dict[int, Tuple[float, int, int, int, int]]] = defaultdict(dict)

    for label in range(generator.num_classes()):
        false_positives: Dict[float, float] = {iou_threshold: 0.0 for iou_threshold in iou_thresholds}
        true_positives: Dict[float, float] = {iou_threshold: 0.0 for iou_threshold in iou_thresholds}
        false_negatives: Dict[float, float] = {iou_threshold: 0.0 for iou_threshold in iou_thresholds}
        num_annotations = 0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations: Dict[float, list] = defaultdict(list)

            for d in detections:
                if annotations.shape[0] == 0:
                    for iou_threshold in iou_thresholds:
                        false_positives[iou_threshold] += 1.0
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                for iou_threshold in iou_thresholds:
                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations[iou_threshold]:
                        true_positives[iou_threshold] += 1.0
                        detected_annotations[iou_threshold].append(assigned_annotation)
                    else:
                        false_positives[iou_threshold] += 1.0

            for iou_threshold in iou_thresholds:
                for i in range(annotations.shape[0]):
                    if i not in detected_annotations[iou_threshold]:
                        false_negatives[iou_threshold] += 1.0

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            for iou_threshold in iou_thresholds:
                average_precisions[iou_threshold][label] = 0.0, 0, 0, 0, 0
            continue

        # sort by score
        for iou_threshold in iou_thresholds:
            # compute average precision
            average_precision = true_positives[iou_threshold] / (
                true_positives[iou_threshold] + false_positives[iou_threshold] + false_negatives[iou_threshold])
            average_precisions[iou_threshold][label] = average_precision, \
                                                       true_positives[iou_threshold], \
                                                       false_positives[iou_threshold], \
                                                       false_negatives[iou_threshold], \
                                                       num_annotations

    return average_precisions


def generate_metrics(metrics_per_iou: Dict[float, Dict[int, Tuple[float, int, int, int, int]]],
                     val: bool, weighted_average=False, verbose: int = 1):
    tag_prefix = "val_" if val else ""

    metrics = dict()

    mean_ap_sum = 0.0
    true_positives_sum = np.int(0)
    false_positives_sum = np.int(0)
    false_negatives_sum = np.int(0)
    for iou_threshold, iou_metrics in metrics_per_iou.items():
        # compute per class average precision
        total_instances = []
        precisions = []
        true_positives: np.int = np.int(0)
        false_positives: np.int = np.int(0)
        false_negatives: np.int = np.int(0)
        for label, (average_precision, tp, fp, fn, num_annotations) in iou_metrics.items():
            total_instances.append(num_annotations)
            precisions.append(average_precision)
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        if weighted_average:
            mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        mean_ap_sum += mean_ap
        true_positives_sum += true_positives
        false_positives_sum += false_positives
        false_negatives_sum += false_negatives

        mAP_tag = "{}mAP(@IoU>={:.2f})".format(tag_prefix, iou_threshold)
        true_positives_tag = "{}true_positives(@IoU>={:.2f})".format(tag_prefix, iou_threshold)
        false_positives_tag = "{}false_positives(@IoU>={:.2f})".format(tag_prefix, iou_threshold)
        false_negatives_tag = "{}false_negatives(@IoU>={:.2f})".format(tag_prefix, iou_threshold)

        metrics[mAP_tag] = np.float32(mean_ap)
        metrics[true_positives_tag] = np.float32(true_positives)
        metrics[false_positives_tag] = np.float32(false_positives)
        metrics[false_negatives_tag] = np.float32(false_negatives)

        if verbose == 1:
            print('{}: {:.4f}'.format(mAP_tag, mean_ap))

    avg_mean_ap = mean_ap_sum / len(metrics_per_iou)
    avg_true_positives = true_positives_sum / len(metrics_per_iou)
    avg_false_positives = false_positives_sum / len(metrics_per_iou)
    avg_false_negatives = false_negatives_sum / len(metrics_per_iou)

    mAP_tag = "{}avg_mAP".format(tag_prefix)
    true_positives_tag = "{}avg_true_positives".format(tag_prefix)
    false_positives_tag = "{}avg_false_positives".format(tag_prefix)
    false_negatives_tag = "{}avg_false_negatives".format(tag_prefix)

    metrics[mAP_tag] = np.float32(avg_mean_ap)
    metrics[true_positives_tag] = np.float32(avg_true_positives)
    metrics[false_positives_tag] = np.float32(avg_false_positives)
    metrics[false_negatives_tag] = np.float32(avg_false_negatives)
    if verbose == 1:
        print('{}: {:.4f}'.format(mAP_tag, avg_mean_ap))

    return metrics


def pred_probas_for_classifier(model: Model, generator: ImageIterator) -> np.ndarray:
    with _ordered(generator) as ordered_generator:
        return model.predict_generator(ordered_generator, steps=len(ordered_generator), verbose=1).flatten()


def evaluate_classifier(generator: ImageIterator, filenames: List[str], ground_truths: np.ndarray,
                        probas: np.ndarray = None,
                        model: Model = None, threshold=0.5) -> EvaluationReport:
    with _ordered(generator) as ordered_generator:
        if probas is None:
            probas = pred_probas_for_classifier(model, ordered_generator)
        preds = (probas > threshold).astype(int)

        acc, precision, recall, f1_score, confusion_matrix = calculate_scores(ground_truths, preds)

        misclassified_filenames = np.array(filenames)[ground_truths != preds]
        misclassified_proba_preds = probas[ground_truths != preds]
        misclassified_inputs = {filename: float(proba_pred) for filename, proba_pred in
                                zip(misclassified_filenames, misclassified_proba_preds)}

        return EvaluationReport(acc, precision, recall, f1_score, confusion_matrix, misclassified_inputs)


def calculate_scores(trues: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float, float, List[List[float]]]:
    acc = metrics.accuracy_score(trues, preds)
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(trues, preds, average='binary')
    confusion_matrix = metrics.confusion_matrix(trues, preds).tolist()

    return acc, precision, recall, f1_score, confusion_matrix


@contextlib.contextmanager
def _ordered(generator: ImageIterator) -> ImageIterator:
    old_shuffle = generator.shuffle

    generator.shuffle = False
    generator.on_epoch_end()

    yield generator

    generator.shuffle = old_shuffle
    generator.on_epoch_end()
