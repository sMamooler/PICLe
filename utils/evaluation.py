import ssl
from collections import defaultdict

import nltk
import regex as re
from nltk.stem import WordNetLemmatizer

### for nltk downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("punkt")
from nltk.corpus import wordnet


def _tokenize_with_special_characters(text):
    """Tokenize a text while keeping special characters and numbers as separate tokens."""
    pattern = r"[a-zA-Z']+|[0-9]+|[\W]"
    tokens = re.findall(pattern, text)
    tokens = [token for token in tokens if token.strip()]

    return tokens


def _find_subsequence_index(main_sequence, subsequence, start_index=0):
    """Finds the index of the first appearance of an ordered set of items (subsequence) in a referenced ordered set of items (main_sequence). Returns -1 if the subsequence is not found."""
    len_main = len(main_sequence)
    len_sub = len(subsequence)

    if len_sub == 0:
        return 0  # Empty subsequence is always found at index 0
    if len_sub > len_main:
        return -1  # Subsequence longer than main sequence can't be found

    for i in range(start_index, len_main - len_sub + 1):
        if main_sequence[i : i + len_sub] == subsequence:
            return i

    return -1  # Subsequence not found


# TODO(smamooler): remove the deprecated function
def entity_list_to_iob_format_deprecated(
    context: str, entities: list[str]
) -> list[str]:
    """[Deprecated] Convert a list of entities to IOB format.
    - If an entity is repeated multiple times in the text, only the first appearance is considered for the evaluation.
    - If two entities have overlapping tokens, only the first one that appears in the list us considered for the evaluation.
    """
    entities = entities.copy()
    original_entities = _tokenize_with_special_characters(context.lower())
    bio_entities = ["O" for _ in original_entities]
    for entity in entities:
        entity_tokens = _tokenize_with_special_characters(entity.lower())
        # The index() method only returns the first occurrence of the matching element.
        try:
            start_idx = original_entities.index(entity_tokens[0])
        except ValueError:
            print(f"Entity {entity} not found in context {context}")
            continue
        bio_entities[start_idx] = f"B"
        for i in range(1, len(entity_tokens)):
            bio_entities[start_idx + i] = f"I"

    return bio_entities


def entity_list_to_iob_format(context: str, entities: list[str]) -> list[str]:
    """Convert a list of entities to IOB format.

    Args:
        context (str): The context where the entities are located.
        entities (list[str]): The list of entities extracted from the context.

    Raises:
        ValueError: If an entity is not found in the context or has no tokens.

    Returns:
        list[str]: The list of tokens in IOB format.
    """
    original_entities = _tokenize_with_special_characters(context.lower())
    iob_entities = ["O" for _ in original_entities]
    # sort the list of entities by length in descending order
    entities = sorted(entities, key=lambda x: len(x), reverse=True)
    entity2start_index = defaultdict(int)
    for entity in entities:
        if entity.lower() not in context.lower():
            print(f"Entity {entity} not found in context {context}")
            continue

        entity_tokens = _tokenize_with_special_characters(entity.lower())
        try:
            # start_idx = original_entities.index(entity_tokens[0])
            start_idx = _find_subsequence_index(
                original_entities, entity_tokens, start_index=entity2start_index[entity]
            )
            entity2start_index[entity] = start_idx + 1
        except ValueError:
            print(f"Entity {entity} not found in context {context}")
            continue
        except IndexError:
            if len(entity.strip()) == 0:
                continue
            else:
                raise ValueError(f"Entity {entity} has no tokens")

        if iob_entities[start_idx] == "O":
            iob_entities[start_idx] = "B"
        for i in range(1, len(entity_tokens)):
            try:
                if iob_entities[start_idx + i] == "O":
                    iob_entities[start_idx + i] = "I"
            except IndexError:
                print(f"Entity {entity} not found in context {context}")

    return iob_entities


def _lemmatize(phrase: str):
    """Lemmatizes a phrase using the WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()

    def pos_tagger(nltk_tag):
        if nltk_tag.startswith("J"):
            return wordnet.ADJ
        elif nltk_tag.startswith("V"):
            return wordnet.VERB
        elif nltk_tag.startswith("N"):
            return wordnet.NOUN
        elif nltk_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(phrase))
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lemmatized_phrase = []
    for word, tag in wordnet_tagged:

        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_phrase.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_phrase.append(lemmatizer.lemmatize(word, tag))

    lemmatized_phrase = " ".join(lemmatized_phrase)

    return lemmatized_phrase


def evaluate_entities(predictions: list[str], ground_truth: list[str]) -> tuple:
    """Evaluates extracted entities against ground truth entities for a sample.

    Args:
        predictions (list[str]): list of predicted entities for a sample
        ground_truth (list[str]): list of ground truth entities for a sample

    Returns:
        tuple: precision, recall, f1 score, true positives, false positives, false negatives, and true positive score. true positive score takes into account the partail overlaps between the predicted and ground truth entities.
    """
    if not len(ground_truth) and not len(predictions):
        return 1, 1, 1, [], [], [], 0
    elif not len(ground_truth):
        return 0, 1, 0, [], predictions, [], 0
    else:
        predictions = list(map(lambda x: _lemmatize(x.lower()), predictions))
        ground_truth = list(map(lambda x: _lemmatize(x.lower()), ground_truth))

        tp = []
        fp = []
        fn = []

        tp_score = 0

        nb_ground_truth = len(ground_truth)
        for pred in predictions:
            if pred in ground_truth:
                tp.append(pred)
                ground_truth.remove(pred)
                tp_score += 1
            # for cases like "pecam-1 (cd31)"
            # elif re.search(r" \(.*?\)", pred):
            #     pred1 = re.search(r"\(.*?\)", pred).group(0).replace("(", "").replace(")", "").strip()
            #     pred2 = re.sub(r"\(.*?\)", "", pred)
            #     # print("(", pred1, "and", pred2, ")")
            #     if pred1 in ground_truth:
            #         tp.append(pred1)
            #         ground_truth.remove(pred1)
            #     elif pred2 in ground_truth:
            #         tp.append(pred2)
            #         ground_truth.remove(pred2)
            else:
                # accept partial matches too
                for gt in ground_truth:
                    if pred in gt or gt in pred:
                        ground_truth.remove(gt)
                        nb_gt_tokens = len(gt.split())
                        nb_pred_tokens = len(pred.split())
                        tp_score += min(nb_gt_tokens, nb_pred_tokens) / max(
                            nb_gt_tokens, nb_pred_tokens
                        )
                        break

                fp.append(pred)

        fn = [e for e in ground_truth if e not in tp]

        prc = len(tp) / len(predictions) if len(predictions) else 0
        rec = len(tp) / nb_ground_truth
        f1 = 2 * rec * prc / (prc + rec) if prc + rec else 0

        return prc, rec, f1, tp, fp, fn, tp_score


def _find_substring_indices(sentence, substring):
    """Find the start and end indices of all occurrences of a substring in a sentence"""
    indices = []
    start = 0
    while start < len(sentence):

        if len(substring.split()) > 1:
            pattern = re.compile(rf"{re.escape(substring)}", re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{re.escape(substring)}\b", re.IGNORECASE)
        match_ = pattern.search(sentence, start)
        if match_ is None:
            break

        start = match_.span()[0] + start
        end = start + len(match_.group(0)) - 1
        indices.append((start, end))
        start = end + 1

    return indices


def post_process_extractions(
    predictions: list[str],
    context: str,
    treat_special_tokens: bool = False,
    resolve_overlapping_entities: bool = False,
) -> list[str]:
    """
    Processes the extracted entities by
    1. remvoing the ones that are not in the text
    2. repeating the predicted entities that are in the text as many times as they appear in the text
    3. ordering the entities according to their appearance in the text

    Args:
        predictions (list[str]): list of predicted entities
        context (str): the text
        treat_special_tokens (bool): whether to treat special tokens like - in the text
        resolve_overlapping_entities (bool): whether to resolve overlapping entities

    Returns:
        list[str]: list of post-processed entities
    """
    # in bc2gm dataset there is sapce before and after - in the text, whihch is why we need to adapt the predictions accordingly
    if treat_special_tokens:
        for ent_i, entity in enumerate(predictions):

            pattern = re.compile(r"(?<=[^\s])-|-(?=[^\s])")
            predictions[ent_i] = pattern.sub(" - ", entity)

            pattern = re.compile(r"\(([^)]+)\)")
            predictions[ent_i] = pattern.sub(r"( \1 )", predictions[ent_i])

    # order the entities
    index2entity = {}
    for entity in set(predictions):
        if len(entity) > 1:
            indices = _find_substring_indices(context.lower(), entity.lower())
            for index in indices:
                index2entity[index] = entity

    if resolve_overlapping_entities:
        # find overlapping spans and remove the shorter span
        indices = list(index2entity.keys())
        indices_to_remove = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if indices[i][0] <= indices[j][0] and indices[i][1] >= indices[j][1]:
                    indices_to_remove.append(indices[j])
                elif indices[i][0] >= indices[j][0] and indices[i][1] <= indices[j][1]:
                    indices_to_remove.append(indices[i])

        for index in set(indices_to_remove):
            index2entity.pop(index)

    ordered_entites = [
        value for _, value in sorted(index2entity.items(), key=lambda x: x[0])
    ]

    return ordered_entites


def ner_list_eval(predictions: list[str], ground_truth: list[str]) -> dict[str, float]:
    """
    Computes the Micor and span precision, recall, and F1 score for entity extraction. This function evaluated a list of extracted entities against a list of ground truth entities.

    Args:
        predictions (list[str]): list of lists of predicted entities for all samples in the dataset
        ground_truth (list[str]): list of lists of ground truth entities for all samples in the dataset

    Returns:
        dict[str, float]: dictionary containing the micro and span precision, recall, and F1 score
    """
    tps = []
    fps = []
    fns = []
    global_tp_score = 0

    precisions = []
    recalls = []
    f1s = []

    for i in range(len(predictions)):

        prec, rec, f1, tp, fp, fn, tp_score = evaluate_entities(
            predictions[i], ground_truth[i]
        )
        global_tp_score += tp_score

        tps.extend(tp)
        fps.extend(fp)
        fns.extend(fn)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    micro_prc = (
        len(tps) / sum([len(prd) for prd in predictions]) if len(tps) + len(fps) else 0
    )
    micro_rec = (
        len(tps) / sum([len(gt) for gt in ground_truth]) if len(tps) + len(fns) else 0
    )
    micro_f1 = (
        2 * micro_prc * micro_rec / (micro_prc + micro_rec)
        if micro_prc + micro_rec
        else 0
    )

    span_prc = (
        global_tp_score / sum([len(prd) for prd in predictions])
        if sum([len(prd) for prd in predictions])
        else 0
    )
    span_rec = (
        global_tp_score / sum([len(gt) for gt in ground_truth])
        if sum([len(gt) for gt in ground_truth])
        else 0
    )
    span_f1 = (
        2 * span_prc * span_rec / (span_prc + span_rec) if span_prc + span_rec else 0
    )

    metrics = {
        "micro_prc": micro_prc,
        "micro_rec": micro_rec,
        "micro_f1": micro_f1,
        "span_prc": span_prc,
        "span_rec": span_rec,
        "span_f1": span_f1,
    }

    return metrics
