from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from stopwords import STOPWORDS


DOMAIN_TERMS: tuple[str, ...] = (
    "abandon ship drills",
    "accident investigation",
    "administrative requirements",
    "air pollution",
    "alternative design and arrangements",
    "alternative fuel",
    "alternative fuels",
    "alternative maritime power",
    "annual ghg fuel contribution",
    "armed robbery",
    "ballast water management system",
    "ballast water management systems",
    "ballast water management convention",
    "ballast water management",
    "ballast water exchange",
    "ballast water record book",
    "ballast water",
    "ballast tanks",
    "basel convention",
    "black carbon",
    "black carbon emissions",
    "bunker delivery note",
    "bulk carriers",
    "bwms code",
    "carbon capture",
    "carbon capture and storage",
    "carbon intensity",
    "carbon intensity indicator",
    "carbon intensity reduction",
    "carbon pricing",
    "carbon dioxide",
    "circular economy",
    "challenging water quality",
    "cargo area",
    "cargo hold",
    "cargo holds",
    "cargo ship",
    "cargo ships",
    "cargo space",
    "cargo spaces",
    "cargo tank",
    "cargo tanks",
    "cargo transport units",
    "co2 system",
    "code of safety for diving systems",
    "commercial diving safety",
    "container ships",
    "control stations",
    "conventional ships",
    "cruise passenger ships",
    "cruise ships",
    "decarbonization measure",
    "default emission factors",
    "diesel gas oil",
    "domestic ferry safety",
    "draft guidelines",
    "draft interim guidelines",
    "duct penetrations",
    "electric cables",
    "emission control area",
    "emission control areas",
    "emission factors",
    "energy efficiency",
    "energy efficiency existing ship index",
    "engine room areas",
    "enclosed spaces",
    "escape route signs",
    "equipment location markings",
    "existing ships",
    "existing ship index",
    "exhaust gas cleaning systems",
    "fire alarm systems",
    "fire detection",
    "fire detection and alarm systems",
    "fire dampers",
    "fire door",
    "fire extinguishing capability",
    "fire extinguisher",
    "fire fighter's communication",
    "fire fighter's outfits",
    "fire fighters' communication",
    "fire fighters' outfits",
    "fire fighting",
    "fire-fighting",
    "fire fighting foams",
    "fire-fighting foams",
    "fire integrity",
    "fire insulation",
    "fire protection",
    "fire safety",
    "fixed carbon dioxide fire extinguishing systems",
    "fixed carbon dioxide fire-extinguishing systems",
    "fixed co2 fire extinguishing systems",
    "fixed fire extinguishing systems",
    "fixed fire-extinguishing systems",
    "fixed gas fire extinguishing systems",
    "fixed gas fire-extinguishing systems",
    "fixed water based fire fighting systems",
    "fixed water-based fire-fighting systems",
    "formal safety assessment",
    "free fall lifeboat",
    "free fall lifeboats",
    "fuel lifecycle",
    "fuel oil consumption",
    "fuel oil consumption data",
    "fuel oil consumption database",
    "fuel oil",
    "fuel tank",
    "fuel tanks",
    "gas carrier",
    "gas carriers",
    "gas freeing piping",
    "general cargo ships",
    "ghg fuel standard",
    "ghg emissions",
    "global integrated shipping information system",
    "goal based ship construction standards",
    "goal based measure",
    "greenhouse gas",
    "greenhouse gas emission",
    "greenhouse gas emissions",
    "grey water",
    "heavy fuel oil",
    "high pressure co2 cylinders",
    "high voltage shore connection",
    "hong kong convention",
    "hong kong international convention",
    "hydrostatic test",
    "igc code",
    "igf code",
    "iii code",
    "imdg code",
    "imo net zero framework",
    "imo ship fuel oil consumption database",
    "imsbc code",
    "incinerator spaces",
    "in water performance",
    "inflatable liferafts",
    "interim guidelines",
    "lifecycle assessment",
    "life saving appliances",
    "life-saving appliances",
    "lifejacket",
    "lifejackets",
    "lifeboat drills",
    "lifeboat release and retrieval systems",
    "lifeboat simulator training",
    "lifeboats",
    "liferaft",
    "liferafts",
    "lifting appliances",
    "liquefied hydrogen",
    "london convention",
    "low fire risk",
    "low carbon fuel",
    "low flashpoint fuels",
    "low voltage shore connection",
    "lsa code",
    "maritime autonomous surface ships",
    "maritime cyber risk management",
    "maritime ghg emissions pricing mechanism",
    "maritime industry",
    "maritime safety committee",
    "maritime safety information",
    "maritime security",
    "maritime transport",
    "marine diesel oil",
    "marine diesel engine",
    "marine diesel engines",
    "marine environment protection committee",
    "marine environmental protection committee",
    "marine fuel",
    "marine fuels",
    "marine fuel life cycle ghg analysis",
    "marine life",
    "marine plastic litter",
    "marine pollution",
    "marpol annex vi",
    "marpol convention",
    "mass code",
    "means of rescue",
    "methane emissions",
    "mid term measures",
    "net zero",
    "net zero emissions",
    "net zero ghg emissions",
    "new ships",
    "non combustible sills",
    "nox technical code",
    "oil fuel",
    "on board ships",
    "on load release capability",
    "on-load release capability",
    "on load release mechanisms",
    "on-load release mechanisms",
    "on shore power supply",
    "on-shore power supply",
    "onshore power supply",
    "onshore power supply ops service",
    "onshore power supply service",
    "onboard carbon capture",
    "onshore power supply service in port",
    "open deck",
    "open ro ro deck",
    "operational carbon intensity",
    "operational carbon intensity indicator",
    "operational carbon intensity indicators",
    "other low flashpoint fuels",
    "partially enclosed lifeboats",
    "passenger ship",
    "passenger ships",
    "perfluorooctane sulfonic acid",
    "pipe penetrations",
    "polar code",
    "port facility security level",
    "port state control",
    "port state",
    "port states",
    "power reserve",
    "protection of penetrations",
    "radio telephone apparatus",
    "record test data",
    "renewable fuel",
    "renewing falls",
    "rescue boat",
    "rescue boats",
    "rescue craft",
    "revised guidelines",
    "revised recommendation on testing of life saving appliances",
    "revised recommendation on testing of life-saving appliances",
    "ro ro passenger ships",
    "ro-ro passenger ships",
    "ro ro spaces",
    "ro-ro spaces",
    "safety management",
    "safety management system",
    "safety measures",
    "safety regulatory framework",
    "search and rescue",
    "security level",
    "seemp",
    "ship design",
    "ship energy efficiency management plan",
    "ship fuel oil consumption data",
    "ship operators",
    "ship security assessments",
    "ship systems and equipment",
    "ship type",
    "shipping companies",
    "shipping industry",
    "shipping sector",
    "short term measure",
    "shore connection",
    "shore power",
    "shore power supply",
    "ship port interface",
    "ship-port interface",
    "shore side electricity",
    "shore-side electricity",
    "single fall and hook systems",
    "solid bulk cargoes",
    "solas chapter",
    "solas convention",
    "solas lifejackets",
    "solas regulation",
    "special category spaces",
    "standardized life saving appliance evaluation and test report forms",
    "structural divisions",
    "stcw code",
    "stcw convention",
    "sub committee on ship systems and equipment",
    "technical cooperation",
    "test lifeboat",
    "testing of life saving appliances",
    "testing of life-saving appliances",
    "totally enclosed lifeboats",
    "two way portable radiotelephone apparatus",
    "unified interpretation",
    "vehicle decks",
    "water spray system",
    "well to wake",
    "waste stowage spaces",
    "winch brake",
    "world maritime university",
    "zero carbon shipping",
    "zero emission shipping fund",
    "zero or near zero ghg emission technologies",
)

DOMAIN_ENTITY_LABEL = "MARITIME_TERM"
DOMAIN_SPAN_KEY = "maritime_terms"
SPACY_MODEL = "en_core_web_sm"
ALLOWED_POS = {"NOUN", "PROPN", "ADJ"}


@dataclass(frozen=True)
class TokenRecord:
    text: str
    lemma: str
    pos: str
    tag: str
    ent_type: str
    is_domain_term: bool


def _normalize_term(term: str) -> str:
    return "_".join(part for part in term.lower().split() if part)


def _build_match_patterns(nlp: Language, phrases: Sequence[str]) -> list[Doc]:
    return [nlp.make_doc(phrase) for phrase in phrases]


@Language.factory("maritime_term_component")
def create_maritime_term_component(
    nlp: Language,
    name: str,
    phrases: Sequence[str] = DOMAIN_TERMS,
):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("MARITIME_TERM", _build_match_patterns(nlp, phrases))

    def maritime_term_component(doc: Doc) -> Doc:
        matches = matcher(doc)
        matched_spans = [Span(doc, start, end, label=DOMAIN_ENTITY_LABEL) for _, start, end in matches]
        spans = filter_spans(list(doc.ents) + matched_spans)
        doc.ents = spans
        doc.spans[DOMAIN_SPAN_KEY] = [span for span in spans if span.label_ == DOMAIN_ENTITY_LABEL]

        merge_targets = filter_spans([span for span in doc.spans[DOMAIN_SPAN_KEY]])
        with doc.retokenize() as retokenizer:
            for span in merge_targets:
                normalized = _normalize_term(span.text)
                attrs = {
                    "LEMMA": normalized,
                    "ENT_TYPE": DOMAIN_ENTITY_LABEL,
                }
                retokenizer.merge(span, attrs=attrs)
        return doc

    return maritime_term_component


@lru_cache(maxsize=1)
def get_nlp(model_name: str = SPACY_MODEL) -> Language:
    nlp = spacy.load(model_name)
    if "maritime_term_component" not in nlp.pipe_names:
        if "ner" in nlp.pipe_names:
            nlp.add_pipe("maritime_term_component", before="ner")
        else:
            nlp.add_pipe("maritime_term_component", last=True)
    return nlp


def iter_docs(texts: Iterable[str], batch_size: int = 32) -> Iterable[Doc]:
    valid_texts = [text if isinstance(text, str) else "" for text in texts]
    yield from get_nlp().pipe(valid_texts, batch_size=batch_size)


def analyze_text(text: str) -> dict[str, list[dict[str, str]]]:
    doc = get_nlp()(text if isinstance(text, str) else "")
    tokens = [
        {
            "text": token.text,
            "lemma": token.lemma_.lower(),
            "pos": token.pos_,
            "tag": token.tag_,
            "ent_type": token.ent_type_,
            "is_domain_term": str(token.ent_type_ == DOMAIN_ENTITY_LABEL),
        }
        for token in doc
        if not token.is_space
    ]
    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": str(ent.start_char),
            "end": str(ent.end_char),
        }
        for ent in doc.ents
    ]
    return {"tokens": tokens, "entities": entities}


def _normalized_token(token) -> str:
    if token.ent_type_ == DOMAIN_ENTITY_LABEL:
        return _normalize_term(token.text)
    lemma = token.lemma_.lower().strip()
    if not lemma:
        return ""
    return _normalize_term(lemma)


def extract_content_tokens(
    text: str,
    stopwords: set[str] | None = None,
    allowed_pos: set[str] | None = None,
) -> list[str]:
    doc = get_nlp()(text if isinstance(text, str) else "")
    effective_stopwords = STOPWORDS if stopwords is None else stopwords
    effective_pos = ALLOWED_POS if allowed_pos is None else allowed_pos

    tokens: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct or token.like_num:
            continue
        normalized = _normalized_token(token)
        if not normalized:
            continue
        if token.ent_type_ != DOMAIN_ENTITY_LABEL and token.pos_ not in effective_pos:
            continue
        if token.ent_type_ != DOMAIN_ENTITY_LABEL:
            if not normalized.replace("_", "").isalpha():
                continue
            if normalized in effective_stopwords:
                continue
            if len(normalized) <= 1:
                continue
        tokens.append(normalized)
    return tokens


def build_topic_analyzer(
    stopwords: set[str] | None = None,
    allowed_pos: set[str] | None = None,
):
    def analyzer(text: str) -> list[str]:
        return extract_content_tokens(text, stopwords=stopwords, allowed_pos=allowed_pos)

    return analyzer


if __name__ == "__main__":
    sample = (
        "The working group discussed marine plastic litter reduction and greenhouse gas "
        "emission measures for international shipping."
    )
    result = analyze_text(sample)
    print(result)