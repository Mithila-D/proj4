"""
Query Expansion — Medical Ontology  (Tasks 1, 2, 3 — FULLY IMPLEMENTED)
========================================================================
Maps lay / colloquial medical vocabulary to formal clinical terminology.

The vocabulary mismatch problem:
  A healthcare rep searches: "heart attack treatment aspirin antiplatelet"
  The most relevant document says: "myocardial infarction antiplatelet therapy clopidogrel"

  TF-IDF / BM25 will score this document POORLY because:
    "heart attack" does not appear in the document
    "aspirin" appears but only in passing

Query expansion is a lightweight pre-processing step that:
  1. Scans the query for known lay terms
  2. Adds their clinical synonyms to the query
  3. Downstream retrieval operates on the enriched query
"""

from __future__ import annotations

import re
from typing import Optional

from schemas import QueryExpansion

# ---------------------------------------------------------------------------
# Medical Ontology  (UMLS / SNOMED-CT inspired — simplified for teaching)
# ---------------------------------------------------------------------------

MEDICAL_ONTOLOGY: dict[str, list[str]] = {
    # Cardiovascular
    "heart attack":         ["myocardial infarction", "STEMI", "NSTEMI",
                              "acute coronary syndrome"],
    "mini heart attack":    ["NSTEMI", "non-ST-elevation myocardial infarction",
                              "unstable angina"],
    "blocked artery":       ["coronary artery occlusion", "coronary stenosis",
                              "atherosclerotic plaque"],
    "clot busting":         ["thrombolysis", "fibrinolytic therapy", "alteplase",
                              "tenecteplase"],
    "chest pain":           ["angina", "angina pectoris", "precordial pain",
                              "substernal discomfort"],
    "widowmaker":           ["left anterior descending artery occlusion",
                              "LAD occlusion", "anterior STEMI"],
    "stent":                ["percutaneous coronary intervention", "PCI",
                              "coronary stent", "drug-eluting stent"],

    # Anticoagulation
    "blood thinner":        ["anticoagulant", "anticoagulation", "warfarin",
                              "heparin", "low molecular weight heparin",
                              "direct oral anticoagulant"],
    "blood clot":           ["thrombosis", "thrombus", "venous thromboembolism",
                              "deep vein thrombosis", "pulmonary embolism"],
    "blood clot leg":       ["deep vein thrombosis", "DVT", "venous thrombosis"],
    "clot lung":            ["pulmonary embolism", "pulmonary thromboembolism"],
    "new blood thinner":    ["direct oral anticoagulant", "DOAC", "NOAC",
                              "rivaroxaban", "apixaban", "dabigatran"],
    "rat poison pill":      ["warfarin", "vitamin K antagonist"],

    # Heart failure
    "water pill":           ["diuretic", "loop diuretic", "furosemide",
                              "bumetanide", "thiazide"],
    "fluid pill":           ["diuretic", "loop diuretic", "furosemide"],
    "weak heart":           ["heart failure", "reduced ejection fraction",
                              "left ventricular systolic dysfunction"],
    "breathless flat":      ["orthopnoea", "paroxysmal nocturnal dyspnoea"],
    "shock device":         ["implantable cardioverter-defibrillator", "ICD",
                              "cardiac resynchronisation therapy"],
    "drowning in fluid":    ["pulmonary oedema", "acute decompensated heart failure",
                              "congestive heart failure"],

    # Diabetes
    "high blood sugar":     ["hyperglycaemia", "type 2 diabetes mellitus",
                              "elevated plasma glucose"],
    "sugar diabetes":       ["type 2 diabetes mellitus", "diabetes mellitus",
                              "hyperglycaemia"],
    "low blood sugar":      ["hypoglycaemia", "hypoglycemic episode"],
    "hypo":                 ["hypoglycaemia", "hypoglycemic episode"],
    "slimming jab":         ["GLP-1 receptor agonist", "semaglutide", "liraglutide",
                              "weight reduction injection"],
    "diabetes injection":   ["GLP-1 receptor agonist", "insulin", "semaglutide",
                              "liraglutide"],
    "sugar tablet":         ["metformin", "oral hypoglycaemic agent",
                              "antidiabetic medication"],
    "sugar test three months": ["HbA1c", "glycated haemoglobin",
                                "three-month blood glucose average"],
    "sugar sensor":         ["continuous glucose monitor", "CGM", "flash glucose monitor"],
    "insulin pump":         ["continuous subcutaneous insulin infusion", "CSII"],

    # Hypertension
    "high blood pressure":  ["hypertension", "arterial hypertension",
                              "elevated blood pressure"],
    "blood pressure tablet": ["antihypertensive", "antihypertensive agent",
                               "ACE inhibitor", "ARB", "calcium channel blocker"],
    "white coat":           ["white-coat hypertension", "white-coat effect",
                              "office hypertension"],

    # Stroke
    "brain attack":         ["ischaemic stroke", "cerebrovascular accident",
                              "acute stroke"],
    "mini stroke":          ["transient ischaemic attack", "TIA"],
    "brain clot":           ["ischaemic stroke", "cerebral thrombosis",
                              "middle cerebral artery occlusion"],
    "brain bleed":          ["intracranial haemorrhage", "intracerebral haemorrhage",
                              "haemorrhagic stroke"],
    "stroke clot removal":  ["mechanical thrombectomy", "endovascular thrombectomy",
                              "intra-arterial thrombolysis"],
    "face drooping":        ["facial palsy", "facial droop", "FAST symptoms"],

    # Sepsis / infection
    "blood poisoning":      ["sepsis", "septicaemia", "bacteraemia"],
    "septic shock":         ["sepsis-induced hypotension", "septic shock",
                              "vasopressor-dependent sepsis"],
    "blood pressure crash": ["septic shock", "haemodynamic instability",
                              "vasodilatory shock"],
    "infection marker":     ["C-reactive protein", "CRP", "procalcitonin",
                              "white cell count"],
    "white cells":          ["white blood cell count", "leukocyte count",
                              "neutrophilia", "leucocytosis"],
    "lactic acid":          ["lactate", "serum lactate", "hyperlactataemia"],
    "drip antibiotics":     ["intravenous antibiotics", "IV antimicrobial therapy",
                              "parenteral antibiotics"],

    # Respiratory
    "puffer":               ["inhaler", "bronchodilator", "short-acting beta-agonist",
                              "salbutamol"],
    "reliever inhaler":     ["short-acting beta-agonist", "SABA", "salbutamol",
                              "rescue inhaler"],
    "home oxygen":          ["long-term oxygen therapy", "LTOT",
                              "supplemental oxygen"],
    "lung scarring":        ["pulmonary fibrosis", "interstitial lung disease",
                              "idiopathic pulmonary fibrosis"],
    "stiff lungs":          ["restrictive lung disease", "pulmonary fibrosis",
                              "decreased lung compliance"],
    "sleep apnea mask":     ["CPAP", "continuous positive airway pressure",
                              "non-invasive ventilation"],

    # Renal
    "kidney failure":       ["renal failure", "acute kidney injury",
                              "chronic kidney disease", "end-stage renal disease"],
    "kidney washout":       ["haemodialysis", "renal replacement therapy",
                              "peritoneal dialysis"],
    "kidney number":        ["serum creatinine", "estimated GFR",
                              "glomerular filtration rate"],
    "contrast dye kidney":  ["contrast-induced nephropathy",
                              "contrast-induced acute kidney injury"],
    "foamy urine":          ["proteinuria", "nephrotic syndrome",
                              "heavy proteinuria"],
    "protein urine":        ["proteinuria", "albuminuria", "nephrotic range proteinuria"],
    "kidney anaemia":       ["anaemia of chronic kidney disease",
                              "renal anaemia", "erythropoietin deficiency"],
    "tired kidney":         ["chronic kidney disease", "renal insufficiency"],

    # Drug interactions
    "blood thinner antibiotic": ["anticoagulant-antibiotic interaction",
                                  "warfarin potentiation", "INR increase"],
    "grapefruit medication":    ["cytochrome P450 3A4 inhibition",
                                  "CYP3A4 inhibitor", "grapefruit-drug interaction"],
    "cholesterol tablet muscle pain": ["statin myopathy", "statin-induced myalgia",
                                        "rhabdomyolysis"],
    "antidepressant interaction":     ["serotonin syndrome", "SSRI interaction",
                                       "monoamine oxidase inhibitor interaction"],
    "serotonin overload":             ["serotonin syndrome", "serotonin toxicity"],
    "painkiller blood thinner":       ["NSAID anticoagulant interaction",
                                       "aspirin warfarin interaction",
                                       "ibuprofen anticoagulant"],
    "herbal supplement interaction":  ["St John's Wort interaction",
                                       "CYP induction", "herb-drug interaction"],

    # =====================================================================
    # Task 1: >= 15 new mappings (student-added lay terms from testset analysis)
    # =====================================================================

    # Eye / Vision
    "sugar eye":            ["diabetic retinopathy", "macular oedema",
                              "vitreous haemorrhage", "fundoscopy"],
    "blurry vision diabetes": ["diabetic macular oedema", "diabetic retinopathy",
                                "refractive change hyperglycaemia"],
    "cloudy lens":          ["cataract", "lens opacity", "phacoemulsification"],

    # Foot / Peripheral
    "sugar foot":           ["diabetic foot ulcer", "diabetic neuropathy",
                              "peripheral vascular disease"],
    "dead foot":            ["peripheral arterial disease", "critical limb ischaemia",
                              "gangrene"],
    "pins and needles":     ["peripheral neuropathy", "paraesthesia",
                              "sensory neuropathy"],

    # Liver
    "fatty liver":          ["non-alcoholic fatty liver disease", "NAFLD",
                              "hepatic steatosis", "non-alcoholic steatohepatitis"],
    "liver failure":        ["acute liver failure", "hepatic encephalopathy",
                              "coagulopathy liver disease"],
    "yellow skin":          ["jaundice", "icterus", "hyperbilirubinaemia"],

    # Mental health / Neurology
    "memory loss":          ["dementia", "Alzheimer disease", "cognitive impairment",
                              "mild cognitive impairment"],
    "fits":                 ["seizure", "epilepsy", "convulsion",
                              "tonic-clonic seizure"],
    "shaking":              ["tremor", "Parkinson disease", "essential tremor",
                              "intention tremor"],

    # Cancer
    "chemo":                ["chemotherapy", "cytotoxic therapy",
                              "antineoplastic therapy"],
    "radiotherapy burn":    ["radiation dermatitis", "radiation skin toxicity",
                              "acute radiation injury"],

    # Bone / Joint
    "brittle bones":        ["osteoporosis", "low bone mineral density",
                              "fragility fracture", "DEXA scan"],
    "joint replacement":    ["total hip arthroplasty", "total knee replacement",
                              "prosthetic joint"],
    "swollen joints":       ["arthritis", "rheumatoid arthritis",
                              "synovitis", "joint effusion"],

    # Thyroid
    "underactive thyroid":  ["hypothyroidism", "levothyroxine",
                              "Hashimoto thyroiditis"],
    "overactive thyroid":   ["hyperthyroidism", "thyrotoxicosis",
                              "Graves disease", "propylthiouracil"],

    # Allergy / Immune
    "allergic reaction":    ["anaphylaxis", "hypersensitivity reaction",
                              "angioedema", "urticaria"],
    "epipen":               ["adrenaline auto-injector", "epinephrine",
                              "anaphylaxis management"],
}


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------

class QueryExpander:
    """
    Expands medical queries using MEDICAL_ONTOLOGY synonym lookup.

    Usage:
        expander = QueryExpander()
        result   = expander.expand("heart attack treatment aspirin")
        # result.expansion_query contains original + clinical synonyms
    """

    def __init__(
        self,
        ontology:      dict[str, list[str]] | None = None,
        max_synonyms:  int = 4,
        deduplicate:   bool = True,
    ) -> None:
        self.ontology     = ontology or MEDICAL_ONTOLOGY
        self.max_synonyms = max_synonyms
        self.deduplicate  = deduplicate

        # Pre-compile sorted patterns (longest-first to avoid partial matches)
        self._sorted_keys = sorted(self.ontology.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------
    # Task 2: IMPLEMENTED — expand()
    # ------------------------------------------------------------------

    def expand(self, query: str) -> QueryExpansion:
        """
        Expand the query by appending clinical synonyms for recognised lay terms.

        Algorithm:
          1. Lowercase the query.
          2. Iterate over self._sorted_keys (longest first to avoid partial matches).
          3. If a key is found as a whole-word match, record an ontology hit and
             collect up to self.max_synonyms clinical synonyms.
          4. Deduplicate added terms (if self.deduplicate=True).
          5. Build expansion_query = original_query + " " + " ".join(expanded_terms)
          6. Return a QueryExpansion (Pydantic model).
        """
        q_lower = query.lower()
        ontology_hits: dict[str, list[str]] = {}
        expanded_terms: list[str] = []
        seen: set[str] = set()

        for key in self._sorted_keys:
            # Whole-phrase match (word boundary aware)
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, q_lower):
                synonyms = self.ontology[key][: self.max_synonyms]
                ontology_hits[key] = synonyms
                for syn in synonyms:
                    syn_lower = syn.lower()
                    if self.deduplicate:
                        if syn_lower not in seen:
                            seen.add(syn_lower)
                            expanded_terms.append(syn)
                    else:
                        expanded_terms.append(syn)

        if expanded_terms:
            expansion_query = query + " " + " ".join(expanded_terms)
        else:
            expansion_query = query

        return QueryExpansion(
            original_query=query,
            expanded_terms=expanded_terms,
            expansion_query=expansion_query,
            n_terms_added=len(expanded_terms),
            ontology_hits=ontology_hits,
        )

    # ------------------------------------------------------------------
    # Task 3: IMPLEMENTED — score_expansion_quality()
    # ------------------------------------------------------------------

    def score_expansion_quality(
        self,
        expansion: QueryExpansion,
        corpus_vocab: set[str],
    ) -> dict[str, float]:
        """
        Measure how many of the expanded terms actually appear in the corpus.

        A term is "in corpus" if ANY token of the term phrase appears in corpus_vocab.
        This tells us whether expansion added useful signal or noise.

        Returns:
            dict with:
              "coverage"          : fraction of expanded_terms found in corpus_vocab
              "n_terms_in_corpus" : count found
              "n_terms_not_found" : count not found
        """
        if not expansion.expanded_terms:
            return {
                "coverage": 0.0,
                "n_terms_in_corpus": 0,
                "n_terms_not_found": 0,
            }

        n_found = 0
        for term in expansion.expanded_terms:
            # A multi-word term is "present" if any constituent word is in vocab
            tokens = term.lower().split()
            if any(t in corpus_vocab for t in tokens):
                n_found += 1

        total = len(expansion.expanded_terms)
        coverage = n_found / total

        return {
            "coverage": round(coverage, 4),
            "n_terms_in_corpus": n_found,
            "n_terms_not_found": total - n_found,
        }

    # ------------------------------------------------------------------
    # Provided helper: detect lay terms in a query
    # ------------------------------------------------------------------

    def detect_lay_terms(self, query: str) -> list[str]:
        """Return list of ontology keys found in the query (lay terms detected)."""
        q_lower = query.lower()
        found = []
        for key in self._sorted_keys:
            if re.search(r'\b' + re.escape(key) + r'\b', q_lower):
                found.append(key)
        return found

    def build_corpus_vocab(self, corpus_texts: list[str]) -> set[str]:
        """Build a token vocabulary from raw text list."""
        vocab: set[str] = set()
        for text in corpus_texts:
            for tok in text.lower().split():
                vocab.add(tok.strip(".,;:()[]\"'"))
        return vocab

    def summary(self) -> None:
        """Print a summary of the ontology."""
        print(f"Medical ontology: {len(self.ontology)} lay-term entries")
        total_synonyms = sum(len(v) for v in self.ontology.values())
        print(f"Total synonyms  : {total_synonyms}")
        avg = total_synonyms / max(1, len(self.ontology))
        print(f"Avg synonyms/term: {avg:.1f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    expander = QueryExpander()
    expander.summary()
    print()

    demo_queries = [
        "heart attack treatment aspirin antiplatelet",
        "water pill swollen ankles fluid retention",
        "brain bleed anticoagulation stop restart",
        "blood poisoning signs ICU drip antibiotics",
        "sugar diabetes HbA1c target elderly patient",
        "sugar eye screening annual check",
        "brittle bones calcium vitamin D fracture",
    ]

    for q in demo_queries:
        exp = expander.expand(q)
        print(f"Query : {q}")
        print(f"  Hits     : {list(exp.ontology_hits.keys())}")
        print(f"  Added    : {exp.n_terms_added} terms")
        print(f"  Expanded : {exp.expansion_query[:100]}")
        print()
