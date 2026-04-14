"""
Dataset Generator — Project 4: Semantic Search & Information Retrieval
=======================================================================
Produces two CSV files:

  clinical_corpus.csv   (2 000 clinical documents using formal medical terminology)
  query_testset.csv     (100 queries using lay / colloquial vocabulary + ground-truth labels)

The central challenge: the SAME clinical concept appears with different vocabulary
in queries versus documents.

  Query (patient / GP language)     Document (clinical language)
  --------------------------------   --------------------------------
  "heart attack"                  -> "myocardial infarction / STEMI"
  "blood thinner"                 -> "anticoagulant / heparin"
  "water pill"                    -> "diuretic / furosemide"
  "sugar diabetes"                -> "type 2 diabetes mellitus"
  "brain bleed"                   -> "intracranial haemorrhage"

A system that uses only TF-IDF / BM25 will MISS these matches because it relies
on lexical overlap.  Semantic retrieval + query expansion are needed.

Run:
    python data/generate_dataset.py
"""

from __future__ import annotations

import csv
import random
import textwrap
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Vocabulary — clinical corpus side (formal medical language)
# ---------------------------------------------------------------------------

TOPIC_VOCAB: dict[str, dict] = {
    "acute_coronary_syndrome": {
        "title_patterns": [
            "Management of {c} in hospitalised patients",
            "Percutaneous coronary intervention for {c}: a retrospective study",
            "{c} presenting with ST-elevation: outcomes analysis",
            "Dual antiplatelet therapy following {c}",
            "Risk stratification in {c} using troponin biomarkers",
        ],
        "concepts": [
            "myocardial infarction", "STEMI", "NSTEMI", "acute coronary syndrome",
            "percutaneous coronary intervention", "coronary artery disease",
            "left ventricular dysfunction", "troponin elevation", "ECG changes",
            "antiplatelet therapy", "clopidogrel", "ticagrelor", "aspirin",
            "thrombolysis", "fibrinolytic therapy", "stent implantation",
            "revascularisation", "cardiac catheterisation", "angioplasty",
            "atherosclerotic plaque rupture",
        ],
        "fillers": {
            "c": ["STEMI", "NSTEMI", "acute coronary syndrome",
                  "myocardial infarction", "unstable angina"],
        },
    },
    "anticoagulation_therapy": {
        "title_patterns": [
            "Direct oral anticoagulants versus warfarin in {c}",
            "Bleeding risk stratification with {c}",
            "Reversal agents for {c}: clinical review",
            "Perioperative management of {c}",
            "Dose adjustment of {c} in renal impairment",
        ],
        "concepts": [
            "anticoagulation", "warfarin", "heparin", "low molecular weight heparin",
            "direct oral anticoagulants", "rivaroxaban", "apixaban", "dabigatran",
            "international normalised ratio", "INR monitoring", "bleeding risk",
            "thromboembolic prophylaxis", "venous thromboembolism",
            "deep vein thrombosis", "pulmonary embolism", "atrial fibrillation",
            "factor Xa inhibition", "reversal agent", "idarucizumab",
            "vitamin K antagonist",
        ],
        "fillers": {
            "c": ["atrial fibrillation", "venous thromboembolism",
                  "deep vein thrombosis", "pulmonary embolism"],
        },
    },
    "heart_failure": {
        "title_patterns": [
            "Neurohormonal blockade in {c} with reduced ejection fraction",
            "Diuretic therapy titration in {c}",
            "Device therapy for {c}: ICD and CRT outcomes",
            "Hospital readmission rates in {c}: a quality improvement study",
            "SGLT2 inhibitors in {c}: mechanism and outcomes",
        ],
        "concepts": [
            "heart failure", "reduced ejection fraction", "preserved ejection fraction",
            "left ventricular systolic dysfunction", "BNP", "NT-proBNP",
            "loop diuretics", "furosemide", "bumetanide", "thiazide diuretics",
            "ACE inhibitors", "angiotensin receptor blockers", "beta-blockers",
            "aldosterone antagonists", "spironolactone", "sacubitril-valsartan",
            "SGLT2 inhibitors", "dapagliflozin", "empagliflozin",
            "implantable cardioverter-defibrillator", "cardiac resynchronisation",
            "pulmonary oedema", "orthopnoea", "paroxysmal nocturnal dyspnoea",
        ],
        "fillers": {
            "c": ["heart failure", "dilated cardiomyopathy",
                  "ischaemic cardiomyopathy", "chronic heart failure"],
        },
    },
    "diabetes_mellitus": {
        "title_patterns": [
            "Glycaemic control targets in {c}: updated guidelines",
            "Renal outcomes with {c} and SGLT2 inhibitor use",
            "Insulin intensification strategies in {c}",
            "Microvascular complications of {c}: screening and prevention",
            "GLP-1 receptor agonists for weight reduction in {c}",
        ],
        "concepts": [
            "type 2 diabetes mellitus", "hyperglycaemia", "HbA1c", "glycated haemoglobin",
            "insulin resistance", "pancreatic beta-cell dysfunction",
            "metformin", "sulfonylureas", "glipizide", "glibenclamide",
            "GLP-1 receptor agonists", "semaglutide", "liraglutide",
            "SGLT2 inhibitors", "canagliflozin", "DPP-4 inhibitors", "sitagliptin",
            "basal insulin", "prandial insulin", "continuous glucose monitoring",
            "diabetic nephropathy", "diabetic retinopathy", "peripheral neuropathy",
            "hypoglycaemia", "ketoacidosis",
        ],
        "fillers": {
            "c": ["type 2 diabetes mellitus", "poorly controlled diabetes",
                  "insulin-requiring diabetes", "diabetic patients"],
        },
    },
    "hypertension": {
        "title_patterns": [
            "Target blood pressure in {c}: systematic review",
            "Combination therapy for {c} in elderly patients",
            "Resistant {c}: evaluation and management",
            "White-coat {c} versus masked {c}: ambulatory monitoring",
            "Secondary causes of {c}: a diagnostic approach",
        ],
        "concepts": [
            "hypertension", "systolic blood pressure", "diastolic blood pressure",
            "ambulatory blood pressure monitoring", "white-coat hypertension",
            "resistant hypertension", "secondary hypertension",
            "renin-angiotensin-aldosterone system", "ACE inhibitors", "ramipril",
            "lisinopril", "angiotensin receptor blockers", "losartan",
            "calcium channel blockers", "amlodipine", "thiazide diuretics",
            "hydrochlorothiazide", "chlorthalidone", "beta-blockers",
            "end-organ damage", "left ventricular hypertrophy",
            "hypertensive retinopathy", "chronic kidney disease",
        ],
        "fillers": {
            "c": ["hypertension", "arterial hypertension", "elevated blood pressure"],
        },
    },
    "stroke": {
        "title_patterns": [
            "Thrombolysis time window in acute {c}",
            "Mechanical thrombectomy outcomes in {c}: multicentre study",
            "Secondary prevention after {c}: antithrombotic strategies",
            "Rehabilitation outcomes following {c}",
            "Cryptogenic {c}: patent foramen ovale closure versus medical therapy",
        ],
        "concepts": [
            "ischaemic stroke", "haemorrhagic stroke", "cerebrovascular accident",
            "transient ischaemic attack", "intracranial haemorrhage",
            "subarachnoid haemorrhage", "lacunar infarction",
            "middle cerebral artery occlusion", "mechanical thrombectomy",
            "intravenous thrombolysis", "alteplase", "tenecteplase",
            "NIHSS score", "modified Rankin scale", "diffusion-weighted MRI",
            "carotid stenosis", "atrial fibrillation", "anticoagulation",
            "dual antiplatelet therapy", "patent foramen ovale",
        ],
        "fillers": {
            "c": ["ischaemic stroke", "acute stroke", "cerebrovascular accident"],
        },
    },
    "sepsis": {
        "title_patterns": [
            "Hour-1 bundle compliance and mortality in {c}",
            "Antibiotic stewardship in {c}: de-escalation strategies",
            "Vasopressor selection in {c}-induced hypotension",
            "Lactate clearance as a prognostic marker in {c}",
            "Source control in abdominal {c}",
        ],
        "concepts": [
            "sepsis", "septic shock", "systemic inflammatory response syndrome",
            "bacteraemia", "blood culture", "broad-spectrum antibiotics",
            "empirical antibiotic therapy", "de-escalation", "carbapenem",
            "piperacillin-tazobactam", "vasopressors", "noradrenaline",
            "vasopressin", "fluid resuscitation", "crystalloid",
            "mean arterial pressure", "lactate", "organ dysfunction",
            "sequential organ failure assessment", "ICU admission",
            "procalcitonin", "C-reactive protein",
        ],
        "fillers": {
            "c": ["sepsis", "septic shock", "gram-negative bacteraemia",
                  "healthcare-associated sepsis"],
        },
    },
    "respiratory_disease": {
        "title_patterns": [
            "Inhaled corticosteroid step-down in {c}",
            "Non-invasive ventilation in {c} exacerbation",
            "Biologics for severe {c}: dupilumab and mepolizumab",
            "Pulmonary rehabilitation in {c}: quality-of-life outcomes",
            "Ventilator-associated pneumonia prevention in {c}",
        ],
        "concepts": [
            "chronic obstructive pulmonary disease", "asthma", "bronchiectasis",
            "pulmonary fibrosis", "respiratory failure", "hypoxaemia",
            "short-acting beta-agonist", "salbutamol", "inhaled corticosteroid",
            "budesonide", "fluticasone", "long-acting beta-agonist",
            "salmeterol", "formoterol", "LAMA", "tiotropium",
            "non-invasive ventilation", "CPAP", "BiPAP",
            "arterial blood gas", "oxygen saturation", "peak flow",
            "FEV1", "spirometry", "bronchodilator reversibility",
        ],
        "fillers": {
            "c": ["COPD", "severe asthma", "pulmonary fibrosis",
                  "chronic respiratory failure"],
        },
    },
    "renal_disease": {
        "title_patterns": [
            "AKI staging and outcomes in hospitalised {c} patients",
            "Contrast-induced nephropathy prevention in {c}",
            "Dialysis initiation timing in {c}",
            "Anaemia management in {c}",
            "Mineral and bone disorder in {c}",
        ],
        "concepts": [
            "acute kidney injury", "chronic kidney disease", "end-stage renal disease",
            "glomerular filtration rate", "serum creatinine", "urea",
            "proteinuria", "haematuria", "glomerulonephritis",
            "nephrotic syndrome", "renal replacement therapy",
            "haemodialysis", "peritoneal dialysis", "renal transplant",
            "RAAS blockade", "erythropoiesis-stimulating agents", "ferritin",
            "phosphate binders", "vitamin D supplementation",
            "contrast-induced nephropathy", "AKI staging",
        ],
        "fillers": {
            "c": ["chronic kidney disease", "acute kidney injury",
                  "renal failure", "diabetic nephropathy"],
        },
    },
    "drug_interactions": {
        "title_patterns": [
            "Cytochrome P450 mediated {c}: clinical implications",
            "QT prolongation risk with {c}: a pharmacovigilance study",
            "Pharmacokinetic interactions between {c} and common co-medications",
            "Clinically significant {c} in polypharmacy patients",
            "Drug-drug interactions in elderly patients prescribed {c}",
        ],
        "concepts": [
            "drug-drug interaction", "cytochrome P450", "CYP3A4 inhibition",
            "CYP2C19 polymorphism", "pharmacokinetic interaction",
            "pharmacodynamic interaction", "QT prolongation",
            "torsades de pointes", "warfarin interaction",
            "serotonin syndrome", "rhabdomyolysis", "statin interaction",
            "anticoagulant potentiation", "antibiotic interaction",
            "renal dosing adjustment", "therapeutic drug monitoring",
            "narrow therapeutic index", "polypharmacy", "medication reconciliation",
        ],
        "fillers": {
            "c": ["warfarin-antibiotic interaction", "statin-fibrate combination",
                  "antidepressant-MAOI interaction", "QT-prolonging drug combination"],
        },
    },
}

# ---------------------------------------------------------------------------
# Query test set — lay / colloquial vocabulary mapped to clinical concepts
# ---------------------------------------------------------------------------

QUERY_TEMPLATES: list[dict] = [
    # Acute coronary syndrome
    {"query": "heart attack treatment aspirin antiplatelet", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["heart attack", "blood thinner"]},
    {"query": "blocked artery stent procedure", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["blocked artery", "stent"]},
    {"query": "chest pain ECG diagnosis hospital", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["chest pain"]},
    {"query": "heart attack recovery rehabilitation exercise", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["heart attack", "recovery"]},
    {"query": "cholesterol heart disease prevention", "relevant_topics": ["acute_coronary_syndrome", "hypertension"], "lay_terms": ["cholesterol"]},
    {"query": "clot busting drug heart attack emergency", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["clot busting", "clot dissolving"]},
    {"query": "silent heart attack no symptoms diagnosis", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["silent heart attack"]},
    {"query": "widowmaker heart attack left artery", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["widowmaker"]},
    {"query": "mini heart attack enzyme blood test", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["mini heart attack"]},
    {"query": "heart attack survival rate hospital outcomes", "relevant_topics": ["acute_coronary_syndrome"], "lay_terms": ["heart attack"]},

    # Anticoagulation
    {"query": "blood thinner dose adjustment kidney", "relevant_topics": ["anticoagulation_therapy", "renal_disease"], "lay_terms": ["blood thinner"]},
    {"query": "blood clot leg swelling DVT", "relevant_topics": ["anticoagulation_therapy"], "lay_terms": ["blood clot", "leg clot"]},
    {"query": "warfarin INR monitoring diet food", "relevant_topics": ["anticoagulation_therapy"], "lay_terms": ["rat poison pill", "warfarin"]},
    {"query": "new blood thinner pill no monitoring needed", "relevant_topics": ["anticoagulation_therapy"], "lay_terms": ["new blood thinner"]},
    {"query": "blood clot lung shortness breath treatment", "relevant_topics": ["anticoagulation_therapy"], "lay_terms": ["blood clot lung"]},
    {"query": "blood thinners surgery stop before operation", "relevant_topics": ["anticoagulation_therapy"], "lay_terms": ["blood thinners"]},
    {"query": "clot lung hospital anticoagulant how long", "relevant_topics": ["anticoagulation_therapy"], "lay_terms": ["clot lung"]},
    {"query": "bleeding risk assessment anticoagulation score", "relevant_topics": ["anticoagulation_therapy"], "lay_terms": ["bleeding risk"]},

    # Heart failure
    {"query": "water pill swollen ankles fluid retention", "relevant_topics": ["heart_failure"], "lay_terms": ["water pill", "fluid pill"]},
    {"query": "weak heart pump ejection fraction low", "relevant_topics": ["heart_failure"], "lay_terms": ["weak heart", "weak pump"]},
    {"query": "breathless lying flat night heart swelling", "relevant_topics": ["heart_failure"], "lay_terms": ["breathless flat", "night breathing trouble"]},
    {"query": "heart failure new pill SGLT2 benefits", "relevant_topics": ["heart_failure", "diabetes_mellitus"], "lay_terms": ["heart failure pill"]},
    {"query": "ICD defibrillator implant heart failure sudden death", "relevant_topics": ["heart_failure"], "lay_terms": ["shock device", "defibrillator vest"]},
    {"query": "heart failure readmission prevention self-monitoring weight", "relevant_topics": ["heart_failure"], "lay_terms": ["heart failure readmission"]},
    {"query": "shortness breath heart failure lung fluid diuretic", "relevant_topics": ["heart_failure"], "lay_terms": ["drowning in fluid"]},

    # Diabetes
    {"query": "high blood sugar type 2 diabetes first treatment", "relevant_topics": ["diabetes_mellitus"], "lay_terms": ["high blood sugar", "sugar diabetes"]},
    {"query": "diabetes injection GLP weight loss pen", "relevant_topics": ["diabetes_mellitus"], "lay_terms": ["diabetes injection", "slimming jab"]},
    {"query": "low blood sugar hypoglycaemia dangerous driving", "relevant_topics": ["diabetes_mellitus"], "lay_terms": ["low blood sugar", "hypo"]},
    {"query": "diabetes eye check retina damage prevention", "relevant_topics": ["diabetes_mellitus"], "lay_terms": ["diabetes eye", "sugar eye"]},
    {"query": "diabetic foot ulcer infection amputation risk", "relevant_topics": ["diabetes_mellitus"], "lay_terms": ["diabetic foot", "sugar foot"]},
    {"query": "sugar tablet metformin side effects stomach", "relevant_topics": ["diabetes_mellitus"], "lay_terms": ["sugar tablet"]},
    {"query": "diabetes kidney damage protein urine creatinine", "relevant_topics": ["diabetes_mellitus", "renal_disease"], "lay_terms": ["sugar kidney"]},
    {"query": "HbA1c target elderly frail patient diabetic", "relevant_topics": ["diabetes_mellitus"], "lay_terms": ["sugar test three months"]},
    {"query": "diabetes technology CGM pump closed loop", "relevant_topics": ["diabetes_mellitus"], "lay_terms": ["sugar sensor", "insulin pump"]},

    # Hypertension
    {"query": "high blood pressure no symptoms silent killer", "relevant_topics": ["hypertension"], "lay_terms": ["high blood pressure", "silent killer"]},
    {"query": "blood pressure tablet side effects tiredness", "relevant_topics": ["hypertension"], "lay_terms": ["blood pressure tablet"]},
    {"query": "white coat blood pressure home monitoring", "relevant_topics": ["hypertension"], "lay_terms": ["white coat", "doctor anxiety BP"]},
    {"query": "resistant high blood pressure multiple tablets", "relevant_topics": ["hypertension"], "lay_terms": ["hard to control BP"]},
    {"query": "blood pressure stroke risk prevention treatment", "relevant_topics": ["hypertension", "stroke"], "lay_terms": ["blood pressure stroke"]},
    {"query": "high blood pressure pregnancy preeclampsia risk", "relevant_topics": ["hypertension"], "lay_terms": ["blood pressure pregnancy"]},
    {"query": "salt diet blood pressure reduce sodium", "relevant_topics": ["hypertension"], "lay_terms": ["salty food BP"]},

    # Stroke
    {"query": "brain attack clot busting treatment time window", "relevant_topics": ["stroke"], "lay_terms": ["brain attack", "brain clot"]},
    {"query": "mini stroke TIA warning sign prevention", "relevant_topics": ["stroke"], "lay_terms": ["mini stroke", "little stroke"]},
    {"query": "stroke recovery physiotherapy speech rehabilitation", "relevant_topics": ["stroke"], "lay_terms": ["stroke recovery"]},
    {"query": "brain bleed anticoagulation stop restart when", "relevant_topics": ["stroke", "anticoagulation_therapy"], "lay_terms": ["brain bleed"]},
    {"query": "stroke prevention AF heart rhythm blood thinner", "relevant_topics": ["stroke", "anticoagulation_therapy"], "lay_terms": ["stroke prevention AF"]},
    {"query": "FAST test stroke symptoms recognition face arm", "relevant_topics": ["stroke"], "lay_terms": ["FAST stroke symptoms", "face drooping"]},
    {"query": "stroke clot removal surgery catheter brain", "relevant_topics": ["stroke"], "lay_terms": ["stroke clot removal"]},

    # Sepsis
    {"query": "blood poisoning signs symptoms diagnosis hospital", "relevant_topics": ["sepsis"], "lay_terms": ["blood poisoning", "blood infection"]},
    {"query": "sepsis antibiotic one hour protocol bundle", "relevant_topics": ["sepsis"], "lay_terms": ["sepsis antibiotics fast"]},
    {"query": "ICU infection drip antibiotics IV treatment", "relevant_topics": ["sepsis"], "lay_terms": ["drip infection", "IV antibiotics"]},
    {"query": "sepsis shock low blood pressure ICU treatment", "relevant_topics": ["sepsis"], "lay_terms": ["septic shock", "blood pressure crash"]},
    {"query": "infection blood test CRP white cells high", "relevant_topics": ["sepsis"], "lay_terms": ["infection marker", "white cells"]},
    {"query": "lactic acid high blood test infection organ", "relevant_topics": ["sepsis"], "lay_terms": ["lactic acid", "lactate"]},

    # Respiratory
    {"query": "COPD inhaler puffer choice which one", "relevant_topics": ["respiratory_disease"], "lay_terms": ["puffer", "reliever inhaler"]},
    {"query": "asthma attack emergency hospital breathing", "relevant_topics": ["respiratory_disease"], "lay_terms": ["asthma attack"]},
    {"query": "breathless COPD oxygen home long-term", "relevant_topics": ["respiratory_disease"], "lay_terms": ["home oxygen", "breathless"]},
    {"query": "lung scarring fibrosis treatment prognosis", "relevant_topics": ["respiratory_disease"], "lay_terms": ["lung scarring", "stiff lungs"]},
    {"query": "steroid inhaler side effects osteoporosis", "relevant_topics": ["respiratory_disease"], "lay_terms": ["steroid inhaler", "puffer"]},
    {"query": "CPAP mask sleep apnea breathing night", "relevant_topics": ["respiratory_disease"], "lay_terms": ["CPAP machine", "sleep apnea mask"]},

    # Renal
    {"query": "kidney failure dialysis when to start", "relevant_topics": ["renal_disease"], "lay_terms": ["kidney failure", "kidney washout"]},
    {"query": "kidney disease creatinine rising eGFR low", "relevant_topics": ["renal_disease"], "lay_terms": ["kidney number", "kidney score"]},
    {"query": "contrast dye CT scan kidney damage prevention", "relevant_topics": ["renal_disease"], "lay_terms": ["contrast dye kidney"]},
    {"query": "kidney transplant rejection immunosuppression", "relevant_topics": ["renal_disease"], "lay_terms": ["kidney transplant"]},
    {"query": "protein urine kidney nephrotic syndrome foamy", "relevant_topics": ["renal_disease"], "lay_terms": ["protein urine", "foamy urine"]},
    {"query": "kidney anaemia erythropoietin injection iron", "relevant_topics": ["renal_disease"], "lay_terms": ["kidney anaemia", "tired kidney"]},

    # Drug interactions
    {"query": "blood thinner antibiotic interaction dangerous", "relevant_topics": ["drug_interactions", "anticoagulation_therapy"], "lay_terms": ["blood thinner antibiotic"]},
    {"query": "grapefruit juice medication dangerous interaction", "relevant_topics": ["drug_interactions"], "lay_terms": ["grapefruit medication"]},
    {"query": "statins muscle pain aching side effect", "relevant_topics": ["drug_interactions"], "lay_terms": ["cholesterol tablet muscle pain"]},
    {"query": "antidepressant other medication serotonin overload", "relevant_topics": ["drug_interactions"], "lay_terms": ["antidepressant interaction", "serotonin overload"]},
    {"query": "over counter painkiller blood thinner interaction", "relevant_topics": ["drug_interactions", "anticoagulation_therapy"], "lay_terms": ["painkiller blood thinner"]},
    {"query": "herbal supplement St Johns Wort drug interaction", "relevant_topics": ["drug_interactions"], "lay_terms": ["herbal supplement interaction"]},
    {"query": "heart tablet grapefruit avoid QT interval", "relevant_topics": ["drug_interactions"], "lay_terms": ["heart tablet grapefruit"]},
]


# ---------------------------------------------------------------------------
# Document generation helpers
# ---------------------------------------------------------------------------

ABSTRACT_TEMPLATES = [
    (
        "Background: {c1} is a major cause of morbidity and mortality in hospitalised patients. "
        "Methods: We retrospectively analysed {n} patients presenting with {c2} between "
        "{yr1} and {yr2}. Primary outcome was {primary_outcome}. "
        "Results: {c3} was administered in {pct}% of cases. "
        "Patients receiving {c4} had significantly lower {adverse_event} rates "
        "(OR {or_val}, 95% CI {ci}, p<0.001). "
        "Conclusion: Early initiation of {c5} is associated with improved outcomes in {c6}."
    ),
    (
        "Objective: To evaluate the efficacy of {c1} in patients with {c2}. "
        "Design: Prospective randomised controlled trial. Participants: {n} patients "
        "with confirmed {c3}. Intervention: {c4} versus standard care. "
        "Main outcomes: {primary_outcome} at {follow_up} follow-up. "
        "Results: {pct}% improvement in the {c5} arm (p<0.05). "
        "{c6} was the most frequently reported adverse event."
    ),
    (
        "Purpose: {c1} remains challenging to manage in the setting of {c2}. "
        "This review summarises current evidence for {c3}, focusing on {c4}. "
        "A systematic search of {n} studies was conducted. Key findings: {c5} "
        "demonstrated superiority over {c6} in patients with co-morbid {c2}. "
        "Recommendations are provided for clinical practice."
    ),
]

def _make_abstract(concepts: list[str]) -> str:
    tpl = random.choice(ABSTRACT_TEMPLATES)
    c = random.sample(concepts, min(6, len(concepts)))
    while len(c) < 6:
        c.append(random.choice(concepts))
    return tpl.format(
        c1=c[0], c2=c[1], c3=c[2], c4=c[3], c5=c[4], c6=c[5],
        n=random.randint(120, 4800),
        yr1=random.randint(2015, 2019),
        yr2=random.randint(2020, 2024),
        pct=round(random.uniform(42, 87), 1),
        primary_outcome=random.choice([
            "30-day mortality", "major adverse cardiovascular events",
            "hospital length of stay", "90-day readmission", "composite endpoint"
        ]),
        adverse_event=random.choice([
            "bleeding", "acute kidney injury", "hypotension",
            "hypoglycaemia", "drug-induced hepatotoxicity"
        ]),
        or_val=round(random.uniform(0.3, 2.8), 2),
        ci=f"{round(random.uniform(0.2,0.9),2)}-{round(random.uniform(1.0,3.5),2)}",
        follow_up=random.choice(["6 months", "1 year", "2 years", "90 days"]),
    )


def generate_clinical_corpus(n: int = 2000) -> list[dict]:
    topic_names = list(TOPIC_VOCAB.keys())
    n_per_topic = n // len(topic_names)
    rows = []
    doc_id = 0

    for topic_name, topic_data in TOPIC_VOCAB.items():
        for _ in range(n_per_topic):
            # Title
            tpl = random.choice(topic_data["title_patterns"])
            filler_key = random.choice(list(topic_data["fillers"].keys()))
            title = tpl.format(**{filler_key: random.choice(topic_data["fillers"][filler_key])})

            # Abstract  (3-4 sentences using clinical vocabulary)
            abstract = _make_abstract(topic_data["concepts"])

            # Keywords from topic vocabulary
            keywords = ", ".join(random.sample(topic_data["concepts"], min(6, len(topic_data["concepts"]))))

            full_text = f"{title}. {abstract} Keywords: {keywords}."

            rows.append({
                "doc_id":    f"DOC-{doc_id:05d}",
                "topic":      topic_name,
                "title":      title,
                "abstract":   abstract,
                "keywords":   keywords,
                "full_text":  full_text,
            })
            doc_id += 1

    # Pad any remainder
    while len(rows) < n:
        extra_topic = random.choice(topic_names)
        rows.append(rows[random.randint(0, len(rows)-1)].copy())
        rows[-1]["doc_id"] = f"DOC-{doc_id:05d}"
        doc_id += 1

    random.shuffle(rows)
    return rows


def generate_query_testset() -> list[dict]:
    """
    Return list of test queries, each with:
      - query_id
      - query_text (lay language)
      - relevant_topics (list)
      - lay_terms (list of informal terms used — absent from clinical documents)
      - relevant_doc_topics (CSV of expected document topics)
    """
    rows = []
    for i, qt in enumerate(QUERY_TEMPLATES):
        rows.append({
            "query_id":           f"Q-{i:03d}",
            "query_text":         qt["query"],
            "lay_terms":          "; ".join(qt["lay_terms"]),
            "relevant_topics":    "; ".join(qt["relevant_topics"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Clinical Corpus ----
    corpus = generate_clinical_corpus(n=2000)
    corpus_path = out_dir / "clinical_corpus.csv"
    with open(corpus_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "topic", "title", "abstract", "keywords", "full_text"])
        writer.writeheader()
        writer.writerows(corpus)
    print(f"Saved {len(corpus)} documents -> {corpus_path}")

    # Topic distribution
    from collections import Counter
    topic_counts = Counter(r["topic"] for r in corpus)
    print("Topic distribution:")
    for t, c in sorted(topic_counts.items()):
        print(f"  {t:<35} {c:>5} docs")

    # ---- Query Test Set ----
    queries = generate_query_testset()
    query_path = out_dir / "query_testset.csv"
    with open(query_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query_text", "lay_terms", "relevant_topics"])
        writer.writeheader()
        writer.writerows(queries)
    print(f"\nSaved {len(queries)} queries -> {query_path}")

    # ---- Vocabulary overlap analysis ----
    corpus_words: set[str] = set()
    for r in corpus:
        for w in r["full_text"].lower().split():
            corpus_words.add(w.strip(".,;:()[]"))

    query_words: set[str] = set()
    for q in queries:
        for w in q["query_text"].lower().split():
            query_words.add(w.strip(".,;:()[]"))

    lay_words: set[str] = set()
    for q in queries:
        for term in q["lay_terms"].split("; "):
            for w in term.lower().split():
                lay_words.add(w.strip(".,;:()[]"))

    overlap_query = corpus_words & query_words
    overlap_lay   = corpus_words & lay_words

    print("\nVocabulary overlap analysis:")
    print(f"  Clinical corpus vocab size    : {len(corpus_words):>6}")
    print(f"  Query vocab (all words)       : {len(query_words):>6}")
    print(f"  Query lay terms vocab         : {len(lay_words):>6}")
    print(f"  Overlap (query vs corpus)     : {len(overlap_query):>6}  ({100*len(overlap_query)/len(query_words):.1f}%)")
    print(f"  Overlap (lay terms vs corpus) : {len(overlap_lay):>6}  ({100*len(overlap_lay)/len(lay_words):.1f}%)")
    print()
    print("  Key vocabulary mismatch examples:")
    mismatches = [
        ("heart attack",        "myocardial infarction / STEMI"),
        ("blood thinner",       "anticoagulant / warfarin / heparin"),
        ("water pill",          "furosemide / loop diuretic / diuretic"),
        ("sugar diabetes",      "type 2 diabetes mellitus / hyperglycaemia"),
        ("brain bleed",         "intracranial haemorrhage"),
        ("blood poisoning",     "sepsis / bacteraemia"),
        ("clot busting drug",   "thrombolysis / alteplase / fibrinolytic"),
        ("weak heart pump",     "reduced ejection fraction / systolic dysfunction"),
    ]
    for lay, clinical in mismatches:
        print(f"  '{lay}' -> '{clinical}'")
 