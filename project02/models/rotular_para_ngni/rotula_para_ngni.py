"""
Rotula os splits (train/test) com as tags de nó usadas pelo NGramsNodeIdentifier.

Uso (a partir de project02/):
    python label_splits.py
    python label_splits.py --splits-dir data/processed/splits

Entrada : train_cases.csv, test_cases.csv, split_manifest.csv
Saída   : train_cases_labeled.csv, test_cases_labeled.csv
          (mesmas colunas + `case_text_tagged`; `case_text` original é mantido)

Sem dependências externas. Cada split é rotulado de forma independente com o
MESMO léxico fixo -> nenhuma estatística do teste entra no treino.
Exemplo:
    "The child has pain on the right knee and should not run."
 -> "The TAG-PATIENT has TAG-SYMPTOM on the TAG-ANATOMY and should not run."
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# mesmos tipos de diplo_graph2/node_identifiers/ngrams_node_identify.py
NODE_TYPES = ['PATIENT', 'DISEASE', 'SYMPTOM', 'DRUG', 'EXAME', 'ANATOMY', 'EFFECT', 'RESULT']

# --------------------------------------------------------------------------
# LÉXICOS  (ordem de definição = prioridade quando o mesmo termo está em 2 tags)
# --------------------------------------------------------------------------
LEX = {
    'SYMPTOM': """
pain; abdominal pain; chest pain; chest tightness; tightness; fever; headache; headaches; nausea; vomiting;
emesis; diarrhea; diarrhoea; constipation; fatigue; malaise; dyspnea; shortness of breath; respiratory distress;
cough; rash; edema; oedema; swelling; weakness; paresis; dizziness; dysphagia; odynophagia; weight loss;
weight gain; bleeding; rectal bleeding; melena; hematochezia; abdominal distension; abdominal distention;
abdominal fullness; fullness; pruritus; jaundice; chills; rigor; myalgia; drowsiness; syncope; confusion;
seizure; seizures; tachycardia; bradycardia; hypotension; hypoxemia; hypoxia; desaturation; dysaesthesia;
hypoesthesia; anorexia; heartburn; erythema; ulcer; ulcers; petechiae; tenderness; murmur; fasciculations;
cramps; incontinence; urinary incontinence; dysphonia; nystagmus; hemoglobinuria; proteinuria; ascites;
pleural effusion; effusion; pericardial effusion; pallor; alopecia; dyspnoea; bloody diarrhea; poor vision;
sweating; night sweats; breathlessness; hematuria; epistaxis; menorrhagia; dysmenorrhea; alopecia; vesicles;
papules; pustules; blisters; erythematous patches; leukocytosis; thrombocytopenia; thrombocytosis; hypokalemia;
acidosis; uremia; hyperthyroidism; hypothyroidism; subcutaneous emphysema; infiltrates; air leak
""",
    'DISEASE': """
cancer; adenocarcinoma; carcinoma; tumor; tumour; neoplasm; sarcoma; undifferentiated sarcoma; angiosarcoma;
schwannoma; ancient schwannoma; ganglioglioma; meningioma; lymphoma; myxoma; leiomyoma; fibroid; fibroids; cyst;
abscess; infection; pneumonia; aspiration pneumonia; sepsis; septicemia; septic shock; endocarditis;
infective endocarditis; tuberculosis; hypertension; malignant hypertension; diabetes mellitus; diabetes; asthma;
lupus; systemic lupus erythematosus; SLE; lupus nephritis; COVID-19; influenza; thrombosis; deep vein thrombosis;
pulmonary embolism; embolism; embolus; aneurysm; subarachnoid hemorrhage; intraventricular hemorrhage;
hemorrhage; stenosis; mitral stenosis; fistula; tracheoesophageal fistula; esophagobronchial fistula;
esophagobronchial fistulae; fistulae; cirrhosis; anemia; hemolytic anemia; autoimmune hemolytic anemia;
cold agglutinin syndrome; Graves' disease; Graves disease; ulcerative colitis; colitis; gastritis; pemphigoid;
bullous pemphigoid; Sjogren's syndrome; epilepsy; hydrocephalus; appendicitis; endometriosis; hernia;
parastomal hernia; obstruction; gastric outlet obstruction; ischemia; infarction; myocardial infarction;
thrombotic thrombocytopenic purpura; TTP; preeclampsia; necrotizing enterocolitis; NEC; intrauterine fetal demise;
shoulder dystocia; pneumothorax; pulmonary abscess; atrial fibrillation; mitral regurgitation;
tricuspid regurgitation; regurgitation; cardiomyopathy; postpartum cardiomyopathy; nephritis; vasculitis;
Behcet's disease; osteoporosis; depression; cellulitis; herpes zoster; esophagitis; diverticulum; stricture;
peptic stricture; gastroesophageal reflux disease; GERD; coronary artery spasm; mass; lesion; nodule; nodules;
polyps; adenoma; oxyntic gland adenoma; vegetation; vegetations; pannus; aortic aneurysm; ventricular septal defect;
VSD; patent ductus arteriosus; truncus arteriosus; interrupted aortic arch; bronchomalacia; fibrous dysplasia;
fibrous dysplasia; keratocystic odontogenic tumor; steroid cell tumor; gastric duplication cyst; polycystic ovaries;
infertility; blindness; fracture; fractures; quadriplegia; pressure injury; septic emboli; hematoma; leak;
H. pylori; Helicobacter pylori; Pseudomonas aeruginosa; Staphylococcus aureus; Klebsiella pneumoniae; adenovirus;
iron deficiency; iron deficiency anemia; hypoproteinemia; enteropathy; heterotopic tissue; ectopic endometrial tissue
""",
    'DRUG': """
prednisone; prednisolone; methylprednisolone; dexamethasone; hydroxychloroquine; cyclophosphamide;
mycophenolate mofetil; azathioprine; rituximab; infliximab; vedolizumab; mesalamine; 5-ASA; carbimazole;
levothyroxine; heparin; low molecular weight heparin; LMWH; warfarin; aspirin; clopidogrel; vancomycin;
clindamycin; piperacillin-tazobactam; ceftaroline; gentamicin; linezolid; meropenem; levofloxacin; moxifloxacin;
ciprofloxacin; acyclovir; amoxicillin; clarithromycin; omeprazole; pantoprazole; diltiazem; atorvastatin;
dupilumab; alectinib; brigatinib; pembrolizumab; carboplatin; pemetrexed; ketamine; dexmedetomidine; lidocaine;
ropivacaine; sevoflurane; epinephrine; noradrenaline; adrenaline; dopamine; dobutamine; isoniazid; rifampin;
ethambutol; pyrazinamide; IVIG; intravenous immunoglobulin; immunoglobulin; insulin; acarbose; oseltamivir;
arbidol; cyclosporine; UDCA; nitroglycerin; misoprostol; Pitocin; triheptanoin; diphenhydramine; triamcinolone;
thymosin; irbesartan; hydrochlorothiazide; calcium channel blocker; oxcarbazepine; valproic acid; cefotaxime;
cefatoxime; co-amoxyclav; ceftazidime; loxoprofen; pethidine; antibiotic; antibiotics; antibiotic therapy;
chemotherapy; corticosteroids; corticosteroid; cortisone; steroid; steroids; steroid therapy; antihypertensives;
inotropes; inotropic drugs; proton pump inhibitor; proton pump inhibitors; vitamin D; calcium; iron;
EMLA cream; EMLA; Enfaport; MCT oil; dextrose; Intralipid; TrophAmine; ampicillin-sulbactam; vitamin K antagonist;
DOAC; DOACs; antiplatelet therapy; antithrombotics; anticoagulation; anticoagulant therapy; Lianhuaqingwen;
gonadotropin; oral contraceptives; ACTH; hematinics; ursodeoxycholic acid
""",
    'EXAME': """
computed tomography; CT; CT scan; CTA; CT angiography; CTv; MRv; MRI; magnetic resonance imaging; MRA;
magnetic resonance angiography; ultrasound; ultrasonography; transvaginal ultrasonography; echocardiography;
echocardiogram; transesophageal echocardiogram; transthoracic echocardiogram; TEE; TTE; X-ray; X-rays; radiograph;
radiographs; chest X-ray; abdominal radiographs; PET-CT; PET CT; EEG; video-EEG; ECG; EKG; electrocardiogram;
EGD; esophagogastroduodenoscopy; oesophagogastroduodenoscopy; endoscopy; colonoscopy; enteroscopy; push enteroscopy;
bronchoscopy; biopsy; biopsies; histology; histopathology; histological examination; histopathological examination;
cytology; immunohistochemistry; immunohistochemical study; culture; cultures; blood culture; blood cultures;
laparoscopy; angiography; coronary angiography; CAG; lumbar puncture; blood count; complete blood count;
laboratory tests; laboratory examinations; laboratory investigations; blood tests; serology; urinalysis;
barium swallow; barium enema; UGI study; UGI contrast study; EUS-FNA; FNA; Doppler; flow cytometry; PCR;
rRT-PCR; sequencing; next-generation sequencing; NGS; direct immunofluorescence; DIF; Coombs test;
Donath-Landsteiner test; Schirmer test; glucose tolerance test; GTT; electromyography; nerve conduction;
electroneurography; OCT; biometry; fundus photographs; fundus examination; ERCP; video capsule endoscopy;
videoangiography; urine drug screen; intravascular ultrasound; arteriography; hysterosalpingography;
cycloplegic refraction; slit-lamp examination; nasopharyngeal swabs; swabs; ferritin; hemoglobin; hemoglobin A1c;
troponin; troponin I; CK-MB; creatinine; CRP; C-reactive protein; ESR; erythrocyte sedimentation rate; LDH;
lactate dehydrogenase; platelet count; white blood cell count; WBC; D-dimer; INR; PTT; PT; haptoglobin;
reticulocyte count; transferrin saturation; bilirubin; albumin; ALT; AST; ALP; tumor markers; CEA; CA 19-9;
CA19-9; CA12-5; AFP; HE4; ANA; antinuclear antibody; anti-dsDNA; ADAMTS13 activity; urine protein/creatinine;
physical examination; neurological examination; cardiac auscultation; auscultation; serum ferritin; blood gas analysis;
arterial blood gas analysis; ELISA; Sanger; Ishihara; Farnsworth Munsell; Revised Cardiac Risk Index; pulse oximetry
""",
    'EFFECT': """
complication; complications; recurrence; relapse; relapses; remission; resolution; improvement; improved;
improving; died; death; expired; succumbed; hemodynamic instability; anastomotic leak; migration;
stent migration; adverse event; hypersensitivity; drug eruption; seizure-free; symptom-free; asymptomatic;
recovery; recovered; cardiac arrest; respiratory failure; multiple organ dysfunction syndrome; MODS;
clinical improvement; failure to thrive; hemolysis; exacerbation; exacerbations; deterioration; deteriorate;
no recurrence; uneventful; spontaneous circulation; return of spontaneous circulation; ROSC; extubated;
discharged; hospice care; stable; stabilization
""",
    'RESULT': """
normal; abnormal; abnormalities; unremarkable; negative; positive; elevated; elevation; decreased; reduced;
reduction; increased; within normal limits; within normal range; nondiagnostic; nonspecific; benign;
malignant; consistent with; suggestive of; compatible with; diagnostic; non-contributory; noncontributory;
borderline; markedly elevated; mildly elevated; slightly elevated; low; high; no evidence of malignancy
""",
}

# ANATOMY: substantivos + modificadores opcionais ("right", "upper", ...) -> "right knee" vira 1 nó
ANATOMY_NOUNS = """
abdomen; abdominal wall; stomach; pancreas; pancreatic tail; pancreatic mass; spleen; liver; kidney; kidneys;
lung; lungs; heart; atrium; ventricle; aorta; aortic arch; mitral valve; tricuspid valve; aortic valve;
truncal valve; valve; esophagus; oesophagus; duodenum; jejunum; ileum; colon; rectum; appendix;
cecal appendix; bladder; ovary; ovaries; uterus; fallopian tube; mandible; skull; brain; temporal lobe;
bronchus; trachea; thorax; chest; chest wall; flank; quadrant; pelvis; mesentery; omentum; bile duct;
gallbladder; artery; arteries; vein; veins; pulmonary artery; femoral vein; iliac vein; coronary artery;
lymph node; lymph nodes; skin; hand; arm; forearm; finger; fingers; fingertips; thumb; knee; foot; leg;
lip; tongue; eye; eyes; retina; fovea; macula; sternum; rib; ribs; neck; head; nerve; spinal cord; cord;
cervical spine; bone; cecum; thyroid; adrenal gland; peritoneum; pleura; diaphragm; septum; sinus;
coronary sinus; jaw; scalp; dura; sigmoid colon; cardia; antrum; fundus; bowel; intestine; mesocolon;
celiac axis; coeliac axis; celiac vessels; coeliac vessels; salivary gland; gastric wall; gastrojejunostomy;
cervical vertebra; vertebra; upper extremity; extremities; lower extremities; extremity; limbs; limb;
joint; joints; metacarpophalangeal joints; wrist; elbow; shoulder; axilla; groin; scrotum; buttocks;
sacral region; sacrum; trunk; palm; palms; soles; face; mouth; oropharynx; hypopharynx; pharynx; nose;
airway; lobe; hilum; mediastinum; pleural cavity; thoracic cavity; peritoneal cavity; abdominal cavity;
efferent limb; afferent limb; ductus; stoma; parietal bone; foveal pit; GCL; INL; ONL; kidney; mucosa;
gastric mucosa; endometrium; cervix; vagina; vulva; inguinal region; parotid; hypochondrium; epigastrium
"""
ANATOMY_MODIFIERS = """
right; left; bilateral; upper; lower; proximal; distal; posterior; anterior; superior; inferior; lateral;
medial; central; ascending; descending; small; large; middle; mid; transverse; cervical; thoracic; lumbar;
lateral; frontal; temporal; parietal; common; external; internal; superficial; deep; first; second; third;
fourth; subcutaneous; gastric; pulmonary; cardiac; hepatic; renal; pancreatic; esophageal; intestinal;
abdominal; pelvic; coronary; ventricular; atrial
"""

# paciente: "57-year-old man", "6-day-old twin male infant", "7-month-old ... case" etc.
AGE_PATIENT = (r"\b(?:an?\s+)?(?:\d+(?:\.\d+)?|[a-z]+)[- ](?:year|month|week|day)s?[- ]old"
               r"(?:\s+[A-Za-z-]+){0,3}?\s+"
               r"(?:(?:man|woman|male|female|boy|girl|infant|newborn|baby|child|patient|lady|nulligravida)\s+){0,2}"
               r"(?:man|woman|male|female|boy|girl|infant|newborn|baby|child|patient|lady|nulligravida)\b")
PATIENT_WORDS = ["patient", "patients", "man", "woman", "boy", "girl", "infant", "newborn", "baby", "child"]

PATIENT_WORDS = []


def _alts(terms, plural=True):
    terms = sorted({t.strip() for t in terms.replace("\n", " ").split(";") if t.strip()},
                   key=len, reverse=True)
    out = []
    for t in terms:
        e = re.escape(t).replace(r"\ ", r"\s+")
        if plural and t[-1].isalpha():
            if t[-1] == "y" and len(t) > 3:
                e = e[:-1] + r"(?:y|ies)"
            else:
                e += r"(?:s|es)?"
        out.append(e)
    return out


def build_patterns():
    pats = []  # (compiled_regex, tag)  -- ordem = prioridade em empate de tamanho
    #pats.append((re.compile(AGE_PATIENT, re.I), "PATIENT"))
    #pats.append((re.compile(r"\b(?:" + "|".join(PATIENT_WORDS) + r")\b", re.I), "PATIENT"))
    for tag in ['SYMPTOM', 'DISEASE', 'DRUG', 'EXAME']:
        pats.append((re.compile(r"(?<![\w-])(?:" + "|".join(_alts(LEX[tag])) + r")(?![\w-])", re.I), tag))
    mods = "|".join(_alts(ANATOMY_MODIFIERS, plural=False))
    nouns = "|".join(_alts(ANATOMY_NOUNS))
    pats.append((re.compile(r"(?<![\w-])(?:(?:" + mods + r")\s+){0,3}(?:" + nouns + r")(?![\w-])", re.I), "ANATOMY"))
    for tag in ['EFFECT', 'RESULT']:
        pats.append((re.compile(r"(?<![\w-])(?:" + "|".join(_alts(LEX[tag])) + r")(?![\w-])", re.I), tag))
    return pats


PATTERNS = build_patterns()


def tag_text(text: str) -> tuple[str, Counter]:
    """Rotula um texto. Maior span ganha; empate -> início mais cedo, depois prioridade da tag."""
    cands = []
    for prio, (rx, tag) in enumerate(PATTERNS):
        for m in rx.finditer(text):
            s, e = m.span()
            # modificador sozinho nunca vira nó (ex.: "left" sem substantivo já é impossível pelo regex)
            cands.append((-(e - s), s, prio, e, tag))
    cands.sort()
    taken = [False] * (len(text) + 1)
    chosen = []
    for neg_len, s, prio, e, tag in cands:
        if any(taken[s:e]):
            continue
        for i in range(s, e):
            taken[i] = True
        chosen.append((s, e, tag))
    chosen.sort()
    out, last, counts = [], 0, Counter()
    for s, e, tag in chosen:
        out.append(text[last:s])
        out.append(f"TAG-{tag}")
        counts[tag] += 1
        last = e
    out.append(text[last:])
    return "".join(out), counts


def label_file(src: Path, dst: Path) -> Counter:
    csv.field_size_limit(sys.maxsize)
    total = Counter()
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or []) + ["case_text_tagged"]
    for r in rows:
        r["case_text_tagged"], c = tag_text(r["case_text"])
        total.update(c)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{src.name}: {len(rows)} casos -> {dst.name} | " +
          ", ".join(f"{t}={total[t]}" for t in NODE_TYPES))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="data/processed/splits")
    d = Path(ap.parse_args().splits_dir)

    # sanidade: nenhum artigo pode estar em treino e teste ao mesmo tempo
    with open(d / "split_manifest.csv", newline="", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    tr = {r["article_id"] for r in man if r["split"] == "train"}
    te = {r["article_id"] for r in man if r["split"] == "test"}
    assert not (tr & te), f"vazamento treino/teste: {tr & te}"

    for name in ("train", "test"):
        label_file(d / f"{name}_cases.csv", d / f"{name}_cases_labeled.csv")


if __name__ == "__main__":
    main()