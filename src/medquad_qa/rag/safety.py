"""Rule-based refusal of personalized medical advice (all four experiment modes; D-023).

These rules are a scope policy, not a diagnosis: they decide whether a question asks for an individual
clinical decision (personal dosing, self-diagnosis, judging one's own results, choosing/starting/stopping a
treatment for a specific person, emergencies) instead of general medical information. General questions
about a condition, test, drug or dose range are NOT refused, even when phrased with "we" or "I".

Design (safety-v2, after evaluator finding F-002):
- Emergencies need an *event* or *intent* (someone took/swallowed something, stopped breathing, chest pain with
  arm/jaw symptoms, self-harm intent). The bare words "overdose" or "suicide" do not trigger.
- Personal advice needs a *specific person* (I/me/my, or my/our <relative>) together with an advice cue
  (dose, safety, choice of treatment, diagnosis, judging a value, what to do). Generic "we"/"our" alone is not
  a specific person.
- Regular expressions are incomplete by construction: some personal requests will be missed and some general
  questions refused. The residual rates are measured by the evaluator (Track C), not here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

_REL_NOUN = (
    r"(?:\d+[- ]?(?:year|yr|month|week|day)s?[- ]?old|child|kid|kids|son|daughter|baby|infant|toddler|newborn|"
    r"wife|husband|partner|spouse|mother|mom|mum|father|dad|parent|grandmother|grandfather|grandma|grandpa|"
    r"brother|sister|uncle|aunt|cousin|niece|nephew|grandson|granddaughter|grandchild|friend|boyfriend|girlfriend|"
    r"roommate|patient|dog|cat)"
)
_REL = rf"(?:my|our) (?:\w+ )?{_REL_NOUN}"  # "my 3-year-old", "our baby", "my little sister"
_SELF = r"(?:i|i'm|im|i've|ive|i'd|i'll|me|my|myself)"
_PERSON = rf"(?:{_REL}|{_SELF})"  # a specific person
_SUBJ = rf"(?:i|{_REL}|he|she|they)"  # grammatical subject of a personal yes/no question
_CLAUSE = r"(?:^|[.;,!?:]\s*|\b(?:and|but|so|then|or)\s+)"  # start of a clause

_DRUG_ADMIN = r"(?:take|taking|took|inject|injecting|give|giving|use|using|drink|drinking|have|apply|double|skip)"
_TREATMENT_VERBS = (
    r"(?:take|stop|start|quit|skip|double|increase|decrease|reduce|lower|raise|switch|change|combine|mix|use|try|"
    r"drink|eat|inject|give|continue|keep taking|come off|wean)"
)
_DOSE_CUE = (
    r"(?:dose|doses|dosage|dosing|how much|how many (?:mg|milligrams|ml|units|tablets|pills|capsules|doses|puffs))"
)
_MEASURE = (
    r"(?:blood pressure|bp|blood sugar|sugar|glucose|a1c|hba1c|cholesterol|ldl|hdl|heart rate|pulse|temperature|"
    r"fever|oxygen|saturation|potassium|sodium|psa|tsh|creatinine|bmi|weight|levels?|results?|test results?|"
    r"lab results?|blood test|scan|x-?ray|mri|ct|biopsy|ecg|ekg)"
)
_STATEMENTS = (
    r"\b(?:i'm|im|i am) (?:\d+ (?:weeks|months) )?(?:pregnant|breastfeeding|nursing|diabetic|asthmatic|epileptic|"
    r"on \w+|taking \w+|allergic)",
    r"\b(?:i|i've|ive) (?:have|had|got|get|feel|am having|'m having|been having)\b"
    r"(?! (?:a |some |another |one more )?(?:question|questions|query|doubt)\b)",
    r"\bi've (?:been|got)\b",
    rf"\b{_REL} (?:has|had|is|was|got|keeps|feels|seems)\b",
    rf"\bmy (?:\w+ )?{_MEASURE} (?:is|are|was|were|came|reads?|shows?)\b",
    r"\bmy \w+ (?:hurts?|aches?|is swollen|is bleeding|itches)\b",
)
_HEALTH_STATEMENT = "(?:" + "|".join(_STATEMENTS) + ")"
_THIRD_PARTY_HARM = r"(?:hurting|harming|cutting|injuring) (?:themselves|himself|herself|themself)"
_SUSPICION = (
    rf"\bi (?:think|suspect|worry|fear) (?:i|i'm|{_REL}|he|she) (?:\w+ )?(?:have|has|had|is|am|might|may|could|got)\b"
)

# (rule_id, pattern) — any match refuses. Applied to a normalised, lowercased question.
RULES: tuple[tuple[str, str], ...] = (
    # ---- self-harm ideation: no event needed (F-007); crisis message
    (
        "emergency",
        r"\b(?:end(?:ing)? (?:it all|my (?:own )?life|things)|(?:take|took|taking|takes) my (?:own )?life|"
        r"(?:i|i'm|im|i am) (?:\w+ ){0,3}want(?:ing)? to die|"
        r"wish i (?:was|were|could be) dead|no (?:reason|point) (?:to|in) liv(?:e|ing)|better off dead|"
        r"(?:hurt|hurting|harm|harming|cut|cutting|kill|killing|killed) myself|"
        r"(?:i|i'm|im|i am|i've been|i have been|i was|i keep|i've|been) (?:\w+ ){0,2}thinking (?:about|of) "
        r"(?:suicide|killing myself|dying|death)|"
        r"don'?t want to (?:live|be alive|wake up))\b",
    ),
    # ---- a specific person the asker knows has suicidal thoughts: crisis message (D-041)
    (
        "emergency",
        rf"\b{_REL}\b[^?.!]{{0,40}}?\b(?:is|are|was|has been|have been|keeps|kept|started|been) (?:\w+ )?"
        r"(?:thinking|talking) (?:about|of) (?:suicide|killing (?:themselves|himself|herself)|ending (?:their|his|her) "
        r"(?:own )?life|dying)",
    ),
    (
        "emergency",
        rf"\b{_REL}\b[^?.!]{{0,40}}?\b(?:is|are|was|seems|sounds|feels|has been|have been|keeps|became) "
        rf"(?:\w+ )?(?:suicidal|{_THIRD_PARTY_HARM})",
    ),
    (
        "emergency",
        rf"\b{_REL}\b[^?.!]{{0,40}}?\b(?:wants|wanted|keeps saying (?:he|she|they) wants?|says (?:he|she|they) wants?) "
        r"to (?:die|end (?:it|it all|their life|his life|her life))",
    ),
    # ---- the asker seeking help for a suicidal / self-harming person: crisis resources (D-041)
    (
        "emergency",
        r"\b(?:what (?:should|can|do|must) (?:i|we) do|how (?:do|can|should) (?:i|we) "
        r"(?:help|support|talk to|deal with|handle|approach|save)|what (?:should|can) (?:i|we) say)\b[^?]*\b(?:suicid\w*|want(?:s|ing)? to die|"
        rf"{_THIRD_PARTY_HARM}|kill(?:ing)? (?:themselves|himself|herself)|end(?:ing)? (?:their|his|her) (?:own )?life|"
        r"self-?harm\w*)",
    ),
    # ---- emergencies: events or intent, never bare keywords
    (
        "emergency",
        rf"\b{_PERSON}\b[^?.!]{{0,60}}?\b(?:took|swallowed|ate|drank|injected|overdosed|has overdosed|"
        r"just took|accidentally took)\b[^?.!]{0,60}(?:\d+|pills?|tablets?|capsules?|medicine|"
        r"medication|bottle|bleach|poison|detergent|battery|batteries|too much|too many|whole)",
    ),
    (
        "emergency",
        rf"\b(?:{_SELF}|{_REL}|he|she) (?:\w+ ){{0,2}}(?:overdosed|took an overdose|took too (?:much|many))\b",
    ),
    (
        "emergency",
        r"\b(?:stopped breathing|not breathing|isn't breathing|is not breathing|can'?t breathe|"
        r"cannot breathe|can'?t breath|unresponsive|won'?t wake up|passed out|is unconscious|"
        r"turning blue|choking right now)\b",
    ),
    ("emergency", r"\b(?:having|having a|i think i'?m having a) (?:heart attack|stroke|seizure)\b"),
    (
        "emergency",
        r"\b(?:my|i have) (?:chest (?:hurts|pain|is tight|feels tight))\b[^?]{0,60}\b(?:arm|jaw|sweat|"
        r"breath|numb)",
    ),
    ("emergency", r"\bchest pain (?:right )?now\b"),
    (
        "emergency",
        r"\b(?:kill|hurt|harm|cut) (?:myself|me)\b|\b(?:i'm|im|i am|i feel|feeling|i've been|been) (?:\w+ )?suicidal\b|"
        r"\b(?:i|i'm|im) (?:going|planning|want) to (?:overdose|end it)\b",
    ),
    # ---- dosing for a specific person
    ("personal_dosing", rf"^(?=.*\b{_PERSON}\b)(?=.*\b(?:dose|doses|dosage|dosing|mg|milligrams|units of)\b)"),
    ("personal_dosing", rf"^(?=.*\b{_PERSON}\b)(?=.*\b{_DOSE_CUE}\b[^?]*\b(?:take|inject|give|use|double|apply)\b)"),
    (
        "personal_dosing",
        r"\b(?:dose|dosage|how much|how many)\b[^?]{0,60}\bfor (?:a|an|my|our) "
        r"(?:\d+(?:\.\d+)? ?(?:kg|kilo|kilos|lb|lbs|pounds?)|\d+[- ]?(?:year|month|week)s?[- ]?old)",
    ),
    # ---- starting/stopping/choosing a treatment for a specific person
    (
        "start_stop_treatment",
        rf"{_CLAUSE}(?:should|can|could|may|must|do|is it ok(?:ay)? (?:for|if)) "
        rf"{_SUBJ} (?:\w+ ){{0,2}}{_TREATMENT_VERBS}\b",
    ),
    (
        "start_stop_treatment",
        rf"\b(?:what|which) (?:\w+ )?(?:medication|medicine|drug|antibiotic|pill|tablet|"
        rf"treatment|painkiller|cream|supplement)s? (?:should|can|do|must) {_SUBJ} "
        rf"(?:{_DRUG_ADMIN}|get|try)\b",
    ),
    (
        "start_stop_treatment",
        rf"\b(?:what|which) (?:\w+ ){{0,2}}(?:is|are|would be|works?) (?:the )?(?:best|better|"
        rf"right|good|safest|safe) (?:\w+ )?for (?:me|my|{_REL})\b",
    ),
    (
        "start_stop_treatment",
        rf"{_CLAUSE}should {_SUBJ} (?:\w+ ){{0,2}}(?:go to|see|call|visit|be seen|take "
        r"(?:him|her|them) to) (?:the |a |an )?(?:er|a&e|emergency|hospital|doctor|gp|"
        r"urgent care|911|999|112|ambulance)",
    ),
    # ---- safety for a specific person
    (
        "personal_safety",
        rf"\bis (?:it|this|that|\w+(?: \w+)?) (?:safe|ok|okay|dangerous|harmful|bad|risky) "
        rf"(?:for|to give|to take|if) (?:me|my|{_REL}|him|her|i)\b",
    ),
    (
        "personal_safety",
        rf"\bis (?:it|this|that|\w+(?: \w+)?) (?:safe|ok|okay|dangerous|harmful|bad|risky)\b"
        rf"[^?]*\b(?:while|since|when|because|as|if) (?:i|i'm|im|{_REL})\b",
    ),
    ("personal_safety", rf"{_HEALTH_STATEMENT}[^?]*\b(?:safe|ok|okay|dangerous|harmful)\b"),
    (
        "personal_safety",
        r"\bis (?:it|this|that) (?:\w+ )?(?:safe|ok|okay|fine|alright|all right) to (?:give|take|use|share|mix|"
        rf"combine|put)\b[^?]*\b(?:my|me|{_REL}|him|her)\b",
    ),
    ("personal_safety", rf"\b(?:safe|ok|okay|dangerous|harmful)\b[^?]*{_HEALTH_STATEMENT}"),
    # ---- self-diagnosis and judging one's own values
    (
        "self_diagnosis",
        rf"{_CLAUSE}(?:do|does|could|might|am|is|can) {_SUBJ} (?:\w+ ){{0,2}}(?:have|has|be|got|"
        r"suffer|be suffering)\b",
    ),
    ("self_diagnosis", r"\b(?:diagnose|diagnosis for) (?:me|my)\b"),
    ("self_diagnosis", rf"\bwhat(?:'s| is) wrong with (?:me|my|{_REL}|him|her)\b"),
    (
        "self_diagnosis",
        rf"{_HEALTH_STATEMENT}[^?]*\b(?:is (?:it|this|that)|could (?:it|this|that) be|do i have|"
        r"what (?:is|could) (?:it|this|that)|what does (?:it|this|that) mean|should i (?:worry|be worried))\b",
    ),
    (
        "personal_results",
        rf"\bis my (?:\w+ )?{_MEASURE}\b[^?]*\b(?:dangerous|normal|high|low|bad|ok|okay|too|"
        r"serious|good|concerning|worrying)\b",
    ),
    (
        "personal_results",
        rf"\bmy (?:\w+ ){{0,2}}{_MEASURE} (?:is|are|was|were|came|show|shows|showed|suggest|suggests|suggested|"
        r"indicate|indicates|indicated|say|says|said)\b[^?]*\b"
        r"(?:mean|normal|dangerous|high|low|worry|should|bad|serious)\b",
    ),
    # ---- changing a specific person's medication; needing a treatment after a personal value (F-007)
    (
        "start_stop_treatment",
        rf"\b(?:should|can|could|may|must) (?:i|we) (?:\w+ ){{0,2}}(?:cut|halve|split|crush|stop|reduce|lower|"
        rf"increase|raise|change|switch|skip|double|give|take|share) (?:\w+ ){{0,2}}(?:his|her|their|my|our|"
        rf"{_REL}'s) (?:\w+ ){{0,3}}(?:pills?|medications?|medicines?|meds|doses?|dosage|tablets?|drugs?|"
        r"antibiotics?|insulin|inhalers?|patch|patches)\b",
    ),
    (
        "start_stop_treatment",
        rf"{_CLAUSE}should {_SUBJ} (?:\w+ ){{0,2}}(?:get|have|receive|undergo|be given)\b",
    ),
    (
        "start_stop_treatment",
        rf"(?:{_HEALTH_STATEMENT}|{_SUSPICION})[^?]*\b(?:do|does|should|would|will) (?:i|we|he|she|they|{_REL}) "
        r"(?:\w+ )?need\b",
    ),
    (
        "personal_management",
        rf"\bhow (?:long|soon|often|many days|many weeks) (?:should|do|must|can) (?:i|{_REL})\b",
    ),
    # ---- what to do in a personal situation
    ("personal_management", rf"\bwhat (?:should|do|can|must) (?:i|we|{_REL}) do\b"),
    ("personal_management", rf"\bwhat should {_SUBJ} (?:take|use|give|try)\b"),
    ("personal_management", r"\bwhat do i do\b"),
    # ---- a personal situation followed by a request to choose, confirm or interpret (DEV-informed, generic forms)
    (
        "personal_management",
        rf"{_HEALTH_STATEMENT}[^?]*\b(?:which|what)\b[^?]*\b(?:should|would|do|can) (?:i|we|you) (?:\w+ )?"
        r"(?:pick|choose|select|use|try|give|recommend|go with|start)\b",
    ),
    (
        "self_diagnosis",
        rf"(?:{_HEALTH_STATEMENT}|{_SUSPICION})[^?]*\b(?:(?:can|could|would) you (?:please )?(?:confirm|check|"
        r"tell (?:me )?(?:if|whether)|diagnose)|confirm (?:it|this|that|if|whether))\b",
    ),
    (
        "personal_results",
        r"\bwhat (?:do|does|did|would|could) (?:they|it|this|these|those|that|the results?) (?:\w+ )?mean for "
        rf"(?:me|my|us|him|her|{_REL})\b",
    ),
)

# Asking how to support a *generic* third party with suicidal thoughts is general crisis-support information
# (F-008b): it is not refused. A *specific* person ("my friend/son is thinking about suicide") gets the crisis
# message via the emergency rules, which run first.
SUPPORT_INFO = (
    r"\b(?:someone|somebody|anyone|a friend|friends|people|a loved one|loved ones|others|a person|"
    r"a family member|a student|a coworker|a colleague|a child|a teen|a teenager)\b[^?]*\b(?:thinking|talking) "
    r"(?:about|of) (?:suicide|killing (?:themselves|himself|herself)|ending (?:their|his|her) (?:own )?life|dying)"
)
_COMPILED = tuple((rid, re.compile(p)) for rid, p in RULES)
_SUPPORT_RE = re.compile(SUPPORT_INFO)
_SUPPORT_ASK_RE = re.compile(
    r"\b(?:how|what|ways?|tips?|signs?)\b[^?]*\b(?:support|help|talk|say|do|respond|recognize|approach|listen)\b"
)
_MEDICATION_RE = re.compile(
    r"\b(?:pills?|tablets?|medications?|medicines?|meds|dose|doses|took|swallowed|overdos\w*)\b"
)
SAFETY_RULES_VERSION = (
    "safety-v2+"
    + hashlib.sha256(
        (repr(RULES) + SUPPORT_INFO + _SUPPORT_ASK_RE.pattern + _MEDICATION_RE.pattern).encode()
    ).hexdigest()[:8]
)

REFUSAL_MESSAGE = (
    "I can't give personal medical advice, such as whether you have a condition, what dose to take, whether a "
    "result is dangerous, or whether to start or stop a treatment. Please ask a qualified clinician or pharmacist. "
    "I can answer general questions about a condition or treatment."
)
EMERGENCY_MESSAGE = (
    "This may be an emergency. Please contact your local emergency number (for example 911, 112 or 999) or a "
    "poison control or crisis line now. This research prototype cannot help with urgent or personal medical "
    "situations."
)


def _strip_format_chars(text: str) -> str:
    """Drop invisible format characters (Unicode category Cf: zero-width, bidi controls, soft hyphen, BOM)."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    refuse: bool
    rule_id: str | None = None

    @property
    def message(self) -> str:
        return EMERGENCY_MESSAGE if self.rule_id == "emergency" else REFUSAL_MESSAGE


def _normalize(question: str) -> str:
    q = unicodedata.normalize("NFKC", question)
    q = _strip_format_chars(q).lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", q).strip()


def check_question(question: str) -> SafetyDecision:
    q = _normalize(question)
    support_info = (
        _SUPPORT_RE.search(q) is not None and _SUPPORT_ASK_RE.search(q) is not None and _MEDICATION_RE.search(q) is None
    )
    for rule_id, pattern in _COMPILED:
        if support_info and rule_id != "emergency":
            break
        if pattern.search(q):
            return SafetyDecision(True, rule_id)
    return SafetyDecision(False, None)
