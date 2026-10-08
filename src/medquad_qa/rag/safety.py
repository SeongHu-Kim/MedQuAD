"""Rule-based refusal of personalized medical advice and crisis routing (all four experiment modes; D-023).

These rules are a scope policy, not a diagnosis: they decide whether a question asks for an individual
clinical decision (personal dosing, self-diagnosis, judging one's own results, choosing/starting/stopping a
treatment for a specific person, triage, emergencies) instead of general medical information.

safety-v4 (F-009, D-060/D-062): the safety-v3 rule stages below, plus a constrained model check
(``safety_check.py``) where the rules leave the decision open: questions no rule matched outside the safe
harbour are classified by the base model; rule-refused personal questions are also classified, but the model
may only escalate them to the crisis message (never-downgrade). Any check failure fails closed.

Rule stages (safety-v3):

1. **Crisis** (never overridden) -> ``rule_id="emergency"`` and the crisis message. A crisis lexicon covers
   suicidal ideation (explicit and passive, any tense), intent, plans and methods, self-harm, a specific person
   at risk, help-seeking for a person at risk, imminent medical emergencies, and intent to harm others.
   Medical-emergency *events* need a specific subject (I / my <relative> / he / she) or an immediacy cue
   ("right now", "just", "minutes ago"); generic first-aid or symptom questions are not crisis.
2. **Informational safe harbour**: an informational frame ("what are the signs of...", "how is ... treated")
   with no specific person (no I/me/my, no my/our <relative>) skips stage 3.
3. **Personal advice** (generic refusal): a specific person plus an advice cue (dose, medication change,
   safety for that person, interpretation of symptoms or results, treatment choice, triage, special
   populations, disguised framing, or a health narrative followed by any advice question).

Rules are incomplete by construction. Development used only DEV, ``tests/security`` probes, the owner-written
bank (``tests/rag/data/safety_bank_v1.jsonl`` tune split; its holdout is never tuned on) and the D-060 negatives
file (over-refusal counts only). Acceptance is decided by the evaluator's fresh held-out probes (D-059).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from medquad_qa.rag.safety_check import SAFETY_CHECK_DEFINITION, SAFETY_CHECK_FAILED, SafetyChecker

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
_GAP = r"[^.?!;]{0,80}?"  # within one sentence/clause

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

# ---------------------------------------------------------------------------------------------- crisis lexicon
_DEATH_WISH = (
    r"(?:want|wanna|wish|hope|hoping|rather|ready|going|planning|plan)(?: \w+){0,2} to (?:die|be dead|"
    r"not (?:be )?(?:alive|here|around|exist)|disappear (?:forever|for good|permanently)|sleep forever|"
    r"never wake up|not wake up|stop existing|end it|end everything|be gone for good)"
)
_PASSIVE = (
    r"wish i (?:could )?(?:fall asleep|go to sleep|sleep) and (?:never|not) wake up",
    r"(?:hope|wish|pray) i (?:just )?(?:won'?t|wouldn'?t|don'?t|never|didn'?t) wake up",
    r"(?:don'?t|do not|no longer) want to (?:live|be alive|be here|exist|wake up|go on|be around|keep living)",
    r"(?:better off|fine|happier|easier|better|okay|ok) without me",
    r"(?:easier|better) for (?:everyone|everybody|them|my family|my kids|all of them|the people)\b[^.?!]{0,30}"
    r"\bif i (?:wasn'?t|weren'?t|was not|were not|wasn't|weren't|didn'?t exist)",
    r"\bi(?:'m| am) (?:\w+ )?(?:just |such )?a burden\b",
    r"\bi (?:\w+ )?feel (?:like )?(?:i'?m |i am )?(?:\w+ )?a burden\b",
    r"(?:no|not any|isn'?t any|don'?t see (?:a|any)|can'?t see (?:a|any)) (?:point|reason|purpose) (?:in|to|of) "
    r"(?:living|going on|life|carry(?:ing)? on|being alive|keep(?:ing)? going|being here)",
    r"(?:don'?t|can'?t|do not|cannot) see (?:a|any) (?:future|way forward) for (?:me|myself)",
    r"\bno way out (?:except|but|other than)",
    r"(?:can'?t|don'?t|cannot|do not) see (?:a|any) way out\b",
    r"\b(?:i|me)\b[^.?!]{0,60}\bnot (?:being|be) (?:here|around|alive) (?:anymore|any more)?",
    r"\b(?:wrote|written|writing|left|leaving) (?:\w+ ){0,2}(?:letters?|notes?|messages?) "
    r"(?:to say )?(?:goodbye|farewell)\b",
    r"(?:life|living) (?:isn'?t|is not|is no longer|has no) (?:worth|point|meaning|purpose)",
    r"(?:my )?life (?:isn'?t|is not) worth living",
    r"\b(?:i'?m|i am) (?:so )?tired of (?:living|life|being alive|existing)",
    r"want (?:it all|everything|this all|all of it) to (?:stop|end) (?:for good|forever|permanently)",
    r"want (?:it all|everything) to (?:end|be over)\b",
    r"stopped caring (?:what happens to me|if i live|whether i live|about living|if i die|whether i die)",
    r"giving (?:away )?(?:my|all my) (?:things|stuff|belongings|possessions|pets)\b",
    r"\bsaid my goodbyes\b|\b(?:goodbye|farewell) (?:letters?|notes?|messages?) to\b",
    r"imagin\w* my (?:own )?funeral",
    r"(?:won'?t|will not|not going to) be (?:around|here) (?:much longer|for long|anymore|any longer|tomorrow)",
    r"(?:made|make|making) peace with (?:dying|death|leaving)",
    r"\bstop fighting\b|\bcan'?t (?:go on|keep going|do this anymore|carry on) (?:anymore|any longer|like this)?",
    r"(?:my family|everyone|they|my kids|my children|my wife|my husband) (?:would|will|could) (?:be|do) "
    r"(?:fine|better|happier|okay|ok) without me",
    r"(?:the night|the day|time) (?:i|to) (?:finally )?(?:stop fighting|end it|give up on life)",
    r"\b(?:i'?m|i am) (?:\w+ )?suicidal\b",
)
_METHOD = (
    r"(?:saving|saved|stockpiling|hoarding|collecting) (?:up )?(?:my |all my |some )?(?:\w+ )?"
    r"(?:pills|tablets|meds|medication|painkillers)\b",
    r"how many (?:of )?(?:my |these )?(?:\w+ ){0,2}(?:would it take|does it take|do i need) to (?:die|never wake up|"
    r"not wake up|overdose|end it|kill myself|be dead)",
    r"(?:looked up|looking up|searched|searching|researching|googled) (?:how|ways) (?:to (?:die|kill myself|overdose|"
    r"end it)|many (?:of )?(?:my|these) )",
    r"\b(?:jump|jumping|stepping off|step off)\b[^.?!]{0,40}\b(?:bridge|roof|ledge|cliff|balcony|building)\b",
    r"\b(?:bridge|roof|ledge|cliff|balcony|tracks)\b[^.?!]{0,60}\b(?:jump|jumping|stepping off|step off)\b",
    r"standing on (?:the|a|my) (?:roof|ledge|bridge|balcony)",
    r"\b(?:bought|got|have) (?:a )?(?:rope|noose|gun)\b[^.?!]{0,60}\b(?:using it|use it|end it|thinking)",
    r"thinking (?:about|of) (?:jumping|stepping off|overdosing|hanging myself|ending it)",
)
_SELF_HARM = (
    r"(?:cut|cutting|burn|burned|burnt|burning|hurt|hurting|harm|harming|hit|hitting|punch|punching|scratch|"
    r"scratching|bang|banging|bit|biting) (?:myself|my (?:arms?|legs?|wrists?|skin|thighs?|head|body|stomach))"
    r"\b[^.?!]{0,40}\b(?:again|on purpose|deliberately|to feel|until it bleeds|until it bled|until i bleed|"
    r"can'?t stop|when i'?m|every time|to cope|to punish)",
    r"\b(?:i|i'm|i've|been) (?:\w+ )?self-?harm(?:ing|ed)?\b",
    r"(?:cut|burned|burnt|hurt|harmed|hit) myself\b",
)
_REL_AT_RISK = (
    rf"\b{_REL}\b[^?.!]{{0,60}}?\b(?:told me|said|says|keeps saying|texted|posted|wrote|messaged|threatened)\b"
    r"[^?.!]{0,40}\b(?:going to|wants? to|will|plans? to|is going to) (?:kill (?:himself|herself|themselves)|die|"
    r"end it|end (?:his|her|their) (?:own )?life|not be here)",
    rf"\b{_REL}\b[^?.!]{{0,60}}?\b(?:goodbye|farewell|suicide) (?:note|letter|message|post|text)",
    rf"\b{_REL}\b[^?.!]{{0,60}}?\b(?:locked|shut) (?:himself|herself|themselves) in\b[^?.!]{{0,60}}"
    r"(?:razor|blades?|pills|knife|knives|gun|rope)",
    rf"\b{_REL}\b[^?.!]{{0,60}}?\b(?:has|had|have|got) (?:\w+ )?(?:pills|a gun|a rope)\b[^?.!]{{0,40}}"
    r"(?:saved|stockpiled|to end it|to die)",
    rf"\b{_REL}\b[^?.!]{{0,40}}?\b(?:doesn'?t|does not|don'?t) want to (?:live|be alive|be here|go on)",
)
_AT_RISK_TERMS = (
    rf"(?:suicid\w*|want(?:s|ing)? to die|{_THIRD_PARTY_HARM}|kill(?:ing)? (?:themselves|himself|herself)|"
    r"end(?:ing)? (?:their|his|her) (?:own )?life|self-?harm\w*|(?:doesn'?t|does not|don'?t) want to "
    r"(?:live|be alive|go on)|(?:took|swallowed) (?:a bunch of|a lot of|all (?:of )?(?:her|his|their|the)|too many) "
    r"(?:\w+ )?(?:pills|tablets))"
)
_HELP_SEEKING = (
    r"\b(?:what (?:should|can|do|must) (?:i|we) do|how (?:do|can|should) (?:i|we) "
    r"(?:help|support|talk to|deal with|handle|approach|save|stop)|what (?:should|can|do) (?:i|we) say)\b"
)

# Medical-emergency events: need a specific subject in the text, or an immediacy cue.
_EVENTS = (
    r"(?:stopped|stops|isn'?t|is not|not|barely|struggling to|can'?t|cannot|can not) breath(?:e|ing)\b",
    r"\bgasping\b|\bturning blue\b|\blips (?:are |look |turned |went )?blue\b|\bblue lips\b",
    r"\bunresponsive\b|\bunconscious\b|\bpassed out\b|\bjust collapsed\b|\bcollapsed and\b|"
    r"\bwon'?t wake (?:up)?\b|\bcan'?t wake (?:him|her|them)\b",
    r"\b(?:crushing|severe|heavy) chest (?:pain|pressure)\b|\bclutching (?:his|her|my) chest\b",
    r"\bchest (?:pain|hurts|is tight|feels tight|pressure)\b[^.?!]{0,60}\b(?:jaw|arm|back|sweat|sweaty|"
    r"sweating|breath|numb)",
    r"\bface (?:is )?drooping\b|\bdrooping on one side\b|\bslurred speech\b|\bspeech is slurred\b|"
    r"\bcan'?t (?:move|feel) (?:one side|his arm|her arm|my arm|his leg|her leg|my leg)\b",
    r"\bthroat (?:is )?(?:closing|swelling|tight)\b|\b(?:face|tongue|lips|mouth) (?:is |are )?(?:swelling|swollen)\b",
    r"\b(?:seizing|seizure|convulsing|fitting) (?:for|that|and)\b[^.?!]{0,30}\b(?:minutes|mins)\b|"
    r"\b(?:seizure|fit|convulsions?) (?:won'?t|isn'?t|doesn'?t|will not) stop",
    r"\bbleeding (?:won'?t|wont|isn'?t|doesn'?t|does not|will not) stop\b|\b(?:gushing|spurting) blood\b",
    r"\bswallowed (?:a |an |some |the |two |several )?(?:button )?(?:battery|batteries|magnets?|bleach|detergent|"
    r"laundry pods?|pods?|chemicals?|poison|antifreeze|drain cleaner|coins?|pills|tablets)\b",
    r"\bdrank (?:some |a bottle of |the )?(?:bleach|antifreeze|cleaner|chemicals?|poison|lighter fluid|kerosene|"
    r"paint thinner)\b",
    r"\btook (?:a whole|an entire|the whole|half a) (?:bottle|pack|box|packet)\b|"
    r"\btook (?:\d+|all|too many|a bunch|a handful)\b[^.?!]{0,30}\b(?:pills|tablets|capsules)\b|"
    r"\b(?:overdosed|overdosing) on\b|\btook an overdose\b|\btook (?:my|a|his|her) (?:\w+ )?week'?s worth\b|"
    r"\bmix(?:ed|ing) alcohol (?:and|with) (?:pills|sleeping pills|\w+ pills)\b",
    r"\bchoking and (?:can'?t|cannot)\b|\bchoking (?:right )?now\b",
    r"\bhaving a (?:heart attack|stroke)\b",
    r"\b(?:stung|bitten) by\b[^.?!]{0,30}\b(?:swelling|can'?t breathe|throat)",
)
_EVENT = "(?:" + "|".join(_EVENTS) + ")"
_EVENT_SUBJECT = rf"(?:\b(?:i|i'm|i've|i just|my)\b|{_REL}|\b(?:he|she)\b)"
# the gap between subject and event must not introduce a generic third party ("how do I help someone who...")
_NO_GENERIC_GAP = (
    r"(?:(?!\b(?:someone|somebody|anyone|a person|people|a child|a patient|bystanders?|others|a stranger)\b)"
    r"[^.?!;]){0,80}?"
)
_IMMEDIACY = r"\b(?:right now|just now|currently|at the moment|minutes ago|an hour ago|hours ago|just|tonight)\b"
_HARM_OTHERS = (
    r"\b(?:i|i'm|i am|i've|i keep)\b[^.?!]{0,40}\b(?:want|going|plan(?:ning)?|thoughts? (?:about|of)|"
    r"thinking (?:about|of)|urge) (?:to )?(?:hurt|kill|killing|harm|harming|attack|attacking|stab|stabbing|shoot|"
    r"shooting) (?:someone|somebody|people|my \w+|him|her|them|everyone|a person)"
)

# (rule_id, pattern). Stage 1: crisis -- any match routes to the crisis message, never overridden.
CRISIS_RULES: tuple[tuple[str, str], ...] = (
    ("emergency", "(?:" + "|".join(_PASSIVE) + ")"),
    ("emergency", rf"\b(?:i|i'm|im|i am|i've|i really|i just)\b[^.?!]{{0,30}}\b{_DEATH_WISH}"),
    ("emergency", "(?:" + "|".join(_METHOD) + ")"),
    ("emergency", "(?:" + "|".join(_SELF_HARM) + ")"),
    ("emergency", "(?:" + "|".join(_REL_AT_RISK) + ")"),
    ("emergency", rf"{_HELP_SEEKING}[^?]*\b{_AT_RISK_TERMS}"),
    ("emergency", rf"\b{_AT_RISK_TERMS}\b[^?]*{_HELP_SEEKING}"),
    ("emergency", rf"{_EVENT_SUBJECT}{_NO_GENERIC_GAP}{_EVENT}"),
    ("emergency", rf"^(?=.*{_IMMEDIACY})(?=.*{_EVENT})"),
    ("emergency", _HARM_OTHERS),
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
        r"(?:help|support|talk to|deal with|handle|approach|save)|what (?:should|can) (?:i|we) say)\b"
        r"[^?]*\b(?:suicid\w*|want(?:s|ing)? to die|"
        rf"{_THIRD_PARTY_HARM}|kill(?:ing)? (?:themselves|himself|herself)|end(?:ing)? (?:their|his|her) (?:own )?life|"
        r"self-?harm\w*)",
    ),
    # ---- events or intent carried over from safety-v2 (subject-bound)
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
        r"\b(?:my|i have) (?:chest (?:hurts|pain|is tight|feels tight))\b[^?]{0,60}\b(?:arm|jaw|sweat|"
        r"breath|numb)",
    ),
    ("emergency", r"\bchest pain (?:right )?now\b"),
    (
        "emergency",
        r"\b(?:kill|hurt|harm|cut) (?:myself|me)\b|\b(?:i'm|im|i am|i feel|feeling|i've been|been) (?:\w+ )?suicidal\b|"
        r"\b(?:i|i'm|im) (?:going|planning|want) to (?:overdose|end it)\b",
    ),
)

# Stage 3: personal advice -- generic refusal.
PERSONAL_RULES: tuple[tuple[str, str], ...] = (
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
    # ---- safety-v3 additions -----------------------------------------------------------------------------
    # dose/amount for a specific person ("how many of my X can I take")
    (
        "personal_dosing",
        rf"^(?=.*\b{_PERSON}\b)(?=.*\bhow (?:much|many)\b[^?]*\b(?:take|inject|give|use|double|apply|have)\b)",
    ),
    # safety for a specific person, conditional forms ("would it be fine for me to...")
    (
        "personal_safety",
        r"\b(?:is|would|will|could|might) (?:it|this|that) (?:\w+ )?(?:be )?(?:safe|ok|okay|fine|alright|all right|"
        rf"dangerous|harmful|bad|risky|a problem) (?:for|to|if) (?:me|my|i|{_REL}|him|her)\b",
    ),
    # triage for a specific situation
    (
        "triage",
        rf"^(?=.*\b{_PERSON}\b)(?=.*(?:\bwait (?:until|till|for)\b|\bcan (?:it|this|that) wait\b|"
        r"\bdo (?:i|we|they|he|she) need (?:to go|to see|to be seen|stitches|an? x-?ray|a tetanus|antibiotics|surgery|"
        r"a doctor|to call)\b|\bshould (?:i|we|he|she|they) (?:\w+ )?(?:go|call|see|take (?:him|her|them)|"
        r"head|ring|phone)\b|\bget (?:it|this|that|him|her|them) (?:checked|looked at|seen)\b|"
        r"\bneed to be seen\b|\bhow urgent(?:ly)?\b|\bis (?:it|this|that) (?:urgent|an emergency|serious)\b|"
        r"\b(?:something|anything) (?:i|we) should (?:worry|be worried) about\b|\bshould (?:i|we) (?:be )?worr|"
        r"\btake (?:him|her|them) in\b|\burgent care\b|\bthe er\b|\ba&e\b|\bemergency room\b))",
    ),
    # a health narrative followed by a modal action question ("... should we stop it?", "can I hide it in...")
    (
        "medication_change",
        rf"{_HEALTH_STATEMENT}[^?]*\b(?:should|can|could|may|must) (?:i|we|he|she|they) (?:\w+ ){{0,2}}"
        r"(?:stop|start|quit|skip|double|halve|cut|crush|switch|change|hide|sneak|slip|put|give|keep|continue|"
        r"replace|reduce|increase|take|use|try|mix|combine)\b",
    ),
    (
        "medication_change",
        rf"\b{_REL}\b[^?]*\b(?:after|since) (?:starting|taking|switching to|beginning)\b[^?]*"
        r"\b(?:should|can|could) (?:i|we|he|she|they)\b",
    ),
    (
        "medication_change",
        r"\b(?:can|could|should|may) (?:i|we) (?:\w+ ){0,2}(?:hide|sneak|slip|crush|mix) (?:it|them|his|her|the|my)\b",
    ),
    # disguised framing that still concerns the asker
    (
        "disguised",
        r"^(?=.*\b(?:asking for a (?:friend|buddy|coworker|colleague|mate)|hypothetically|theoretically|"
        r"just (?:wondering|curious|asking)|purely (?:out of interest|theoretical(?:ly)?|hypothetical(?:ly)?)|"
        r"for a (?:novel|story|book|friend)|not for me|it'?s (?:actually )?me|who is me|asking for myself)\b)"
        rf"(?=.*\b(?:{_SELF}|{_REL})\b)(?=.*(?:\?|\b(?:should|can|could|would|is it|how (?:much|many))\b))",
    ),
    # what would you recommend / what should I do for a specific person
    (
        "treatment_choice",
        rf"\b(?:recommend|suggest|prescribe|advise)\b[^?]*\bfor (?:me|my|{_REL}|him|her)\b",
    ),
    (
        "treatment_choice",
        rf"\b(?:should|would) (?:i|we|he|she|they|{_REL}) (?:\w+ )?(?:get|have|choose|pick|go for|opt for|try|do)\b"
        r"[^?]*\bor\b",
    ),
    (
        "treatment_choice",
        r"\b(?:what|which) (?:\w+ ){0,3}(?:should|would|can|do) (?:i|we|he|she|they) (?:\w+ )?(?:take|use|put|"
        r"apply|follow|buy|eat|try|go on|get|choose|pick)\b[^?]*\b(?:for|on|with) (?:my|our|his|her|their)\b",
    ),
    # symptom/finding interpretation for a specific person
    (
        "self_diagnosis",
        rf"^(?=.*\b{_PERSON}\b)(?=.*(?:\b(?:is|could|might|does) (?:it|this|that) (?:\w+ )?(?:be|mean|a sign)\b|"
        r"\b(?:am i|is (?:he|she)|are (?:my|his|her)) [^?]{0,40}\b(?:getting|having|failing|dying|infected|"
        r"dehydrated|normal)\b|\bwhat(?:'s| is| could be) (?:causing|wrong)\b|\bwhat (?:does|do) (?:\w+ )?"
        r"(?:she|he|i) have\b|\bdid i (?:tear|break|damage|pull|sprain)\b|\bis (?:that|this|it) (?:normal|infected|"
        r"cancer|serious|bad)\b|\bhow bad is\b|\bshould i be (?:scared|worried|concerned)\b|"
        r"\bwhat are the chances\b|\bdo i have\b))",
    ),
    (
        "personal_results",
        rf"\bmy (?:\w+ ){{0,3}}(?:{_MEASURE}|ferritin|inr|platelets|bilirubin|egfr|enzymes|pap smear|pulse|"
        r"reading|nodule|lead level)\b[^?]*\b(?:\d|high|low|abnormal|positive|negative|came back|dropped|went|rose)",
    ),
    # amount words + an administration verb for a specific person
    (
        "personal_dosing",
        rf"^(?=.*\b{_PERSON}\b)(?=.*\b(?:what amount|the right amount|how much|how many|how often)\b[^?]*"
        r"\b(?:put|give|giving|take|taking|use|apply|inject|drink|eat|spray)\b)",
    ),
    # "is something wrong with my X", "is it an ulcer" after a personal symptom
    (
        "self_diagnosis",
        rf"\b(?:is|could|might) (?:something|anything) (?:be )?wrong with (?:me|my|{_REL}|him|her)\b",
    ),
    (
        "self_diagnosis",
        r"^(?=.*\b(?:my|i)\b)(?=.*\b(?:is|could) (?:it|this|that) (?:be )?(?:a|an|my)\b)",
    ),
    # comparing options for a specific person
    (
        "treatment_choice",
        rf"\b(?:what'?s|what is|which is|which'?s) (?:the )?(?:best|better|right|safest|most effective) "
        rf"(?:\w+ ){{0,2}}for (?:me|my|{_REL}|him|her)\b",
    ),
    (
        "treatment_choice",
        rf"\bor\b[^?]*\b(?:better|best|safer|worse|work better)\b[^?]*\bfor (?:me|my|{_REL}|him|her)\b",
    ),
    # special populations, self- or relative-specific
    (
        "special_population",
        r"\b(?:i'?m|i am) (?:\d+ (?:weeks|months) )?(?:pregnant|breastfeeding|nursing|trying to (?:get pregnant|"
        r"conceive)|post[- ]?partum|\d+ weeks post)\b",
    ),
    (
        "special_population",
        r"\b(?:my|our) (?:\w+ )?(?:\d+[- ]?(?:month|week|day|year)s?[- ]?old|newborn|baby|infant|toddler|"
        r"premature baby|preemie)\b[^?]*\b(?:can|should|is it|how much|safe|okay|ok)\b",
    ),
    # health narrative + any advice question in the same message (fallback)
    (
        "narrative_advice",
        rf"{_HEALTH_STATEMENT}[^?]*\b(?:can|could|should|may|must|is it|would it|do i|does (?:he|she)|what should|"
        r"how (?:much|many|long|soon|often)|which|when can|is that|is this)\b",
    ),
    (
        "narrative_advice",
        r"\b(?:i|i've|ive|i was|i've been|i got|i just got) (?:\w+ )?(?:diagnosed|told i have|found|noticed|"
        r"had (?:my|a|an)|been (?:recovering|having))\b[^?]*\?",
    ),
    (
        "narrative_advice",
        rf"\b{_REL} (?:has|had|is|was|keeps|weighs|got) [^?]*\b(?:can|could|should|is it|would it|how)\b",
    ),
)

# Stage 2: informational safe harbour (applies only if stage 1 did not fire).
INFO_FRAME = (
    r"^(?:what (?:is|are|causes|happens|does|do|should people|can people)|how (?:is|are|do|does|can|common|many|"
    r"often|long)|who (?:is|are|should|gets?)|why (?:do|does|is|are)|when (?:do|does|is|are|should people)|"
    r"which|is there|are there|can (?:you|people|children|adults|someone)|tell me about)\b"
)
GENERIC_SUPPORT = (
    r"\b(?:someone|somebody|anyone|a friend|friends|people|a loved one|loved ones|others|a person|"
    r"a family member|a student|a coworker|a colleague|a child|a teen|a teenager)\b[^?]*\b(?:thinking|talking) "
    r"(?:about|of) (?:suicide|killing (?:themselves|himself|herself)|ending (?:their|his|her) (?:own )?life|dying)"
)
_WEIGHT_AGE = r"\bfor (?:a|an) \d+(?:\.\d+)? ?(?:kg|kilo|kilos|lb|lbs|pounds?|[- ]?(?:year|month|week)s?[- ]?old)\b"
_SPECIFIC_PERSON = rf"\b(?:{_SELF}|{_REL})\b"
_MEDICATION_TERMS = r"\b(?:pills?|tablets?|medications?|medicines?|meds|dose|doses|took|swallowed|overdos\w*)\b"
_SUPPORT_ASK = (
    r"\b(?:how|what|ways?|tips?|signs?)\b[^?]*\b(?:support|help|talk|say|do|respond|recognize|approach|listen)\b"
)

RULES: tuple[tuple[str, str], ...] = CRISIS_RULES + PERSONAL_RULES

# safety-v4 routing (D-062); part of the version hash.
ROUTING_POLICY = (
    "crisis_rule->emergency(no model)|safe_harbour->answer(no model)|"
    "personal_rule->refuse;model may only escalate CRISIS->emergency, never release|"
    "no_rule->model:CRISIS->emergency,PERSONAL->model_personal,GENERAL->answer|"
    "any check failure->safety_check_failed(crisis message, refuse)"
)
SAFETY_RULES_VERSION = (
    "safety-v4+"
    + hashlib.sha256(
        "\x1e".join(
            [
                repr(RULES),
                INFO_FRAME,
                GENERIC_SUPPORT,
                _WEIGHT_AGE,
                _SPECIFIC_PERSON,
                _MEDICATION_TERMS,
                _SUPPORT_ASK,
                ROUTING_POLICY,
                SAFETY_CHECK_DEFINITION,
            ]
        ).encode()
    ).hexdigest()[:8]
)

_CRISIS = tuple(re.compile(p) for _, p in CRISIS_RULES)
_PERSONAL = tuple((rid, re.compile(p)) for rid, p in PERSONAL_RULES)
_INFO_RE = re.compile(INFO_FRAME)
_GENERIC_SUPPORT_RE = re.compile(GENERIC_SUPPORT)
_WEIGHT_AGE_RE = re.compile(_WEIGHT_AGE)
_SPECIFIC_RE = re.compile(_SPECIFIC_PERSON)
_MEDICATION_RE = re.compile(_MEDICATION_TERMS)
_SUPPORT_ASK_RE = re.compile(_SUPPORT_ASK)

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


#: rule_ids whose message is the crisis message (a failed check fails closed with crisis resources).
_CRISIS_MESSAGE_IDS = frozenset({"emergency", SAFETY_CHECK_FAILED})


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    refuse: bool
    rule_id: str | None = None
    stage: str | None = None  # crisis_rule | safe_harbour | personal_rule | model | rules_only
    warnings: tuple[str, ...] = ()

    @property
    def message(self) -> str:
        return EMERGENCY_MESSAGE if self.rule_id in _CRISIS_MESSAGE_IDS else REFUSAL_MESSAGE


def _normalize(question: str) -> str:
    q = unicodedata.normalize("NFKC", question)
    q = _strip_format_chars(q).lower().replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", q).strip()


def _safe_harbour(q: str) -> bool:
    """Informational question with no specific person, or support info about a generic third party."""
    if _MEDICATION_RE.search(q) and _SPECIFIC_RE.search(q):
        return False
    if _GENERIC_SUPPORT_RE.search(q) and _SUPPORT_ASK_RE.search(q) and not _MEDICATION_RE.search(q):
        return True
    return bool(_INFO_RE.search(q)) and not _SPECIFIC_RE.search(q) and not _WEIGHT_AGE_RE.search(q)


def _failed(stage: str) -> SafetyDecision:
    return SafetyDecision(True, SAFETY_CHECK_FAILED, stage, (SAFETY_CHECK_FAILED,))


def check_question(question: str, *, checker: SafetyChecker | None = None) -> SafetyDecision:
    """Safety decision for one question (all four modes).

    Without ``checker`` only the rules run (offline tests; the rule stages are identical). With a checker the
    model classifies questions the rules leave open, and may escalate a rule-refused question to the crisis
    message, but a personal-advice refusal is never released as an answer (never-downgrade, D-062).
    """
    q = _normalize(question)
    if any(p.search(q) for p in _CRISIS):
        return SafetyDecision(True, "emergency", "crisis_rule")
    if _safe_harbour(q):
        return SafetyDecision(False, None, "safe_harbour")
    rule_id = next((rid for rid, pattern in _PERSONAL if pattern.search(q)), None)
    if checker is None:
        return SafetyDecision(rule_id is not None, rule_id, "personal_rule" if rule_id else "rules_only")
    result = checker.classify(question)
    if rule_id is not None:
        # never-downgrade: the model can only escalate a rule refusal to the crisis message
        if result.label is None:
            return _failed("personal_rule")
        if result.label == "CRISIS":
            return SafetyDecision(True, "emergency", "personal_rule")
        return SafetyDecision(True, rule_id, "personal_rule")
    if result.label is None:
        return _failed("model")
    if result.label == "CRISIS":
        return SafetyDecision(True, "emergency", "model")
    if result.label == "PERSONAL":
        return SafetyDecision(True, "model_personal", "model")
    return SafetyDecision(False, None, "model")
