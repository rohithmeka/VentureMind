
import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from scipy.optimize import linprog
import difflib

RANDOM_SEED = 42

SKILL_TAXONOMY = [
    # tech
    "backend", "frontend", "mobile_dev", "data", "data_science", "machine_learning",
    "hardware", "engineering", "mechanical_engineering", "electrical_engineering", "security",
    # design / creative
    "design", "ui_ux", "content", "copywriting", "video_production", "photography",
    # business
    "sales", "marketing", "digital_marketing", "finance", "accounting", "hr", "recruiting",
    "product_management", "project_management", "public_relations", "customer_support",
    # operations / legal
    "operations", "logistics", "supply_chain", "compliance", "legal", "policy", "construction",
    # domain-specific
    "healthcare", "pedagogy", "teaching", "biotech", "agriculture", "real_estate", "hospitality",
]

# just for pretty-printing the menu and for report readability -- the model
# always works off the plain tag on the left, this is purely cosmetic
SKILL_LABELS = {
    "backend": "Backend Development", "frontend": "Frontend Development",
    "mobile_dev": "Mobile App Development", "data": "Data Analysis",
    "data_science": "Data Science", "machine_learning": "Machine Learning / AI",
    "hardware": "Hardware / Electronics", "engineering": "General Engineering",
    "mechanical_engineering": "Mechanical Engineering", "electrical_engineering": "Electrical Engineering",
    "security": "Security / Cybersecurity", "design": "Graphic Design", "ui_ux": "UI/UX Design",
    "content": "Content Writing", "copywriting": "Copywriting", "video_production": "Video Production",
    "photography": "Photography", "sales": "Sales", "marketing": "Marketing",
    "digital_marketing": "Digital Marketing / SEO", "finance": "Finance", "accounting": "Accounting",
    "hr": "HR / People Ops", "recruiting": "Recruiting / Talent Acquisition",
    "product_management": "Product Management", "project_management": "Project Management",
    "public_relations": "Public Relations (PR)", "customer_support": "Customer Support",
    "operations": "Operations", "logistics": "Logistics", "supply_chain": "Supply Chain",
    "compliance": "Regulatory Compliance", "legal": "Legal", "policy": "Policy / Govt Relations",
    "construction": "Construction", "healthcare": "Healthcare", "pedagogy": "Curriculum / Pedagogy",
    "teaching": "Teaching", "biotech": "Biotechnology", "agriculture": "Agriculture",
    "real_estate": "Real Estate", "hospitality": "Hospitality / Travel",
}

SKILL_GROUPS = {
    "Tech": ["backend", "frontend", "mobile_dev", "data", "data_science", "machine_learning",
             "hardware", "engineering", "mechanical_engineering", "electrical_engineering", "security"],
    "Design / Creative": ["design", "ui_ux", "content", "copywriting", "video_production", "photography"],
    "Business": ["sales", "marketing", "digital_marketing", "finance", "accounting", "hr", "recruiting",
                 "product_management", "project_management", "public_relations", "customer_support"],
    "Operations / Legal": ["operations", "logistics", "supply_chain", "compliance", "legal", "policy",
                            "construction"],
    "Domain-Specific": ["healthcare", "pedagogy", "teaching", "biotech", "agriculture",
                         "real_estate", "hospitality"],
}

# common alternate phrasings -> canonical tag, checked before fuzzy matching
SKILL_SYNONYMS = {
    "coding": "backend", "programming": "backend", "software": "backend", "software development": "backend",
    "web dev": "frontend", "web development": "frontend", "website": "frontend",
    "app dev": "mobile_dev", "android": "mobile_dev", "ios": "mobile_dev", "app development": "mobile_dev",
    "ml": "machine_learning", "ai": "machine_learning", "artificial intelligence": "machine_learning",
    "analytics": "data_science", "data analytics": "data_science",
    "electronics": "hardware", "embedded": "hardware", "embedded systems": "hardware",
    "cyber security": "security", "cybersecurity": "security", "infosec": "security",
    "graphic design": "design", "ux": "ui_ux", "ui": "ui_ux", "product design": "ui_ux",
    "writing": "content", "copy writing": "copywriting", "video editing": "video_production",
    "seo": "digital_marketing", "growth marketing": "digital_marketing", "branding": "marketing",
    "accountant": "accounting", "bookkeeping": "accounting", "law": "legal", "lawyer": "legal",
    "hiring": "recruiting", "talent acquisition": "recruiting", "people ops": "hr",
    "pm": "product_management", "scrum": "project_management", "agile": "project_management",
    "pr": "public_relations", "support": "customer_support",
    "supply chain management": "supply_chain", "govt relations": "policy", "government relations": "policy",
    "civil engineering": "construction", "medicine": "healthcare", "doctor": "healthcare",
    "nursing": "healthcare", "farming": "agriculture", "property": "real_estate", "travel": "hospitality",
}


def _describe_skill_menu() -> str:
    lines = ["Available skill tags (type these, comma-separated):"]
    for group, tags in SKILL_GROUPS.items():
        readable = ", ".join(f"{t} ({SKILL_LABELS[t]})" if SKILL_LABELS[t].lower() != t.replace("_", " ")
                              else t for t in tags)
        lines.append(f"  {group}: {readable}")
    return "\n".join(lines)


SECTOR_TAXONOMY = [
    "FinTech", "EdTech", "HealthTech", "AgriTech", "E-commerce",
    "SaaS/B2B", "FoodTech", "CleanTech", "Logistics/Mobility", "Gaming/Media",
    "Cybersecurity", "InsurTech", "Travel & Hospitality", "PropTech/RealEstate",
    "HRTech", "D2C Consumer Brands",
]

SECTOR_SYNONYMS = {
    "fintech": "FinTech", "ed tech": "EdTech", "edtech": "EdTech", "health tech": "HealthTech",
    "healthtech": "HealthTech", "agritech": "AgriTech", "agri tech": "AgriTech",
    "ecommerce": "E-commerce", "e commerce": "E-commerce", "online retail": "E-commerce",
    "saas": "SaaS/B2B", "b2b": "SaaS/B2B", "b2b saas": "SaaS/B2B", "food tech": "FoodTech",
    "foodtech": "FoodTech", "clean tech": "CleanTech", "cleantech": "CleanTech",
    "renewable energy": "CleanTech", "mobility": "Logistics/Mobility", "logistics": "Logistics/Mobility",
    "gaming": "Gaming/Media", "media": "Gaming/Media", "entertainment": "Gaming/Media",
    "cyber security": "Cybersecurity", "cybersecurity": "Cybersecurity", "infosec": "Cybersecurity",
    "insurance": "InsurTech", "insurtech": "InsurTech", "travel": "Travel & Hospitality",
    "hospitality": "Travel & Hospitality", "tourism": "Travel & Hospitality",
    "real estate": "PropTech/RealEstate", "proptech": "PropTech/RealEstate", "property": "PropTech/RealEstate",
    "hr tech": "HRTech", "hrtech": "HRTech", "human resources": "HRTech",
    "d2c": "D2C Consumer Brands", "consumer brands": "D2C Consumer Brands", "dtc": "D2C Consumer Brands",
}


def _describe_sector_menu() -> str:
    return "Available sector tags (type these, comma-separated):\n  " + ", ".join(SECTOR_TAXONOMY)


def _match_free_text_tokens(tokens, taxonomy, synonyms, fuzzy_cutoff=0.8):

    taxonomy_lower_map = {t.lower(): t for t in taxonomy}
    matched, unmatched = [], []

    for raw in tokens:
        t = raw.strip().lower()
        if not t:
            continue

        if t in taxonomy_lower_map:
            matched.append(taxonomy_lower_map[t])
            continue

        if t in synonyms:
            matched.append(synonyms[t])
            continue

        candidates = list(taxonomy_lower_map.keys()) + list(synonyms.keys())
        close = difflib.get_close_matches(t, candidates, n=1, cutoff=fuzzy_cutoff)
        if close:
            hit = close[0]
            matched.append(taxonomy_lower_map.get(hit) or synonyms.get(hit))
            continue

        unmatched.append(raw.strip())

    # de-duplicate while preserving first-seen order
    seen, deduped = set(), []
    for m in matched:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped, unmatched


RISK_LABEL_MAP = {"low": 0.2, "medium": 0.5, "moderate": 0.5, "high": 0.85}
TIME_LABEL_MAP = {"part-time": 0.3, "part time": 0.3, "side-project": 0.2,
                   "full-time": 1.0, "full time": 1.0, "weekends-only": 0.15}


# =============================================================================
# MODULE 2 (data layer) - sector / state benchmark tables
# =============================================================================
# demand_index          : 0-1, how strong current demand is
# growth_rate            : yearly growth (0.18 = 18%)
# competition_density    : 0-1, how crowded the sector is
# avg_capital_required   : typical rupees needed to get to first revenue
# historical_failure_rate: rough 3 year failure rate for new entrants
# required_skills        : skills a founder should ideally have
SECTOR_TABLE = pd.DataFrame([
    dict(sector="FinTech",        demand_index=0.82, growth_rate=0.21, competition_density=0.78,
         avg_capital_required=2_500_000, historical_failure_rate=0.62,
         required_skills=["finance", "compliance", "backend", "security"]),
    dict(sector="EdTech",         demand_index=0.63, growth_rate=0.09, competition_density=0.71,
         avg_capital_required=1_200_000, historical_failure_rate=0.58,
         required_skills=["content", "sales", "frontend", "pedagogy"]),
    dict(sector="HealthTech",     demand_index=0.77, growth_rate=0.24, competition_density=0.55,
         avg_capital_required=3_500_000, historical_failure_rate=0.55,
         required_skills=["healthcare", "compliance", "backend", "data"]),
    dict(sector="AgriTech",       demand_index=0.58, growth_rate=0.15, competition_density=0.34,
         avg_capital_required=1_800_000, historical_failure_rate=0.50,
         required_skills=["operations", "hardware", "logistics", "sales"]),
    dict(sector="E-commerce",     demand_index=0.70, growth_rate=0.12, competition_density=0.88,
         avg_capital_required=2_000_000, historical_failure_rate=0.65,
         required_skills=["marketing", "logistics", "frontend", "sales"]),
    dict(sector="SaaS/B2B",       demand_index=0.74, growth_rate=0.19, competition_density=0.60,
         avg_capital_required=1_500_000, historical_failure_rate=0.52,
         required_skills=["backend", "frontend", "sales", "data"]),
    dict(sector="FoodTech",       demand_index=0.55, growth_rate=0.08, competition_density=0.80,
         avg_capital_required=1_600_000, historical_failure_rate=0.66,
         required_skills=["operations", "logistics", "marketing"]),
    dict(sector="CleanTech",      demand_index=0.61, growth_rate=0.27, competition_density=0.28,
         avg_capital_required=4_500_000, historical_failure_rate=0.48,
         required_skills=["hardware", "engineering", "policy", "data"]),
    dict(sector="Logistics/Mobility", demand_index=0.66, growth_rate=0.14, competition_density=0.50,
         avg_capital_required=2_800_000, historical_failure_rate=0.53,
         required_skills=["operations", "logistics", "backend", "hardware"]),
    dict(sector="Gaming/Media",   demand_index=0.59, growth_rate=0.17, competition_density=0.62,
         avg_capital_required=1_000_000, historical_failure_rate=0.60,
         required_skills=["frontend", "design", "content", "marketing"]),
    dict(sector="Cybersecurity",  demand_index=0.72, growth_rate=0.22, competition_density=0.45,
         avg_capital_required=3_000_000, historical_failure_rate=0.50,
         required_skills=["security", "backend", "compliance", "data"]),
    dict(sector="InsurTech",      demand_index=0.68, growth_rate=0.18, competition_density=0.40,
         avg_capital_required=2_800_000, historical_failure_rate=0.55,
         required_skills=["finance", "compliance", "backend", "data"]),
    dict(sector="Travel & Hospitality", demand_index=0.60, growth_rate=0.10, competition_density=0.65,
         avg_capital_required=1_400_000, historical_failure_rate=0.60,
         required_skills=["operations", "hospitality", "marketing", "sales"]),
    dict(sector="PropTech/RealEstate",  demand_index=0.57, growth_rate=0.13, competition_density=0.55,
         avg_capital_required=3_200_000, historical_failure_rate=0.52,
         required_skills=["real_estate", "sales", "operations", "finance"]),
    dict(sector="HRTech",         demand_index=0.55, growth_rate=0.16, competition_density=0.48,
         avg_capital_required=1_300_000, historical_failure_rate=0.54,
         required_skills=["hr", "recruiting", "backend", "data"]),
    dict(sector="D2C Consumer Brands", demand_index=0.65, growth_rate=0.14, competition_density=0.82,
         avg_capital_required=1_700_000, historical_failure_rate=0.63,
         required_skills=["marketing", "operations", "logistics", "design"]),
]).set_index("sector")

STATE_TABLE = pd.DataFrame([
    dict(state="Karnataka",       ecosystem_index=0.93, tier="Best Performer",   cost_of_living_index=0.80),
    dict(state="Telangana",       ecosystem_index=0.88, tier="Best Performer",   cost_of_living_index=0.68),
    dict(state="Maharashtra",     ecosystem_index=0.85, tier="Top Performer",    cost_of_living_index=0.82),
    dict(state="Tamil Nadu",      ecosystem_index=0.80, tier="Top Performer",    cost_of_living_index=0.62),
    dict(state="Delhi NCR",       ecosystem_index=0.90, tier="Best Performer",   cost_of_living_index=0.85),
    dict(state="Gujarat",         ecosystem_index=0.76, tier="Leader",           cost_of_living_index=0.55),
    dict(state="Kerala",          ecosystem_index=0.70, tier="Leader",           cost_of_living_index=0.50),
    dict(state="Uttar Pradesh",   ecosystem_index=0.60, tier="Aspiring Leader",  cost_of_living_index=0.40),
    dict(state="West Bengal",     ecosystem_index=0.58, tier="Aspiring Leader",  cost_of_living_index=0.45),
    dict(state="Rajasthan",       ecosystem_index=0.55, tier="Emerging",         cost_of_living_index=0.38),
]).set_index("state")


# ---------------------------------------------------------------------------
# Location handling (this is new -- earlier version basically ignored the
# founder's location once it was collected, which reviewer #2 correctly
# called a loose end. "Location" here means: which region the founder is
# based in, used to compute a Location Affinity score between the founder
# and each candidate opportunity's state. We went with "location influences
# but does not force" the recommendation (a founder in Hyderabad CAN still
# get pointed at a Karnataka opportunity if it's a much better fit -- it
# just won't get the same affinity boost as a Telangana opportunity would).
# ---------------------------------------------------------------------------

# crude region grouping, just enough to give "same region" partial credit
REGION_OF_STATE = {
    "Karnataka": "South", "Telangana": "South", "Tamil Nadu": "South", "Kerala": "South",
    "Maharashtra": "West", "Gujarat": "West",
    "Delhi NCR": "North", "Uttar Pradesh": "North", "Rajasthan": "North",
    "West Bengal": "East",
}

# small city -> state lookup so a founder can type a city name (like the
# examples do -- "Hyderabad", "Jaipur") instead of having to know the exact
# state label used in STATE_TABLE
CITY_TO_STATE = {
    "hyderabad": "Telangana", "bengaluru": "Karnataka", "bangalore": "Karnataka",
    "mumbai": "Maharashtra", "pune": "Maharashtra", "chennai": "Tamil Nadu",
    "delhi": "Delhi NCR", "new delhi": "Delhi NCR", "gurugram": "Delhi NCR",
    "gurgaon": "Delhi NCR", "noida": "Delhi NCR", "ahmedabad": "Gujarat",
    "surat": "Gujarat", "kochi": "Kerala", "thiruvananthapuram": "Kerala",
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "varanasi": "Uttar Pradesh",
    "kolkata": "West Bengal", "jaipur": "Rajasthan", "udaipur": "Rajasthan",
    "jodhpur": "Rajasthan",
}


def resolve_founder_state(location_raw: str):
    """Tries to map whatever the founder typed to one of the states in
    STATE_TABLE. Returns None if we can't figure it out -- in that case the
    location affinity score just falls back to neutral for every opportunity
    instead of guessing wrong."""
    if not location_raw:
        return None
    loc = location_raw.strip().lower()

    # direct state name match (or the state name is contained in what they typed)
    for state_name in STATE_TABLE.index:
        if state_name.lower() in loc or loc in state_name.lower():
            return state_name

    # fall back to the small city dictionary
    for city, state_name in CITY_TO_STATE.items():
        if city in loc:
            return state_name

    return None


def location_affinity(founder_state, opportunity_state) -> float:
    """
    Exact state match          -> 1.0  (founder is right there)
    Same region, diff state    -> 0.65 (reasonably close, same broader ecosystem)
    Different region           -> 0.35 (still possible, remote/relocatable venture)
    Founder state unknown      -> 0.5  (neutral -- we genuinely don't know, so we
                                          don't want to penalize OR reward)
    """
    if founder_state is None:
        return 0.5
    if founder_state == opportunity_state:
        return 1.0
    if REGION_OF_STATE.get(founder_state) == REGION_OF_STATE.get(opportunity_state):
        return 0.65
    return 0.35


def load_sector_data():
    return SECTOR_TABLE.copy()


def load_state_data():
    return STATE_TABLE.copy()


# FIX #3 (reproducibility bug): the old version used Python's built-in hash()
# on the sector name to seed the random number generator. hash() on strings is
# randomized per process run (PYTHONHASHSEED), so re-running the same demo
# could quietly give a different WLS trend line each time. We replace it with
# a plain deterministic mapping (sum of character codes), so the same sector
# name always gives the same seed, no matter how many times you run the file.
def _deterministic_seed_for_sector(sector: str) -> int:
    return RANDOM_SEED + sum(ord(ch) for ch in sector)


def historical_demand_series(sector: str, periods: int = 24) -> pd.DataFrame:
    """Fake but internally-consistent monthly demand history for WLS to fit."""
    if sector not in SECTOR_TABLE.index:
        raise KeyError(f"Unknown sector '{sector}'. Known: {list(SECTOR_TABLE.index)}")

    seed = _deterministic_seed_for_sector(sector)
    rng = np.random.default_rng(seed)

    row = SECTOR_TABLE.loc[sector]
    monthly_growth = row["growth_rate"] / 12.0
    end_value = row["demand_index"]

    values = np.empty(periods)
    values[-1] = end_value
    for i in range(periods - 2, -1, -1):
        values[i] = values[i + 1] / (1 + monthly_growth)
    noise = rng.normal(0, 0.02, size=periods)
    values = np.clip(values + noise, 0.01, 1.0)

    recency_weight = np.linspace(0.4, 1.0, periods)   # trust recent months more
    return pd.DataFrame({"period": np.arange(periods), "demand": values, "weight": recency_weight})


# =============================================================================
# MODULE 1 - Founder Profile (feature engineering)
# =============================================================================

def _resolve_scalar(value, label_map, default):
    if isinstance(value, (int, float)):
        return float(np.clip(value, 0.0, 1.0))
    if isinstance(value, str):
        key = value.strip().lower()
        if key in label_map:
            return label_map[key]
    return default


@dataclass
class FounderProfile:
    skills: list
    budget_inr: float
    location: str
    interests: list
    risk_tolerance: float          # 0-1, how much risk THIS FOUNDER is willing to carry
    time_availability: float       # 0-1
    unmatched_skills: list = field(default_factory=list)
    unmatched_interests: list = field(default_factory=list)

    @staticmethod
    def from_raw(skills, budget_inr, location, interests, risk_tolerance, time_availability):
        if budget_inr is None or budget_inr <= 0:
            raise ValueError("budget_inr must be a positive number.")

        # FIX (user feedback): matching used to be exact-string-only, so
        # anything that wasn't spelled exactly like the taxonomy silently
        # got dropped. Now goes through the shared 3-stage matcher (exact ->
        # synonym -> fuzzy spelling) so real-world phrasing actually maps.
        matched_skills, unmatched_skills = _match_free_text_tokens(skills, SKILL_TAXONOMY, SKILL_SYNONYMS)
        matched_interests, unmatched_interests = _match_free_text_tokens(interests, SECTOR_TAXONOMY, SECTOR_SYNONYMS)

        risk_val = _resolve_scalar(risk_tolerance, RISK_LABEL_MAP, default=0.5)
        time_val = _resolve_scalar(time_availability, TIME_LABEL_MAP, default=0.5)

        return FounderProfile(matched_skills, float(budget_inr), location.strip(), matched_interests,
                               risk_val, time_val, unmatched_skills, unmatched_interests)

    def skill_vector(self):
        return np.array([1.0 if s in self.skills else 0.0 for s in SKILL_TAXONOMY])

    def interest_vector(self):
        return np.array([1.0 if s in self.interests else 0.0 for s in SECTOR_TAXONOMY])

    def budget_score(self, min_budget=50_000, max_budget=50_000_000):
        b = np.clip(self.budget_inr, min_budget, max_budget)
        log_b, log_min, log_max = np.log10(b), np.log10(min_budget), np.log10(max_budget)
        return float((log_b - log_min) / (log_max - log_min))

    def to_vector(self):
        """
        [skills(len(SKILL_TAXONOMY))] + [interests(len(SECTOR_TAXONOMY))] +
        [budget_score, risk_tolerance, time_availability]

        NOTE (fix for the "cosine risk dimension" issue raised in review):
        the 3rd extra dimension here is the founder's RISK TOLERANCE - how much
        risk they are personally willing/able to absorb. The matching
        opportunity vector below uses "required_risk_tolerance" for the SAME
        slot, i.e. how much risk-taking capacity this opportunity demands from
        whoever runs it. Both numbers now literally mean "risk-bearing
        capacity" on the same 0-1 scale, so a high value in both dimensions is
        a genuine match, not just two unrelated big numbers coincidentally
        multiplying to a big number in the cosine dot product.
        """
        return np.concatenate([
            self.skill_vector(),
            self.interest_vector(),
            np.array([self.budget_score(), self.risk_tolerance, self.time_availability]),
        ])

    def warnings(self):
        msgs = []
        if self.unmatched_skills:
            msgs.append(f"Skills not in taxonomy (ignored in scoring): {self.unmatched_skills}")
        if self.unmatched_interests:
            msgs.append(f"Interests not in sector taxonomy (ignored in scoring): {self.unmatched_interests}")
        if not self.skills:
            msgs.append("No recognized skills given -- founder-fit will rely mostly on interests/budget.")
        return msgs


# =============================================================================
# MODULE 2 - Market & Opportunity Engine
# =============================================================================

@dataclass(frozen=True)
class Opportunity:
    opportunity_id: str
    sector: str
    state: str
    demand_index: float
    growth_rate: float
    competition_density: float
    avg_capital_required: float
    historical_failure_rate: float       # also used as "required_risk_tolerance", see note above
    ecosystem_index: float
    cost_of_living_index: float
    required_skills: tuple

    def adjusted_capital_required(self):
        return self.avg_capital_required * (0.7 + 0.6 * self.cost_of_living_index)


class MarketOpportunityEngine:
    def __init__(self):
        self.sector_df = load_sector_data()
        self.state_df = load_state_data()

    def all_opportunities(self):
        opportunities = []
        for sector, srow in self.sector_df.iterrows():
            for state, strow in self.state_df.iterrows():
                opportunities.append(Opportunity(
                    opportunity_id=f"{sector}::{state}",
                    sector=sector, state=state,
                    demand_index=float(srow["demand_index"]),
                    growth_rate=float(srow["growth_rate"]),
                    competition_density=float(min(1.0, srow["competition_density"] * (0.6 + 0.5 * strow["ecosystem_index"]))),
                    avg_capital_required=float(srow["avg_capital_required"]),
                    historical_failure_rate=float(srow["historical_failure_rate"]),
                    ecosystem_index=float(strow["ecosystem_index"]),
                    cost_of_living_index=float(strow["cost_of_living_index"]),
                    required_skills=tuple(srow["required_skills"]),
                ))
        return opportunities

    def opportunities_for_sectors(self, sectors):
        if not sectors:
            return self.all_opportunities()
        wanted = set(sectors)
        return [o for o in self.all_opportunities() if o.sector in wanted]


# =============================================================================
# MODULE 3 - Opportunity Matching (cosine similarity + SVM feasibility)
# =============================================================================

def opportunity_requirement_vector(opp: Opportunity, min_budget=50_000, max_budget=50_000_000):
    skill_vec = np.array([1.0 if s in opp.required_skills else 0.0 for s in SKILL_TAXONOMY])
    interest_vec = np.array([1.0 if s == opp.sector else 0.0 for s in SECTOR_TAXONOMY])

    cap = np.clip(opp.adjusted_capital_required(), min_budget, max_budget)
    log_c, log_min, log_max = np.log10(cap), np.log10(min_budget), np.log10(max_budget)
    capital_score = float((log_c - log_min) / (log_max - log_min))

    # required_risk_tolerance: how much risk this opportunity asks the founder to
    # absorb -- same "risk capacity" axis as founder.risk_tolerance (see comment
    # in FounderProfile.to_vector). Using historical_failure_rate here is the
    # modeling choice: a sector that fails more often demands more risk
    # tolerance from whoever attempts it.
    required_risk_tolerance = float(opp.historical_failure_rate)
    commitment_level = 1.0

    return np.concatenate([skill_vec, interest_vec,
                            np.array([capital_score, required_risk_tolerance, commitment_level])])


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _rule_based_feasibility_label(budget, capital_required, skill_overlap, risk_tolerance, failure_rate, rng):
    """This is the hand-written rule that generates TRAINING LABELS for the SVM.
    Important: the SVM below is only as good as this rule -- it is NOT trained
    on real startup outcomes, so its output should be read as 'matches the
    pattern of a benchmark feasibility rule', not 'predicts real success'."""
    capital_ratio = budget / max(capital_required, 1.0)
    score = 0.0
    score += 1.4 if capital_ratio >= 0.6 else (0.4 if capital_ratio >= 0.3 else -0.8)
    score += 0.9 if skill_overlap >= 2 else (0.2 if skill_overlap == 1 else -0.7)
    score += 0.6 if risk_tolerance >= failure_rate - 0.15 else -0.6
    score += rng.normal(0, 0.35)          # label noise so SVM can't just re-derive the exact rule
    return int(score > 0.35)


def make_svm_training_data(n_samples=1200, seed=7):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_samples):
        budget = 10 ** rng.uniform(4.7, 7.3)
        capital_required = 10 ** rng.uniform(5.0, 7.0)
        skill_overlap = rng.integers(0, 5)
        risk_tolerance = rng.uniform(0.0, 1.0)
        failure_rate = rng.uniform(0.4, 0.7)

        cap = np.clip(capital_required, 50_000, 50_000_000)
        log_c = (np.log10(cap) - np.log10(50_000)) / (np.log10(50_000_000) - np.log10(50_000))
        b = np.clip(budget, 50_000, 50_000_000)
        log_b = (np.log10(b) - np.log10(50_000)) / (np.log10(50_000_000) - np.log10(50_000))

        features = [log_b, log_c, skill_overlap / 4.0, risk_tolerance, failure_rate]
        label = _rule_based_feasibility_label(budget, capital_required, skill_overlap,
                                               risk_tolerance, failure_rate, rng)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


@dataclass
class FeasibilityModel:
    scaler: StandardScaler
    svm: SVC
    train_accuracy: float
    test_accuracy: float          # FIX #5: held-out accuracy, not just train accuracy
    cv_accuracy_mean: float       # FIX #5: 5-fold cross validation mean accuracy
    cv_accuracy_std: float

    @staticmethod
    def train(seed=7):
        X, y = make_svm_training_data(seed=seed)

        # FIX #5: train/test split so we can report a number that at least
        # measures generalization ON THE SYNTHETIC TASK (still not real-world
        # validation, but much better than quoting training accuracy alone).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        svm = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=seed)
        svm.fit(X_train_s, y_train)

        train_acc = float(svm.score(X_train_s, y_train))
        test_acc = float(svm.score(X_test_s, y_test))

        # FIX (data-leakage bug caught in review): the CV step used to scale
        # the ENTIRE dataset once with a scaler that had already seen all of
        # it, and only then split into folds -- so every fold's "held out"
        # data had already influenced the mean/variance used to scale it.
        # That leaks a little bit of test information into training and
        # makes the CV score mildly optimistic. Fixed by wrapping
        # scaler+model in a Pipeline and handing cross_val_score the RAW,
        # unscaled data -- sklearn then refits the scaler independently
        # inside each fold, on only that fold's training portion.
        cv_pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=seed),
        )
        cv_scores = cross_val_score(cv_pipeline, X, y, cv=5)

        return FeasibilityModel(scaler=scaler, svm=svm, train_accuracy=train_acc,
                                 test_accuracy=test_acc,
                                 cv_accuracy_mean=float(cv_scores.mean()),
                                 cv_accuracy_std=float(cv_scores.std()))

    def predict_proba(self, founder: FounderProfile, opp: Opportunity):
        skill_overlap = len(set(founder.skills) & set(opp.required_skills))
        capital_required = opp.adjusted_capital_required()

        b = np.clip(founder.budget_inr, 50_000, 50_000_000)
        log_b = (np.log10(b) - np.log10(50_000)) / (np.log10(50_000_000) - np.log10(50_000))
        cap = np.clip(capital_required, 50_000, 50_000_000)
        log_c = (np.log10(cap) - np.log10(50_000)) / (np.log10(50_000_000) - np.log10(50_000))

        features = np.array([[log_b, log_c, skill_overlap / 4.0,
                               founder.risk_tolerance, opp.historical_failure_rate]])
        Xs = self.scaler.transform(features)
        return float(self.svm.predict_proba(Xs)[0, 1])


@dataclass
class MatchResult:
    opportunity: Opportunity
    similarity: float
    feasibility_probability: float
    skill_overlap: int


class OpportunityMatcher:
    def __init__(self, feasibility_model=None):
        self.feasibility_model = feasibility_model or FeasibilityModel.train()

    def match(self, founder: FounderProfile, opportunities):
        f_vec = founder.to_vector()
        results = []
        for opp in opportunities:
            o_vec = opportunity_requirement_vector(opp)
            sim = cosine_similarity(f_vec, o_vec)
            proba = self.feasibility_model.predict_proba(founder, opp)
            overlap = len(set(founder.skills) & set(opp.required_skills))
            results.append(MatchResult(opp, sim, proba, overlap))
        results.sort(key=lambda r: (r.similarity * 0.5 + r.feasibility_probability * 0.5), reverse=True)
        return results


# =============================================================================
# MODULE 4 - Opportunity Scoring
# =============================================================================

# NOTE: added "location" as its own scored component. Earlier version asked
# the founder for a location and then basically never used it again after
# Module 1 -- that was flagged as a real loose end. Weights below were
# rebalanced to make room for it (still sums to 1.0, checked below).
DEFAULT_WEIGHTS = {
    "demand": 0.20, "competition": 0.13, "founder_fit": 0.22,
    "capital_feasibility": 0.17, "risk": 0.13, "location": 0.15,
}


def _validate_weights(weights):
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Scoring weights must sum to a positive number.")
    if not np.isclose(total, 1.0, atol=1e-6):
        weights = {k: v / total for k, v in weights.items()}
    return weights


@dataclass
class ScoredOpportunity:
    match: MatchResult
    components: dict = field(default_factory=dict)
    score: float = 0.0

    @property
    def opportunity(self):
        return self.match.opportunity


class OpportunityScorer:
    def __init__(self, weights=None):
        self.weights = _validate_weights(dict(weights or DEFAULT_WEIGHTS))

    def score(self, match_results, founder_state=None):
        scored = []
        for m in match_results:
            opp = m.opportunity
            demand_component = float(np.clip(0.7 * opp.demand_index + 0.3 * min(opp.growth_rate / 0.3, 1.0), 0, 1))
            competition_component = float(1.0 - opp.competition_density)
            founder_fit_component = float(np.clip((m.similarity + 1) / 2, 0, 1))
            capital_component = float(m.feasibility_probability)
            risk_component = float(1.0 - opp.historical_failure_rate)
            location_component = location_affinity(founder_state, opp.state)

            components = {"demand": demand_component, "competition": competition_component,
                          "founder_fit": founder_fit_component, "capital_feasibility": capital_component,
                          "risk": risk_component, "location": location_component}
            total = sum(self.weights[k] * v for k, v in components.items())
            scored.append(ScoredOpportunity(m, components, float(np.clip(total, 0, 1))))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored


# =============================================================================
# MODULE 5 - Resource Optimizer
# LP feasibility check -> QP allocation -> KKT verification
#
# FIX #6 (naming problem raised in review): the earlier draft used a SQUARE
# ROOT utility (sum w_i * sqrt(x_i)) and called the solve step "QP", which is
# wrong -- sqrt(x) is not quadratic. Here the utility is rewritten as an
# actually-quadratic function:
#
#       U(x_i) = w_i * x_i  -  0.5 * c_i * x_i^2         (concave parabola)
#
# so the objective sum_i U(x_i) really is quadratic in x, subject to linear
# constraints (sum x_i = budget, x_i >= min_i). That makes this a genuine QP.
# =============================================================================

CATEGORIES = ["Product Development", "Marketing & Sales", "Operations",
              "Legal & Compliance", "Contingency Reserve"]

MIN_FRACTIONS_BASE = {
    "Product Development": 0.15, "Marketing & Sales": 0.08, "Operations": 0.10,
    "Legal & Compliance": 0.03, "Contingency Reserve": 0.05,
}

# fixed rupee floors so the LP feasibility check can genuinely fail for a tiny
# budget (fractions of ANY budget always sum below 1, so fraction floors alone
# can never be infeasible -- that was a bug in an earlier draft)
MIN_ABSOLUTE_BASE_INR = {
    "Product Development": 40_000, "Marketing & Sales": 20_000, "Operations": 25_000,
    "Legal & Compliance": 15_000, "Contingency Reserve": 15_000,
}

KKT_TOLERANCE = 1e-3


def _category_weights(opp: Opportunity):
    compliance_sensitive = {"finance", "compliance", "healthcare", "policy", "security", "legal"}
    compliance_load = len(set(opp.required_skills) & compliance_sensitive) / max(len(opp.required_skills), 1)
    return {
        "Product Development": 1.0 + 0.5 * opp.growth_rate * 4,
        "Marketing & Sales": 0.6 + 0.9 * opp.competition_density,
        "Operations": 0.7 + 0.3 * opp.cost_of_living_index,
        "Legal & Compliance": 0.4 + 1.4 * compliance_load,
        "Contingency Reserve": 0.5 + 0.8 * opp.historical_failure_rate,
    }


def _min_fractions(opp: Opportunity):
    fractions = dict(MIN_FRACTIONS_BASE)
    compliance_sensitive = {"finance", "compliance", "healthcare", "policy", "legal"}
    if set(opp.required_skills) & compliance_sensitive:
        fractions["Legal & Compliance"] = max(fractions["Legal & Compliance"], 0.08)
    if opp.historical_failure_rate > 0.6:
        fractions["Contingency Reserve"] = max(fractions["Contingency Reserve"], 0.10)
    return fractions


def _min_amounts(opp: Opportunity, budget: float):
    fractions = _min_fractions(opp)
    compliance_sensitive = {"finance", "compliance", "healthcare", "policy", "legal"}
    col_scale = 0.7 + 0.6 * opp.cost_of_living_index
    amounts = []
    for c in CATEGORIES:
        frac_floor = fractions[c] * budget
        abs_floor = MIN_ABSOLUTE_BASE_INR[c] * col_scale
        if c == "Legal & Compliance" and set(opp.required_skills) & compliance_sensitive:
            abs_floor *= 1.5
        amounts.append(max(frac_floor, abs_floor))
    return np.array(amounts)


@dataclass
class ResourcePlan:
    budget: float
    allocations: dict
    fractions: dict
    weights: dict
    curvature: dict          # the c_i values of the quadratic utility, for transparency
    minimums_scaled: bool
    lp_status: str
    qp_status: str
    kkt_satisfied: bool
    kkt_report: dict


class ResourceOptimizer:
    def solve(self, budget: float, opp: Opportunity) -> ResourcePlan:
        if budget <= 0:
            raise ValueError("Budget must be positive.")

        weights_dict = _category_weights(opp)
        w = np.array([weights_dict[c] for c in CATEGORIES])
        min_amounts = _min_amounts(opp, budget)

        lp_status, min_amounts, minimums_scaled = self._lp_feasibility(budget, min_amounts)

        # curvature c_i chosen so marginal utility w_i - c_i*x_i hits 0 exactly
        # at x_i = budget (i.e. a category can't usefully absorb the ENTIRE
        # budget on its own -- diminishing returns kick in well before that)
        c = w / budget

        x_opt, qp_status = self._qp_allocate(budget, min_amounts, w, c)

        kkt_ok, kkt_report = self._verify_kkt(x_opt, min_amounts, w, c, budget)
        if not kkt_ok:
            raise RuntimeError(f"Resource allocation failed KKT verification: {kkt_report}")

        allocations = {cat: float(x_opt[i]) for i, cat in enumerate(CATEGORIES)}
        fractions = {cat: float(x_opt[i] / budget) for i, cat in enumerate(CATEGORIES)}
        curvature = {cat: float(c[i]) for i, cat in enumerate(CATEGORIES)}

        return ResourcePlan(float(budget), allocations, fractions, weights_dict, curvature,
                             minimums_scaled, lp_status, qp_status, kkt_ok, kkt_report)

    def _lp_feasibility(self, budget, min_amounts):
        """Pure feasibility LP: does a point exist with x_i >= min_i, sum(x_i) = budget?"""
        n = len(min_amounts)
        c_obj = np.zeros(n)               # no preference, just checking feasibility
        A_eq = np.ones((1, n))
        b_eq = np.array([budget])
        bounds = [(m, None) for m in min_amounts]

        res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success:
            return "feasible", min_amounts, False

        # budget too small for the declared minimums -> scale them down and say so
        total_min = min_amounts.sum()
        scaled = min_amounts * (budget / total_min) * 0.98
        return "infeasible_at_declared_minimums_scaled_down", scaled, True

    def _qp_allocate(self, budget, min_amounts, w, c):
        """
        Solves  maximize sum_i (w_i*x_i - 0.5*c_i*x_i^2)  s.t. sum x_i = budget, min_i <= x_i <= budget

        This is solved with a Lagrangian bisection ("water-filling") instead of
        a generic nonlinear solver. From the KKT stationarity condition,
        w_i - c_i*x_i = lambda for interior x_i, which rearranges to
            x_i(lambda) = clip( (w_i - lambda) / c_i , min_i, budget )
        sum_i x_i(lambda) is monotonically non-increasing in lambda (each term is
        non-increasing and clipping preserves monotonicity), so we can just
        bisection-search for the lambda that makes the categories sum to the
        budget exactly. This is guaranteed to converge for a convex QP with a
        single linear equality constraint and box bounds -- no line-search
        failures like a general-purpose solver can hit.
        """
        def x_of_lambda(lam):
            raw = (w - lam) / c
            return np.clip(raw, min_amounts, budget)

        lo = float(np.min(w - c * budget) - 10.0)
        hi = float(np.max(w - c * min_amounts) + 10.0)
        for _ in range(200):
            mid = (lo + hi) / 2
            total = x_of_lambda(mid).sum()
            if total > budget:
                lo = mid          # sum too high -> need higher lambda to push x down
            else:
                hi = mid
        lam = (lo + hi) / 2

        x = x_of_lambda(lam)
        x = np.maximum(x, min_amounts)
        x = x * (budget / x.sum())        # clean up any last floating-point residual
        return x, "converged (Lagrangian bisection)"

    def _verify_kkt(self, x, min_amounts, w, c, budget):
        """
        For this QP the Lagrangian gives, at the optimum:
            interior category (x_i > min_i) : w_i - c_i*x_i = lambda
            bound category    (x_i = min_i) : w_i - c_i*x_i <= lambda   (mu_i >= 0)
        We check both conditions numerically instead of just trusting scipy.
        """
        report = {}
        primal_feasible = bool(np.all(x >= min_amounts - 1e-6) and abs(x.sum() - budget) < 1e-3)
        report["primal_feasible"] = primal_feasible

        marginal_utility = w - c * x
        at_bound = x <= (min_amounts + 1e-4)

        interior_mu = marginal_utility[~at_bound]
        if len(interior_mu) > 0:
            lam = float(np.mean(interior_mu))
            stationarity_ok = bool(np.all(np.abs(interior_mu - lam) < 1e-2 * max(abs(lam), 1.0)))
        else:
            lam = float(np.max(marginal_utility))
            stationarity_ok = True

        bound_mu = marginal_utility[at_bound]
        dual_feasible = bool(np.all(bound_mu <= lam + KKT_TOLERANCE * max(abs(lam), 1.0)))

        report.update({
            "lambda": lam,
            "stationarity_interior_categories_equal_marginal_utility": stationarity_ok,
            "dual_feasible_bound_categories_marginal_utility_leq_lambda": dual_feasible,
            "complementary_slackness": True,
        })
        return (primal_feasible and stationarity_ok and dual_feasible), report


# =============================================================================
# MODULE 6 - Market Trend Engine
# Weighted Least Squares trend + Multivariate Gaussian "market profile deviation"
# =============================================================================

@dataclass
class TrendForecast:
    sector: str
    slope_per_month: float
    intercept: float
    r_squared: float
    forecast_next_periods: np.ndarray


def weighted_least_squares_trend(sector, forecast_horizon=6) -> TrendForecast:
    hist = historical_demand_series(sector)
    t = hist["period"].to_numpy(dtype=float)
    y = hist["demand"].to_numpy(dtype=float)
    w = hist["weight"].to_numpy(dtype=float)

    # weighted normal equations for y = a*t + b
    X = np.column_stack([t, np.ones_like(t)])
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    slope, intercept = beta

    y_pred = X @ beta
    ss_res = np.sum(w * (y - y_pred) ** 2)
    y_bar = np.sum(w * y) / np.sum(w)
    ss_tot = np.sum(w * (y - y_bar) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    future_t = np.arange(t[-1] + 1, t[-1] + 1 + forecast_horizon)
    forecast = np.clip(slope * future_t + intercept, 0.0, 1.0)

    return TrendForecast(sector, float(slope), float(intercept), r_squared, forecast)


@dataclass
class RiskAssessment:
    mahalanobis_distance: float
    risk_score: float


class MarketProfileDeviationModel:
    """
    FIX #9 (interpretation problem raised in review): this used to be called
    a model of "healthy opportunities" and described the fitted mean as the
    healthy center. That's overclaiming -- all we actually computed is the
    MEAN OF ALL OPPORTUNITIES IN OUR TABLE, not a labelled "healthy" cluster
    (we don't have outcome labels to define "healthy" from data). Renamed
    to reflect what it actually measures: how far a specific opportunity's
    numbers sit from the average opportunity in the dataset, weighted toward
    the unfavorable direction only (below-average demand/growth/ecosystem, or
    above-average competition).
    """
    FEATURES = ["demand_index", "growth_rate", "inv_competition", "ecosystem_index"]

    def __init__(self, opportunities):
        if len(opportunities) < 5:
            raise ValueError("Need at least 5 opportunities to fit a stable covariance matrix.")
        X = np.array([[o.demand_index, o.growth_rate, 1 - o.competition_density, o.ecosystem_index]
                       for o in opportunities])
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6   # ridge for invertibility
        self.cov_inv = np.linalg.inv(cov)

    def score(self, opp: Opportunity) -> RiskAssessment:
        x = np.array([opp.demand_index, opp.growth_rate, 1 - opp.competition_density, opp.ecosystem_index])
        diff = x - self.mean
        d2 = float(diff @ self.cov_inv @ diff.T)
        d = float(np.sqrt(max(d2, 0.0)))

        unhealthy_direction = float(np.sum(np.clip(-diff, 0, None)))
        directional_penalty = d * (0.4 + 0.6 * min(unhealthy_direction / (abs(diff).sum() + 1e-9), 1.0))
        risk_score = float(1 / (1 + np.exp(-(directional_penalty - 2.0))))
        return RiskAssessment(d, risk_score)


# =============================================================================
# MODULE 7 - Business Evolution: Markov Chain + Markov Decision Process
# =============================================================================

STATES = ["IDEA", "VALIDATION", "EARLY_TRACTION", "GROWTH", "DECLINE", "FAILURE", "SUCCESS"]
TRANSIENT = STATES[:5]
ABSORBING = STATES[5:]
IDX = {s: i for i, s in enumerate(STATES)}

_BASE_MATRIX = {
    "IDEA":           {"VALIDATION": 0.70, "FAILURE": 0.30},
    "VALIDATION":     {"EARLY_TRACTION": 0.55, "DECLINE": 0.25, "FAILURE": 0.20},
    "EARLY_TRACTION": {"GROWTH": 0.50, "DECLINE": 0.30, "FAILURE": 0.20},
    "GROWTH":         {"SUCCESS": 0.55, "DECLINE": 0.30, "FAILURE": 0.15},
    "DECLINE":        {"EARLY_TRACTION": 0.30, "DECLINE": 0.20, "FAILURE": 0.50},
}
_FAVORABLE_TARGETS = {"VALIDATION", "EARLY_TRACTION", "GROWTH", "SUCCESS"}
_UNFAVORABLE_TARGETS = {"DECLINE", "FAILURE"}


def _build_full_matrix(risk_score, momentum):
    momentum_clipped = float(np.clip(momentum, -0.02, 0.02))
    risk_shift = float(np.clip(risk_score - 0.5, -0.5, 0.5))
    momentum_shift = momentum_clipped / 0.02 * 0.5
    net_pull = 0.5 * (-risk_shift) + 0.5 * momentum_shift

    M = np.zeros((len(STATES), len(STATES)))
    for absorbing in ABSORBING:
        M[IDX[absorbing], IDX[absorbing]] = 1.0

    for src, targets in _BASE_MATRIX.items():
        weights = dict(targets)
        for tgt in list(weights.keys()):
            if tgt in _FAVORABLE_TARGETS:
                weights[tgt] *= (1.0 + 0.6 * net_pull)
            elif tgt in _UNFAVORABLE_TARGETS:
                weights[tgt] *= (1.0 - 0.6 * net_pull)
            weights[tgt] = max(weights[tgt], 1e-4)
        total = sum(weights.values())
        for tgt, wt in weights.items():
            M[IDX[src], IDX[tgt]] = wt / total

    row_sums = M.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8), f"Rows must sum to 1: {row_sums}"
    return M


@dataclass
class MarkovChainResult:
    transition_matrix: np.ndarray
    prob_reach_success_from_idea: float
    prob_reach_failure_from_idea: float
    expected_steps_to_absorption_from_idea: float


def run_markov_chain(risk_score, momentum) -> MarkovChainResult:
    M = _build_full_matrix(risk_score, momentum)
    t_idx = [IDX[s] for s in TRANSIENT]
    a_idx = [IDX[s] for s in ABSORBING]

    Q = M[np.ix_(t_idx, t_idx)]
    R = M[np.ix_(t_idx, a_idx)]
    I = np.eye(len(t_idx))
    N = np.linalg.inv(I - Q)          # fundamental matrix
    B = N @ R                          # absorption probabilities
    expected_steps = N.sum(axis=1)

    idea_row = t_idx.index(IDX["IDEA"])
    prob_failure = float(B[idea_row, ABSORBING.index("FAILURE")])
    prob_success = float(B[idea_row, ABSORBING.index("SUCCESS")])

    return MarkovChainResult(M, prob_success, prob_failure, float(expected_steps[idea_row]))


ACTIONS = ["INVEST_MORE", "HOLD", "PIVOT", "EXIT"]
GAMMA = 0.9

_EXIT_SALVAGE = {"IDEA": -0.65, "VALIDATION": -0.45, "EARLY_TRACTION": -0.15, "GROWTH": 0.20, "DECLINE": -0.55}
_ACTION_COST = {"INVEST_MORE": -0.10, "HOLD": -0.03, "PIVOT": -0.08, "EXIT": 0.0}


def _reweight_row(row, state, boost_favorable):
    new_row = row.copy()
    for tgt in _BASE_MATRIX[state]:
        i = IDX[tgt]
        if tgt in _FAVORABLE_TARGETS:
            new_row[i] *= (1 + boost_favorable)
        elif tgt in _UNFAVORABLE_TARGETS:
            new_row[i] *= (1 - boost_favorable * 0.6)
        new_row[i] = max(new_row[i], 1e-4)
    return new_row / new_row.sum()


def _action_transition_row(state, action, base_M):
    row = base_M[IDX[state]].copy()
    if action == "HOLD":
        return row
    if action == "INVEST_MORE":
        return _reweight_row(row, state, boost_favorable=0.35)
    if action == "PIVOT":
        new_row = np.zeros_like(row)
        new_row[IDX["VALIDATION"]] = 0.70
        new_row[IDX["FAILURE"]] = 0.15
        new_row[IDX["DECLINE"]] = 0.15
        return new_row
    if action == "EXIT":
        new_row = np.zeros_like(row)
        new_row[IDX["FAILURE"]] = 1.0        # bookkeeping only; value comes from the salvage reward
        return new_row
    raise ValueError(f"Unknown action {action}")


@dataclass
class MDPResult:
    policy: dict
    values: dict
    q_values: dict


def solve_mdp(risk_score, momentum, max_iter=500, tol=1e-8) -> MDPResult:
    base_M = _build_full_matrix(risk_score, momentum)
    V = {s: 0.0 for s in STATES}
    V["FAILURE"], V["SUCCESS"] = -1.0, 1.0

    transition_cache = {(s, a): _action_transition_row(s, a, base_M) for s in TRANSIENT for a in ACTIONS}

    for _ in range(max_iter):
        delta = 0.0
        new_V = dict(V)
        for s in TRANSIENT:
            q_values = {}
            for a in ACTIONS:
                if a == "EXIT":
                    q_values[a] = _EXIT_SALVAGE[s]
                else:
                    row = transition_cache[(s, a)]
                    continuation = float(sum(row[IDX[s2]] * V[s2] for s2 in STATES))
                    q_values[a] = _ACTION_COST[a] + GAMMA * continuation
            best_a = max(q_values, key=q_values.get)
            new_V[s] = q_values[best_a]
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < tol:
            break

    policy, q_all = {}, {}
    for s in TRANSIENT:
        q_values = {}
        for a in ACTIONS:
            if a == "EXIT":
                q_values[a] = _EXIT_SALVAGE[s]
            else:
                row = transition_cache[(s, a)]
                continuation = float(sum(row[IDX[s2]] * V[s2] for s2 in STATES))
                q_values[a] = _ACTION_COST[a] + GAMMA * continuation
        best_a = max(q_values, key=q_values.get)
        policy[s] = best_a
        q_all[s] = q_values

    return MDPResult(policy, {s: V[s] for s in STATES}, q_all)


# =============================================================================
# MODULE 8 - Validation Engine: Monte Carlo market experiment + Go/Pivot/Stop
# =============================================================================

N_SIMULATIONS = 5000
COST_PER_VISITOR_BASE_INR = 12.0
TEST_RUNWAY_MONTHS = 6


@dataclass
class KPITargets:
    min_conversion_rate: float
    min_monthly_customers: float
    max_cac_inr: float


@dataclass
class ValidationResult:
    kpi_targets: KPITargets
    kpi_achievement_rate: float
    mean_monthly_customers: float
    mean_conversion_rate: float
    mean_cac_inr: float
    simulations: int


def _kpi_targets(cost_per_visitor, traffic_mean, min_conversion_rate, opp: Opportunity) -> KPITargets:
    """Targets derived from the SAME model used to simulate outcomes (see run_market_experiment):
    under this linear traffic model, CAC = cost_per_visitor / conversion_rate, so the max-CAC bar
    is set at exactly the CAC implied by hitting the minimum conversion rate, with slack."""
    capital_scale = float(np.clip(opp.avg_capital_required / 2_500_000, 0.5, 2.5))
    max_cac = float(np.clip((cost_per_visitor / min_conversion_rate) * 1.3 * capital_scale, 100, 5000))
    min_customers = float(max(15.0, 0.4 * traffic_mean * min_conversion_rate))
    return KPITargets(min_conversion_rate, min_customers, max_cac)


def run_market_experiment(opp: Opportunity, marketing_budget_total, founder_fit_component, seed=11) -> ValidationResult:
    rng = np.random.default_rng(seed)
    monthly_budget = max(marketing_budget_total / TEST_RUNWAY_MONTHS, 1.0)

    cost_per_visitor = COST_PER_VISITOR_BASE_INR * (0.6 + 0.9 * opp.competition_density)
    traffic_mean = max(monthly_budget / cost_per_visitor, 1.0)
    traffic = rng.poisson(lam=traffic_mean, size=N_SIMULATIONS)

    base_conv = 0.012 + 0.035 * opp.demand_index + 0.02 * founder_fit_component - 0.02 * opp.competition_density
    base_conv = float(np.clip(base_conv, 0.003, 0.09))
    concentration = 250.0
    a, b = base_conv * concentration, (1 - base_conv) * concentration
    conversion_rate = rng.beta(a, b, size=N_SIMULATIONS)

    customers = traffic * conversion_rate
    cac = monthly_budget / np.maximum(customers, 1e-6)

    targets = _kpi_targets(cost_per_visitor, traffic_mean, min_conversion_rate=0.02, opp=opp)
    success = ((conversion_rate >= targets.min_conversion_rate) &
               (customers >= targets.min_monthly_customers) &
               (cac <= targets.max_cac_inr))

    return ValidationResult(targets, float(np.mean(success)), float(np.mean(customers)),
                             float(np.mean(conversion_rate)), float(np.mean(cac)), N_SIMULATIONS)


@dataclass
class FinalRecommendation:
    decision: str
    composite_confidence: float
    rationale: list


# FIX #4 (biggest one): the old formula never looked at the Markov chain's
# P(SUCCESS) at all, so a low-probability-of-success opportunity could still
# get GO purely off the other four numbers. Markov success probability is now
# a genuine 5th input, and the weights are rebalanced to make room for it.
DECISION_WEIGHTS = {
    "opportunity_score": 0.22,
    "feasibility_probability": 0.18,
    "risk_inverse": 0.15,
    "kpi_achievement_rate": 0.20,
    "markov_success_probability": 0.25,
}
assert abs(sum(DECISION_WEIGHTS.values()) - 1.0) < 1e-9

GO_THRESHOLD = 0.58
PIVOT_THRESHOLD = 0.40

# GUARDRAIL (reviewer's point #7): even after Markov success probability got
# folded into the weighted composite above, a strong score on the OTHER four
# inputs can still average out to a passing composite even when the lifecycle
# model says the odds of ever reaching SUCCESS are very low. A weighted
# average can hide one bad number behind four good ones -- that's exactly
# the failure mode being pointed out. So this is a hard floor, checked
# BEFORE the composite decides anything: below it, GO is simply not an
# option, no matter how good everything else looks.
MARKOV_SUCCESS_GO_FLOOR = 0.15


def _composite_confidence(opportunity_score, feasibility_probability, risk_score,
                           kpi_achievement_rate, markov_success_probability):
    return (DECISION_WEIGHTS["opportunity_score"] * opportunity_score +
            DECISION_WEIGHTS["feasibility_probability"] * feasibility_probability +
            DECISION_WEIGHTS["risk_inverse"] * (1 - risk_score) +
            DECISION_WEIGHTS["kpi_achievement_rate"] * kpi_achievement_rate +
            DECISION_WEIGHTS["markov_success_probability"] * markov_success_probability)


def _percent(x):
    return f"{x * 100:.0f}%"


def _band(x, labels=("very low", "low", "moderate", "high", "very high")):
    """Turns a 0-1 number into a plain word so normal people don't have to
    read raw decimals -- the exact numbers are still shown alongside it."""
    if x < 0.2:
        return labels[0]
    if x < 0.4:
        return labels[1]
    if x < 0.6:
        return labels[2]
    if x < 0.8:
        return labels[3]
    return labels[4]


# plain-English translation of what the MDP action means for the founder --
# used ONLY in the final recommendation text; the technical stage-by-stage
# policy table is still printed earlier in the report for anyone who wants it
_ACTION_IN_PLAIN_ENGLISH = {
    "INVEST_MORE": "put in more time/money now",
    "HOLD": "keep going steadily without over-committing yet",
    "PIVOT": "change the approach before committing more resources",
    "EXIT": "not pursue this specific opportunity any further",
}


def decide(opportunity_score, feasibility_probability, risk_score, mdp_policy_at_idea,
           kpi_achievement_rate, markov_success_probability) -> FinalRecommendation:
    """
    IMPORTANT (user feedback): the detailed math (the actual Markov transition
    probabilities, the MDP policy table, the SVM accuracy numbers, etc.) is
    already printed further up in the report by print_summary(). This
    function's job is just to turn 5 numbers into ONE decision. So the
    rationale built here is written in plain English for someone who has
    never heard of a Markov chain or an SVM -- it explains WHAT the numbers
    mean for them, not WHICH algorithm produced them. Nothing about the
    actual decision math changed, only how it's explained.
    """
    rationale = []
    composite = _composite_confidence(opportunity_score, feasibility_probability, risk_score,
                                       kpi_achievement_rate, markov_success_probability)

    if mdp_policy_at_idea == "EXIT":
        rationale.append("Looking at how this kind of venture tends to play out stage by stage, "
                          "even accounting for the option to adjust course later, the numbers don't "
                          "support starting down this path at all.")
        return FinalRecommendation("STOP", float(composite), rationale)

    rationale.append(f"Overall opportunity strength is {_band(opportunity_score)} "
                      f"({_percent(opportunity_score)}) -- this blends market demand, competition, "
                      f"how well it fits your skills and location, affordability, and risk.")
    rationale.append(f"Your estimated ability to actually pull this off (skills + budget vs. what's "
                      f"needed) is {_band(feasibility_probability)} ({_percent(feasibility_probability)}).")
    rationale.append(f"Market risk for this opportunity is {_band(risk_score)} ({_percent(risk_score)}).")
    rationale.append(f"In a simulated early marketing test, this idea hit its customer/cost targets "
                      f"{_percent(kpi_achievement_rate)} of the time.")
    rationale.append(f"Looking at how similar ventures typically evolve, this path has roughly a "
                      f"{_percent(markov_success_probability)} chance of reaching a stable, successful "
                      f"stage, and about {_percent(1 - markov_success_probability)} chance of stalling "
                      f"out or failing along the way.")
    rationale.append(f"At this early stage, the recommended move is to "
                      f"{_ACTION_IN_PLAIN_ENGLISH[mdp_policy_at_idea]}.")

    # --- guardrail check happens BEFORE the normal threshold logic ---
    if markov_success_probability < MARKOV_SUCCESS_GO_FLOOR:
        rationale.append(f"Even though some of the other numbers above look decent, the long-run "
                          f"success chance ({_percent(markov_success_probability)}) is too low on its "
                          f"own to justify a full GO -- a few good numbers shouldn't be able to hide "
                          f"one that says this path usually doesn't work out.")
        if composite >= PIVOT_THRESHOLD:
            decision = "PIVOT"
            rationale.append("There's still enough potential here to be worth reworking the idea "
                              "(different niche, pricing, or target customer) rather than dropping it.")
        else:
            decision = "STOP"
            rationale.append("Combined with the other numbers also being weak, this specific "
                              "opportunity isn't worth pursuing right now.")
        return FinalRecommendation(decision, float(composite), rationale)

    if composite >= GO_THRESHOLD and mdp_policy_at_idea in ("INVEST_MORE", "HOLD"):
        decision = "GO"
        rationale.append("Taken together, the opportunity looks strong, feels achievable for you, "
                          "tested well in simulation, and has a reasonable shot at long-term success -- "
                          "this is worth pursuing.")
    elif composite >= PIVOT_THRESHOLD or mdp_policy_at_idea == "PIVOT":
        decision = "PIVOT"
        rationale.append("The picture is mixed -- there's real potential, but not enough certainty "
                          "to commit fully yet. Adjusting the approach first (rather than going all-in "
                          "or walking away) is the safer move.")
    else:
        decision = "STOP"
        rationale.append("Between the market conditions, the fit, and the simulated test results, "
                          "the numbers don't support investing further in this specific opportunity.")

    return FinalRecommendation(decision, float(composite), rationale)


# =============================================================================
# FINAL DECISION - orchestrates modules 1-8
# =============================================================================

@dataclass
class VentureMindReport:
    founder_warnings: list
    founder_location: dict
    top_opportunities: list
    chosen_opportunity: dict
    evidence: dict
    resource_plan: dict
    trend: dict
    risk: dict
    business_evolution: dict
    validation: dict
    final_decision: dict
    model_notes: dict     # honesty section: SVM train/test/cv accuracy etc.

    def print_summary(self):
        co = self.chosen_opportunity
        print("=" * 72)
        print("VENTUREMIND -- FINAL DECISION REPORT")
        print("=" * 72)
        loc = self.founder_location
        print(f"\nFounder location: '{loc['raw_input']}'  ->  resolved state: "
              f"{loc['resolved_state'] if loc['resolved_state'] else 'UNKNOWN (neutral affinity used)'}")
        if self.founder_warnings:
            print("\nFounder profile warnings:")
            for wmsg in self.founder_warnings:
                print(f"  - {wmsg}")

        print(f"\nRecommended Opportunity: {co['sector']} in {co['state']}")
        print(f"Opportunity Score: {self.evidence['opportunity_score']:.3f}")
        print("\nEvidence (score components):")
        for k, v in self.evidence["components"].items():
            print(f"  - {k:20s}: {v:.3f}")
        print(f"  - SVM feasibility probability : {self.evidence['feasibility_probability']:.3f}")
        print(f"  - Cosine founder-fit similarity: {self.evidence['raw_cosine_similarity']:.3f}")

        mn = self.model_notes
        print(f"\nSVM benchmark classifier honesty check (trained on SYNTHETIC rule-based labels, "
              f"not real outcomes):")
        print(f"  - Train accuracy      : {mn['svm_train_accuracy']:.3f}")
        print(f"  - Held-out test accuracy: {mn['svm_test_accuracy']:.3f}")
        print(f"  - 5-fold CV accuracy   : {mn['svm_cv_accuracy_mean']:.3f} (+/- {mn['svm_cv_accuracy_std']:.3f})")

        rp = self.resource_plan
        print(f"\nResource Plan (Budget: Rs.{rp['budget']:,.0f}) "
              f"[LP={rp['lp_status']}, QP={rp['qp_status']}, KKT satisfied={rp['kkt_satisfied']}]:")
        for cat, amt in rp["allocations"].items():
            print(f"  - {cat:22s}: Rs.{amt:>12,.0f}  ({rp['fractions'][cat]*100:5.1f}%)")

        tr = self.trend
        print(f"\nMarket Trend (WLS, deterministic seed): slope={tr['slope_per_month']:+.5f}/month, R^2={tr['r_squared']:.2f}")
        print(f"  6-month demand-index forecast: {['%.3f' % v for v in tr['forecast_next_periods']]}")

        rk = self.risk
        print(f"\nMarket Profile Deviation (Mahalanobis distance from average opportunity): "
              f"{rk['mahalanobis_distance']:.2f}  ->  risk_score={rk['risk_score']:.3f}")

        be = self.business_evolution
        print(f"\nBusiness Evolution (Markov Chain from IDEA):")
        print(f"  P(reach SUCCESS) = {be['markov']['prob_reach_success_from_idea']:.3f}   "
              f"P(reach FAILURE) = {be['markov']['prob_reach_failure_from_idea']:.3f}   "
              f"Expected steps to absorption = {be['markov']['expected_steps_to_absorption_from_idea']:.1f}")
        print("  MDP policy by stage:")
        for s, a in be["mdp"]["policy"].items():
            print(f"    - {s:16s} -> {a}")

        val = self.validation
        print(f"\nValidation (Monte Carlo market experiment, n={val['simulations']}):")
        print(f"  KPI achievement rate            : {val['kpi_achievement_rate']:.3f}")
        print(f"  Mean simulated monthly customers: {val['mean_monthly_customers']:.1f} "
              f"(target >= {val['kpi_targets']['min_monthly_customers']:.0f})")
        print(f"  Mean simulated conversion rate   : {val['mean_conversion_rate']*100:.2f}% "
              f"(target >= {val['kpi_targets']['min_conversion_rate']*100:.2f}%)")
        print(f"  Mean simulated CAC              : Rs.{val['mean_cac_inr']:,.0f} "
              f"(target <= Rs.{val['kpi_targets']['max_cac_inr']:,.0f})")

        fd = self.final_decision
        print(f"\n{'='*72}\nFINAL RECOMMENDATION: {fd['decision']}   "
              f"(overall confidence: {fd['composite_confidence']*100:.0f}%)")
        print("Why:")
        for r in fd["rationale"]:
            print(f"  - {r}")
        print("=" * 72)


class VentureMindPipeline:
    def __init__(self, scoring_weights=None, random_seed=7):
        self.market_engine = MarketOpportunityEngine()
        self.feasibility_model = FeasibilityModel.train(seed=random_seed)
        self.matcher = OpportunityMatcher(self.feasibility_model)
        self.scorer = OpportunityScorer(weights=scoring_weights)
        self.resource_optimizer = ResourceOptimizer()
        self.risk_model = MarketProfileDeviationModel(self.market_engine.all_opportunities())

    def run(self, skills, budget_inr, location, interests, risk_tolerance, time_availability,
            top_n_considered=10) -> VentureMindReport:

        # MODULE 1
        founder = FounderProfile.from_raw(skills, budget_inr, location, interests, risk_tolerance, time_availability)
        founder_state = resolve_founder_state(founder.location)
        warnings_list = founder.warnings()
        if founder_state is None:
            warnings_list.append(f"Could not match location '{founder.location}' to a known state -- "
                                  f"location affinity will be neutral (0.5) for every opportunity.")

        # MODULE 2
        candidate_opportunities = self.market_engine.opportunities_for_sectors(founder.interests)
        if not candidate_opportunities:
            candidate_opportunities = self.market_engine.all_opportunities()

        # MODULE 3
        matches = self.matcher.match(founder, candidate_opportunities)

        # MODULE 4
        scored = self.scorer.score(matches, founder_state)
        top = scored[0]
        opp = top.opportunity

        # MODULE 5
        resource_plan = self.resource_optimizer.solve(founder.budget_inr, opp)

        # MODULE 6
        trend = weighted_least_squares_trend(opp.sector)
        risk = self.risk_model.score(opp)

        # MODULE 7
        markov = run_markov_chain(risk.risk_score, trend.slope_per_month)
        mdp = solve_mdp(risk.risk_score, trend.slope_per_month)

        # MODULE 8
        validation = run_market_experiment(opp, resource_plan.allocations["Marketing & Sales"],
                                            top.components["founder_fit"])
        final = decide(
            opportunity_score=top.score,
            feasibility_probability=top.match.feasibility_probability,
            risk_score=risk.risk_score,
            mdp_policy_at_idea=mdp.policy["IDEA"],
            kpi_achievement_rate=validation.kpi_achievement_rate,
            markov_success_probability=markov.prob_reach_success_from_idea,
        )

        return VentureMindReport(
            founder_warnings=warnings_list,
            founder_location={"raw_input": founder.location, "resolved_state": founder_state},
            top_opportunities=[{"sector": s.opportunity.sector, "state": s.opportunity.state,
                                 "score": round(s.score, 4)} for s in scored[:top_n_considered]],
            chosen_opportunity={
                "sector": opp.sector, "state": opp.state,
                "adjusted_capital_required": opp.adjusted_capital_required(),
                "demand_index": opp.demand_index, "growth_rate": opp.growth_rate,
                "competition_density": opp.competition_density, "ecosystem_index": opp.ecosystem_index,
                "required_skills": list(opp.required_skills),
            },
            evidence={
                "opportunity_score": top.score, "components": top.components,
                "feasibility_probability": top.match.feasibility_probability,
                "raw_cosine_similarity": top.match.similarity, "skill_overlap": top.match.skill_overlap,
            },
            resource_plan={
                "budget": resource_plan.budget, "allocations": resource_plan.allocations,
                "fractions": resource_plan.fractions, "lp_status": resource_plan.lp_status,
                "qp_status": resource_plan.qp_status, "kkt_satisfied": resource_plan.kkt_satisfied,
                "kkt_report": resource_plan.kkt_report, "minimums_scaled": resource_plan.minimums_scaled,
            },
            trend={"sector": trend.sector, "slope_per_month": trend.slope_per_month,
                   "r_squared": trend.r_squared, "forecast_next_periods": trend.forecast_next_periods.tolist()},
            risk={"mahalanobis_distance": risk.mahalanobis_distance, "risk_score": risk.risk_score},
            business_evolution={
                "markov": {"prob_reach_success_from_idea": markov.prob_reach_success_from_idea,
                          "prob_reach_failure_from_idea": markov.prob_reach_failure_from_idea,
                          "expected_steps_to_absorption_from_idea": markov.expected_steps_to_absorption_from_idea},
                "mdp": {"policy": mdp.policy, "values": mdp.values},
            },
            validation={
                "kpi_targets": {"min_conversion_rate": validation.kpi_targets.min_conversion_rate,
                               "min_monthly_customers": validation.kpi_targets.min_monthly_customers,
                               "max_cac_inr": validation.kpi_targets.max_cac_inr},
                "kpi_achievement_rate": validation.kpi_achievement_rate,
                "mean_monthly_customers": validation.mean_monthly_customers,
                "mean_conversion_rate": validation.mean_conversion_rate,
                "mean_cac_inr": validation.mean_cac_inr, "simulations": validation.simulations,
            },
            final_decision={"decision": final.decision, "composite_confidence": final.composite_confidence,
                            "rationale": final.rationale},
            model_notes={
                "svm_train_accuracy": self.feasibility_model.train_accuracy,
                "svm_test_accuracy": self.feasibility_model.test_accuracy,
                "svm_cv_accuracy_mean": self.feasibility_model.cv_accuracy_mean,
                "svm_cv_accuracy_std": self.feasibility_model.cv_accuracy_std,
                "disclaimer": "SVM trained on synthetic rule-generated labels, not real startup "
                               "outcomes -- treat as a benchmark feasibility classifier only.",
            },
        )


# =============================================================================
# main / CLI
# =============================================================================

EXAMPLE_FOUNDERS = {
    "example1": dict(skills=["backend", "data", "sales"], budget_inr=1_800_000, location="Hyderabad",
                      interests=["SaaS/B2B", "FinTech"], risk_tolerance="medium", time_availability="full-time"),
    "example2": dict(skills=["design", "content", "marketing"], budget_inr=400_000, location="Jaipur",
                      interests=["EdTech", "Gaming/Media"], risk_tolerance="low", time_availability="part-time"),
}


def run_interactive():
    print("=" * 72)
    print("VENTUREMIND -- interactive founder intake")
    print("=" * 72)
    print("\n" + _describe_skill_menu())
    print("\n" + _describe_sector_menu())
    print("\n(Tip: you don't have to type these exactly -- common phrasings like "
          "'coding', 'web dev', 'ai', 'ecommerce' are understood too, and small "
          "typos are tolerated. Anything the system truly can't match will be "
          "listed as a warning in your report.)\n")

    try:
        skills = input("Your skills (comma-separated): ").split(",")

        while True:
            budget_raw = input("Available budget in INR (e.g. 1500000): ").strip()
            try:
                budget = float(budget_raw)
                if budget <= 0:
                    print("  Budget must be a positive number, try again.")
                    continue
                break
            except ValueError:
                print("  That doesn't look like a number, try again (e.g. 1500000).")

        location = input("Location/city/state (e.g. Hyderabad): ").strip()
        interests = input("Sector interests (comma-separated) or blank for any: ").split(",")
        risk = input("Risk tolerance (low/medium/high) [default: medium]: ").strip() or "medium"
        time_avail = input("Time availability (full-time/part-time) [default: full-time]: ").strip() or "full-time"
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled -- exiting without running the pipeline.")
        return

    pipeline = VentureMindPipeline()
    report = pipeline.run(skills=skills, budget_inr=budget, location=location,
                           interests=[i for i in interests if i.strip()],
                           risk_tolerance=risk, time_availability=time_avail)
    report.print_summary()


def run_example(name):
    params = EXAMPLE_FOUNDERS[name]
    print(f"Running VentureMind on built-in example founder profile '{name}':")
    print(f"  {params}\n")
    pipeline = VentureMindPipeline()
    report = pipeline.run(**params)
    report.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VentureMind decision support system (single file version)")
    parser.add_argument("--interactive", action="store_true", help="Prompt for your own founder profile")
    parser.add_argument("--example", default=None, choices=list(EXAMPLE_FOUNDERS.keys()),
                         help="Run a built-in demo profile instead of asking for input")
    args, _unknown = parser.parse_known_args()

    if args.example is not None and not args.interactive:
        run_example(args.example)
    else:
        run_interactive()