"""
AASNAA Engineering - Outreach Email Generator
---------------------------------------------
Scrapes a company website, builds a rich company profile, and drafts a
personalized outreach email to win MEP Design & BIM projects.

SETUP
  1. pip install streamlit requests beautifulsoup4 google-generativeai python-dotenv lxml
  2. streamlit run app.py
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import google.generativeai as genai
import os
import re
import json
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------
load_dotenv()

# HARDCODED API KEY (replace with your real key)
API_KEY = "AQ.Ab8RN6Ix9AKnuNXO1R1U71z5deAzYjM-q36UASKVJ5Q-hZ0SGw"

if not os.getenv("GEMINI_API_KEY"):
    API_KEY = "AQ.Ab8RN6Ix9AKnuNXO1R1U71z5deAzYjM-q36UASKVJ5Q-hZ0SGw"
else:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ No GEMINI_API_KEY found.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "models/gemini-3.6-flash"
model = genai.GenerativeModel(MODEL_NAME)

st.set_page_config(page_title="AASNAA Outreach", page_icon="📧", layout="wide")
st.title("📧 AASNAA Engineering - Outreach Email Generator")
st.caption("Find prospects → Research → Generate personalized emails → Win MEP/BIM projects")

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
for key, default in {
    "profile": None,
    "email_draft": None,
    "sender_info": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def call_model_json(prompt: str, retries: int = 2, delay: float = 1.5):
    """Call Gemini with JSON-mode output, with light retry on transient errors."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"Model call failed after retries: {last_err}")


# ------------------------------------------------------------------
# PHASE 1: DISCOVER PAGES
# ------------------------------------------------------------------
DEFAULT_PATHS = [
    "/", "/about", "/about-us", "/company", "/services", "/products",
    "/solutions", "/team", "/leadership", "/contact", "/contact-us", 
    "/our-offices", "/offices", "/locations", "/pricing",
    "/careers", "/news", "/press", "/blog", "/case-studies", "/projects",
]

KEYWORD_HINTS = (
    "about", "team", "leadership", "service", "product", "solution",
    "contact", "press", "news", "case-stud", "customer", "portfolio",
    "mission", "story", "career", "pricing", "project", "office", "location",
)


def discover_urls_from_sitemap(base_url: str, limit: int = 12):
    """Try to pull extra, relevant URLs from sitemap.xml (best-effort)."""
    found = []
    try:
        r = requests.get(urljoin(base_url, "/sitemap.xml"), timeout=6)
        if r.status_code == 200 and "<" in r.text:
            root = ET.fromstring(r.text)
            for loc in root.iter():
                if loc.tag.endswith("loc") and loc.text:
                    u = loc.text.strip()
                    if any(k in u.lower() for k in KEYWORD_HINTS):
                        found.append(u)
                    if len(found) >= limit:
                        break
    except Exception:
        pass
    return found


def check_robots_allowed(base_url: str) -> bool:
    """Best-effort robots.txt check."""
    try:
        robots = requests.get(urljoin(base_url, "/robots.txt"), timeout=5).text
        for block in robots.split("User-agent:"):
            if block.strip().lower().startswith("*"):
                if re.search(r"^\s*Disallow:\s*/\s*$", block, re.MULTILINE):
                    return False
        return True
    except Exception:
        return True


def fetch_page(url: str, headers: dict, timeout: int = 10):
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            return url, r.text
    except Exception:
        pass
    return url, None


def extract_signal_text(html: str, max_chars: int = 3000) -> str:
    """Pull the highest-signal content from a page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "svg", "noscript"]):
        tag.decompose()

    parts = []
    if soup.title and soup.title.string:
        parts.append(f"TITLE: {soup.title.string.strip()}")

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        parts.append(f"META DESCRIPTION: {meta_desc['content'].strip()}")

    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)]
    if headings:
        parts.append("HEADINGS: " + " | ".join(headings[:25]))

    body_text = soup.get_text(separator=" ", strip=True)
    body_text = " ".join(body_text.split())
    parts.append("BODY: " + body_text)

    combined = "\n".join(parts)
    return combined[:max_chars]


# ------------------------------------------------------------------
# PHASE 2: SCRAPE + BUILD RICH TEXT CORPUS
# ------------------------------------------------------------------
def scrape_website(url: str):
    if not check_robots_allowed(url):
        return {"error": "Blocked by robots.txt"}

    urls_to_try = list(dict.fromkeys(
        [urljoin(url, p) for p in DEFAULT_PATHS] + discover_urls_from_sitemap(url)
    ))

    headers = {"User-Agent": "Mozilla/5.0 (compatible; AASNAA-ResearchBot/1.0)"}
    page_texts = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_page, u, headers) for u in urls_to_try]
        for future in as_completed(futures):
            page_url, html = future.result()
            if html:
                path = urlparse(page_url).path or "/"
                page_texts[path] = extract_signal_text(html)

    if not page_texts:
        return {"error": "Could not access website"}

    corpus = ""
    for path, text in page_texts.items():
        corpus += f"\n\n=== PAGE: {path} ===\n{text}"
    corpus = corpus[:16000]

    return {"corpus": corpus, "pages_found": list(page_texts.keys())}


# ------------------------------------------------------------------
# PHASE 3: RICH EXTRACTION (COMPREHENSIVE)
# ------------------------------------------------------------------
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "tagline_or_mission": {"type": "string"},
        "industry": {"type": "string"},
        "founded_year": {"type": "string"},
        "company_size": {"type": "string"},
        "headquarters": {"type": "string"},
        "other_locations": {"type": "array", "items": {"type": "string"}},
        "services": {"type": "array", "items": {"type": "string"}},
        "products": {"type": "array", "items": {"type": "string"}},
        "specializations": {"type": "array", "items": {"type": "string"}},
        "target_customers": {"type": "string"},
        "geographic_markets": {"type": "array", "items": {"type": "string"}},
        "unique_value_proposition": {"type": "string"},
        "key_people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "location": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "department": {"type": "string"},
                    "contact_type": {"type": "string"},
                },
            },
        },
        "all_emails": {"type": "array", "items": {"type": "string"}},
        "all_phones": {"type": "array", "items": {"type": "string"}},
        "all_addresses": {"type": "array", "items": {"type": "string"}},
        "recent_projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "location": {"type": "string"},
                    "type": {"type": "string"},
                    "size": {"type": "string"},
                    "year": {"type": "string"},
                },
            },
        },
        "recent_news_or_achievements": {"type": "array", "items": {"type": "string"}},
        "awards_or_certifications": {"type": "array", "items": {"type": "string"}},
        "tech_stack_or_tools_mentioned": {"type": "array", "items": {"type": "string"}},
        "methodologies": {"type": "array", "items": {"type": "string"}},
        "testimonials_or_social_proof": {"type": "array", "items": {"type": "string"}},
        "tone_of_voice": {"type": "string"},
        "likely_pain_points": {"type": "array", "items": {"type": "string"}},
        "contact_email": {"type": "string"},
        "phone": {"type": "string"},
        "social_links": {"type": "array", "items": {"type": "string"}},
        "partnerships": {"type": "array", "items": {"type": "string"}},
        "revenue_or_funding": {"type": "string"},
    },
    "required": ["company_name", "services", "headquarters", "contact_email"],
}


def build_profile(corpus: str, url: str) -> dict:
    prompt = f"""You are a B2B sales research analyst. Extract MAXIMUM information from this website.

Website URL: {url}

Website text (multiple pages):
{corpus}

IMPORTANT: Look for CONTACT PAGES, TEAM PAGES, and "OUR OFFICES" pages that list:
- Regional managers, area managers, project managers
- Department heads (Sales, Business Development, Operations)
- Contact persons with their emails and phone numbers
- Office addresses with complete contact details

Extract EVERY detail you can find:

1. BASIC INFO:
   - Company name (exact legal name if available)
   - Tagline/mission statement (exact words from website)
   - Industry/sector (be specific, not generic)
   - Founded year (if mentioned)
   - Company size (employee count range)
   - Headquarters (full address if available)
   - Other office locations (list all cities/countries with complete addresses)

2. SERVICES & PRODUCTS:
   - ALL services mentioned (list every single one)
   - ALL products (if any)
   - Specializations/niches
   - Target customers/industries they serve
   - Geographic markets (countries/regions they operate in)

3. PEOPLE & CONTACTS (CRITICAL):
   - ALL operational contacts (Area Managers, Regional Heads, Project Managers, Sales Heads)
   - Their exact titles
   - Their locations (city/office with complete address)
   - Their DIRECT contact details (email, mobile, landline)
   - NOT just board members - focus on operational staff you can actually contact
   - Format: "Name - Title - Location - Email - Phone"
   - Mark each person as "Operational" or "Board" in contact_type field

4. CONTACT INFORMATION (EXTRACT ALL):
   - EVERY email address on the website (info@, contact@, name.surname@, etc.)
   - EVERY phone number (mobile numbers, landlines, toll-free)
   - EVERY office address (street, area, city, state, pincode, country)
   - Contact page information
   - Social media links (LinkedIn, Twitter, Facebook, Instagram, YouTube)

5. PROJECTS & ACHIEVEMENTS:
   - Recent projects (last 2 years, with names/locations/sizes)
   - Project types (residential, commercial, healthcare, infrastructure, etc.)
   - Project values/sizes (sq ft, crore, MW, km - any numbers)
   - Awards/certifications (LEED, ISO, etc.)
   - Press releases/news mentions
   - Client testimonials/case studies

6. TECHNICAL DETAILS:
   - Software/tools they use (Revit, AutoCAD, Navisworks, etc.)
   - Methodologies (BIM Level 2, Lean Construction, etc.)
   - Standards compliance (local building codes, international standards)
   - Sustainability practices (green building, carbon neutral, etc.)

7. BUSINESS MODEL:
   - How they make money (EPC, contracting, development, consulting, etc.)
   - Revenue mentions (if public company)
   - Partnerships/alliances
   - Recent expansions/acquisitions

8. PAIN POINTS (INFER):
   Based on their services, size, and industry, infer 3-5 specific challenges:
   - Operational challenges (scaling, resource constraints, etc.)
   - Technical challenges (coordination, compliance, etc.)
   - Market challenges (competition, pricing pressure, etc.)
   - Growth challenges (hiring, capacity, etc.)

RULES:
- Extract EVERY fact - don't summarize, list everything
- Use exact numbers (300K sq ft, 50 floors, ₹1000 crore, etc.)
- Use exact names (project names, people names, award names)
- For contact details: extract EVERY email, phone, address found
- PRIORITIZE operational contacts over board members
- If something is not found, use "Not Found" (string) or [] (empty array)
- NEVER invent facts - only extract what's actually on the website
- For pain points, clearly mark as "(Inference)"

Return ONLY JSON matching this schema:
{json.dumps(PROFILE_SCHEMA)}
"""
    return call_model_json(prompt)


def clean_profile(data: dict) -> dict:
    """Light normalization pass."""

    def clean_text(t):
        if not t or t == "Not Found":
            return "Not Found"
        return " ".join(str(t).split()).strip()

    def clean_list(lst):
        if not isinstance(lst, list):
            return []
        return [clean_text(x) if isinstance(x, str) else x for x in lst if x]

    def clean_email(e):
        if not e or e == "Not Found":
            return "Not Found"
        e = e.lower().strip()
        return e if re.match(r"^[\w.\-]+@[\w.\-]+\.\w+$", e) else "Not Found"

    abbreviations = {
        "Hyd": "Hyderabad", "Blr": "Bangalore", "USA": "United States",
        "UK": "United Kingdom", "SF": "San Francisco", "NYC": "New York City",
        "UAE": "United Arab Emirates", "KSA": "Saudi Arabia", "Qatar": "Qatar",
    }
    hq = clean_text(data.get("headquarters", "Not Found"))
    for abbr, full in abbreviations.items():
        if re.search(rf"\b{re.escape(abbr)}\b", hq):
            hq = re.sub(rf"\b{re.escape(abbr)}\b", full, hq)

    cleaned = dict(data)
    cleaned["company_name"] = clean_text(data.get("company_name"))
    cleaned["tagline_or_mission"] = clean_text(data.get("tagline_or_mission"))
    cleaned["industry"] = clean_text(data.get("industry"))
    cleaned["headquarters"] = hq
    cleaned["unique_value_proposition"] = clean_text(data.get("unique_value_proposition"))
    cleaned["target_customers"] = clean_text(data.get("target_customers"))
    cleaned["tone_of_voice"] = clean_text(data.get("tone_of_voice"))
    cleaned["contact_email"] = clean_email(data.get("contact_email"))
    cleaned["revenue_or_funding"] = clean_text(data.get("revenue_or_funding"))
    
    for field in ["services", "products", "specializations", "other_locations", 
                  "geographic_markets", "recent_news_or_achievements",
                  "awards_or_certifications", "tech_stack_or_tools_mentioned",
                  "methodologies", "testimonials_or_social_proof", 
                  "likely_pain_points", "social_links", "partnerships",
                  "all_emails", "all_phones", "all_addresses"]:
        cleaned[field] = clean_list(data.get(field, []))
    
    # Clean recent projects (list of dicts)
    projects = data.get("recent_projects", [])
    cleaned["recent_projects"] = []
    for proj in projects:
        if isinstance(proj, dict):
            cleaned_proj = {
                "name": clean_text(proj.get("name", "")),
                "location": clean_text(proj.get("location", "")),
                "type": clean_text(proj.get("type", "")),
                "size": clean_text(proj.get("size", "")),
                "year": clean_text(proj.get("year", "")),
            }
            cleaned["recent_projects"].append(cleaned_proj)
    
    # Clean key people (list of dicts with contact info)
    people = data.get("key_people", [])
    cleaned["key_people"] = []
    for person in people:
        if isinstance(person, dict):
            cleaned_person = {
                "name": clean_text(person.get("name", "")),
                "title": clean_text(person.get("title", "")),
                "location": clean_text(person.get("location", "")),
                "email": clean_email(person.get("email", "")),
                "phone": clean_text(person.get("phone", "")),
                "department": clean_text(person.get("department", "")),
                "contact_type": clean_text(person.get("contact_type", "Operational")),
            }
            cleaned["key_people"].append(cleaned_person)
    
    return cleaned


# ------------------------------------------------------------------
# PHASE 4: PERSONALIZED EMAIL (BULLET-POINT STYLE)
# ------------------------------------------------------------------
EMAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "body": {"type": "string"},
        "personalization_hooks_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["subject", "body"],
}

BANNED_PHRASES = [
    "i hope this email finds you well", "truly commendable", "impressive work",
    "outstanding", "seamless", "seamlessly", "cutting-edge", "state-of-the-art",
    "world-class", "revolutionize", "streamline", "game-changer", "synergy",
    "leverage", "best-in-class", "circle back", "touch base", "excited to",
    "thrilled to", "look forward", "feel free", "don't hesitate", "reach out",
]

SENIOR_TITLE_PRIORITY = [
    "ceo", "founder", "co-founder", "president", "managing director",
    "owner", "director", "partner", "vp", "vice president", "head of",
    "principal", "chief",
]


def pick_decision_maker(profile: dict):
    """Pick the most senior named person from key_people."""
    people = profile.get("key_people", [])
    if not people:
        return None
    for keyword in SENIOR_TITLE_PRIORITY:
        for person in people:
            if not isinstance(person, dict):
                continue
            title = (person.get("title") or "").lower()
            name = (person.get("name") or "").strip()
            if keyword in title and name:
                return {"name": name, "title": person.get("title", "").strip()}
    for person in people:
        if isinstance(person, dict) and person.get("name"):
            return {"name": person["name"].strip(), "title": person.get("title", "").strip()}
    return None


def generate_email(profile: dict, sender: dict) -> dict:
    decision_maker = pick_decision_maker(profile)
    
    if decision_maker:
        first_name = decision_maker["name"].split()[0]
        greeting_instruction = (
            f'Open with "Hi {first_name}," — {decision_maker["name"]} is the {decision_maker["title"]}. '
            f'Keep it professional but conversational.'
        )
    else:
        greeting_instruction = (
            'No named contact. Open with "Hi," or "Hello,". '
            'Do NOT use "Hi Team," or "Dear Sir/Madam,".'
        )

    services_str = ', '.join(profile.get('services', [])) or 'Not Found'
    recent_achievements = ', '.join(profile.get('recent_news_or_achievements', [])) or 'Not Found'
    
    # Find ONE specific fact to reference
    specific_fact = ""
    if recent_achievements and recent_achievements != 'Not Found':
        specific_fact = recent_achievements.split(',')[0].strip()
    elif services_str and services_str != 'Not Found':
        specific_fact = f"your work in {services_str.split(',')[0].strip()}"
    else:
        specific_fact = f"your {profile.get('industry', 'projects')} portfolio"
    
    prompt = f"""You are {sender.get('name')}, {sender.get('title')} at {sender.get('company')}.
Write a SHORT, BULLET-POINT email that can be scanned in 10 seconds.

GREETING:
{greeting_instruction}

OPENING (1 sentence):
- Reference ONE specific fact: {specific_fact}
- Keep it direct, no flattery

BULLET POINTS (3-5 max):
• AASNAA Engineering capabilities (50+ MEP professionals)
• Key services (HVAC, Plumbing, Electrical, BIM, 3D coordination)
• Track record (300+ projects globally)
• Value prop (30-40% cost reduction, 48-hour turnaround)
• Compliance (LEED, ESTIDAMA, local codes)

CALL TO ACTION (1 sentence):
- Simple ask: "Worth a 15-min call?" or "Open to a quick chat?"

HARD RULES:
- TOTAL email: 80-120 words MAX
- Use bullet points (•) for services/capabilities
- Each bullet: 1 line, 10-15 words max
- NO long paragraphs
- NO: {", ".join(BANNED_PHRASES[:10])}
- Sign off: "Best regards," then name, title, company
- Add: "If not relevant, reply 'stop'."

TARGET COMPANY:
- Name: {profile['company_name']}
- Industry: {profile.get('industry', 'Not Found')}
- Location: {profile.get('headquarters', 'Not Found')}
- Specific fact to reference: {specific_fact}

SENDER INFO:
- Name: {sender.get('name')}
- Title: {sender.get('title')}
- Company: {sender.get('company')}
- Email: {sender.get('email')}

EXAMPLE FORMAT:
Subject: MEP/BIM support for [Company]

Hi [Name],

Noticed [specific fact about them].

AASNAA Engineering can help:
• 50+ MEP professionals (HVAC, Plumbing, Electrical, Fire)
• BIM modeling & 3D coordination (48-hour turnaround)
• 300+ projects globally (India, UAE, USA, UK)
• 30-40% cost reduction vs in-house teams
• LEED, ESTIDAMA compliance

Worth a 15-min call this week?

Best regards,
[Your Name]
[Title], AASNAA Engineering
+91 [Phone] | info@aasnaaengineers.com

If not relevant, reply 'stop'.

Return ONLY JSON matching this schema:
{json.dumps(EMAIL_SCHEMA)}
"""
    
    return call_model_json(prompt)


# ------------------------------------------------------------------
# PHASE 5: QUALITY CHECK (LENIENT)
# ------------------------------------------------------------------
def _find_banned_phrases(text: str) -> list:
    lower = text.lower()
    return [p for p in BANNED_PHRASES if p in lower]


def quality_check(email_draft: str, profile: dict, hooks_used: list, sender: dict) -> dict:
    banned_found = _find_banned_phrases(email_draft)
    
    # Allow all numbers that appear in sender info (AASNAA's stats)
    sender_text = json.dumps(sender).lower()
    email_numbers = re.findall(r"\b\d{1,3}\s?%|\b\d+x\b|\$\d[\d,]*", email_draft.lower())
    unverified_numbers = [n for n in email_numbers if n not in sender_text]
    
    lines = email_draft.split('\n')
    body_lines = [l for l in lines[2:] if l.strip() and not l.strip().startswith('•')]
    
    # More lenient company name check
    company_name_check = (
        profile["company_name"].lower() in email_draft.lower() or 
        profile.get("industry", "").lower() in email_draft.lower() or
        any(word.lower() in email_draft.lower() for word in profile["company_name"].split() if len(word) > 3)
    )
    
    checks = {
        "has_company_name": company_name_check,
        "has_subject": "Subject:" in email_draft,
        "has_opt_out": "unsubscribe" in email_draft.lower() or "stop" in email_draft.lower() or "not relevant" in email_draft.lower(),
        "not_too_long": len(email_draft.split()) < 150,
        "not_too_short": len(email_draft.split()) > 60,
        "has_specific_hook": len(hooks_used) > 0,
        "no_banned_phrases": len(banned_found) == 0,
        "banned_phrases_found": banned_found,
        "no_unverified_numbers": len(unverified_numbers) == 0,
        "unverified_numbers_found": unverified_numbers,
        "has_bullet_points": "•" in email_draft or "-" in email_draft,
        "bullet_count": email_draft.count("•") + email_draft.count("-"),
        "no_generic_greeting": "hi team" not in email_draft.lower() and "dear team" not in email_draft.lower(),
    }

    review_prompt = f"""Review this bullet-point outreach email.

Email:
{email_draft}

Company: {profile['company_name']}

Check for:
1. Is it scannable in 10 seconds? (short bullets, no long paragraphs)
2. Does it have at least ONE specific fact about this company?
3. Does it avoid banned sales clichés?
4. Is the ask simple and low-friction?

Return JSON: {{"passed": true/false, "score": 1-10, "issues": ["specific issues"]}}

Pass if it's short, scannable, and has at least one specific fact.
"""
    
    try:
        ai_review = call_model_json(review_prompt)
        checks["ai_passed"] = ai_review.get("passed", True)
        checks["ai_score"] = ai_review.get("score", 8)
        checks["ai_issues"] = ai_review.get("issues", [])
    except Exception:
        checks["ai_passed"] = True
        checks["ai_score"] = 8
        checks["ai_issues"] = []

    # Only require critical checks (be very lenient)
    all_passed = all([
        checks["has_subject"],
        checks["has_opt_out"],
        checks["not_too_long"],
        checks["not_too_short"],
        checks["has_specific_hook"],
        checks["no_banned_phrases"],
        checks["has_bullet_points"],
        checks["ai_passed"],
    ])
    
    return {"passed": all_passed, "checks": checks}


# ------------------------------------------------------------------
# UI — PHASE 1: INPUT
# ------------------------------------------------------------------
st.subheader("1️⃣ Enter Target Company Website")
url = st.text_input("Paste company website URL:", placeholder="https://example.com")

with st.expander("✉️ Your Details (AASNAA Engineering)"):
    c1, c2 = st.columns(2)
    with c1:
        sender_name = st.text_input("Your name", value="[Your Name]")
        sender_title = st.text_input("Your title", value="Business Development Manager")
        sender_company = st.text_input("Your company", value="AASNAA Engineering Private Limited")
    with c2:
        sender_email = st.text_input("Your email", value="info@aasnaaengineers.com")
    sender_offer = st.text_area(
        "What AASNAA offers / why reaching out:",
        value="We are MEP Design and Engineering Consultants specializing in:\n"
              "- HVAC, Plumbing, Electrical design for commercial/residential projects\n"
              "- BIM (Building Information Modeling) Services\n"
              "- 3D Coordination and Clash Detection\n"
              "- Fire Fighting Systems Engineering\n"
              "- Sustainability Consulting\n\n"
              "We help architects, contractors, and developers reduce design errors, cut rework costs, "
              "and accelerate project delivery through precise MEP coordination and BIM implementation.\n\n"
              "Track record: 300+ projects globally, 50+ professionals, serving clients in India, USA, UK, Middle East.",
        height=200,
    )
    st.session_state.sender_info = {
        "name": sender_name, "title": sender_title, "company": sender_company,
        "email": sender_email, "offer": sender_offer,
    }

if st.button("🚀 Research Company", type="primary"):
    if not url:
        st.error("❌ Please enter a URL")
    elif not is_valid_url(url):
        st.error("❌ Invalid URL format")
    else:
        with st.spinner("Crawling site and gathering pages..."):
            scraped = scrape_website(url)

        if "error" in scraped:
            st.error(f"❌ {scraped['error']}")
        else:
            st.info(f"Pulled content from {len(scraped['pages_found'])} pages: {', '.join(scraped['pages_found'])}")
            with st.spinner("Building detailed company profile..."):
                try:
                    raw_profile = build_profile(scraped["corpus"], url)
                    st.session_state.profile = clean_profile(raw_profile)
                    st.session_state.email_draft = None
                    st.success("✅ Research complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Profile extraction failed: {e}")

# ------------------------------------------------------------------
# UI — SHOW PROFILE (COMPREHENSIVE WITH ALL CONTACTS)
# ------------------------------------------------------------------
if st.session_state.profile:
    p = st.session_state.profile

    st.divider()
    st.subheader("📋 Company Profile (Comprehensive)")

    # Basic Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Name:** {p['company_name']}")
        st.markdown(f"**Tagline:** {p.get('tagline_or_mission', 'Not Found')}")
        st.markdown(f"**Industry:** {p.get('industry', 'Not Found')}")
        st.markdown(f"**Founded:** {p.get('founded_year', 'Not Found')}")
    with col2:
        st.markdown(f"**HQ:** {p.get('headquarters', 'Not Found')}")
        st.markdown(f"**Size:** {p.get('company_size', 'Not Found')}")
        st.markdown(f"**Email:** {p.get('contact_email', 'Not Found')}")
        st.markdown(f"**Phone:** {p.get('phone', 'Not Found')}")
    with col3:
        st.markdown(f"**UVP:** {p.get('unique_value_proposition', 'Not Found')}")
        st.markdown(f"**Tone:** {p.get('tone_of_voice', 'Not Found')}")

    # Other locations
    if p.get("other_locations") and p["other_locations"] != ["Not Found"]:
        st.markdown(f"**Other Offices:** {', '.join(p['other_locations'])}")

    # ALL CONTACT INFORMATION (NEW SECTION)
    st.divider()
    st.subheader("📞 Contact Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if p.get("all_emails") and p["all_emails"] != ["Not Found"]:
            st.markdown("**📧 All Emails Found:**")
            for email in p["all_emails"]:
                st.markdown(f"• {email}")
    with col2:
        if p.get("all_phones") and p["all_phones"] != ["Not Found"]:
            st.markdown("**📱 All Phones Found:**")
            for phone in p["all_phones"]:
                st.markdown(f"• {phone}")
    with col3:
        if p.get("all_addresses") and p["all_addresses"] != ["Not Found"]:
            st.markdown("**📍 All Addresses:**")
            for addr in p["all_addresses"]:
                st.markdown(f"• {addr}")

    # Key People WITH CONTACTS - Show operational contacts first
    if p.get("key_people") and len(p["key_people"]) > 0:
        st.divider()
        
        # Separate operational contacts from board members
        operational = [person for person in p["key_people"] 
                      if isinstance(person, dict) and 
                      person.get("contact_type") == "Operational"]
        board = [person for person in p["key_people"] 
                if isinstance(person, dict) and 
                person.get("contact_type") == "Board"]
        
        # Show operational contacts first (these are useful!)
        if operational:
            st.markdown("**👥 Operational Contacts (Area/Regional Managers):**")
            for person in operational:
                if person.get("name"):
                    st.markdown(f"**• {person['name']}** - {person.get('title', 'N/A')}")
                    if person.get("department"):
                        st.markdown(f"  🏢 Department: {person['department']}")
                    if person.get("location"):
                        st.markdown(f"  📍 Location: {person['location']}")
                    if person.get("email") and person['email'] != "Not Found":
                        st.markdown(f"  📧 Email: {person['email']}")
                    if person.get("phone") and person['phone'] != "Not Found":
                        st.markdown(f"  📱 Phone: {person['phone']}")
        
        # Show board members separately (less useful for outreach)
        if board:
            st.markdown("**👔 Board Members (For Reference):**")
            for person in board:
                if person.get("name"):
                    st.markdown(f"• {person['name']} - {person.get('title', 'N/A')}")

    # Services & Specializations
    col1, col2 = st.columns(2)
    with col1:
        if p.get("services") and p["services"] != ["Not Found"]:
            st.markdown("**Services:**")
            for svc in p["services"]:
                st.markdown(f"• {svc}")
    with col2:
        if p.get("specializations") and p["specializations"] != ["Not Found"]:
            st.markdown("**Specializations:**")
            for spec in p["specializations"]:
                st.markdown(f"• {spec}")

    # Recent Projects
    if p.get("recent_projects") and len(p["recent_projects"]) > 0:
        st.divider()
        st.markdown("**🏗️ Recent Projects:**")
        for proj in p["recent_projects"]:
            if isinstance(proj, dict) and proj.get("name"):
                st.markdown(f"**{proj['name']}** ({proj.get('location', 'Unknown')})")
                st.markdown(f"  - Type: {proj.get('type', 'N/A')} | Size: {proj.get('size', 'N/A')} | Year: {proj.get('year', 'N/A')}")

    # Awards & Certifications
    if p.get("awards_or_certifications") and p["awards_or_certifications"] != ["Not Found"]:
        st.divider()
        st.markdown("**🏆 Awards & Certifications:**")
        for award in p["awards_or_certifications"]:
            st.markdown(f"• {award}")

    # Tech Stack
    if p.get("tech_stack_or_tools_mentioned") and p["tech_stack_or_tools_mentioned"] != ["Not Found"]:
        st.divider()
        st.markdown("**💻 Software/Tools:**")
        for tool in p["tech_stack_or_tools_mentioned"]:
            st.markdown(f"• {tool}")

    # Recent News
    if p.get("recent_news_or_achievements") and p["recent_news_or_achievements"] != ["Not Found"]:
        st.divider()
        st.markdown("**📰 Recent News/Achievements:**")
        for news in p["recent_news_or_achievements"]:
            st.markdown(f"• {news}")

    # Pain Points
    if p.get("likely_pain_points") and p["likely_pain_points"] != ["Not Found"]:
        st.divider()
        st.markdown("**⚠️ Likely Pain Points (Inferred):**")
        for pain in p["likely_pain_points"]:
            st.markdown(f"• {pain}")

    # Social Links
    if p.get("social_links") and p["social_links"] != ["Not Found"]:
        st.divider()
        st.markdown("**🔗 Social Media:**")
        for link in p["social_links"]:
            st.markdown(f"• {link}")

    with st.expander("Full raw profile (JSON)"):
        st.json(p)

    # -----------------------------------------------------------
    # UI — GENERATE EMAIL
    # -----------------------------------------------------------
    if st.button("✍️ Generate Personalized Email"):
        with st.spinner("Writing grounded, specific email..."):
            try:
                email_result = generate_email(p, st.session_state.sender_info)
                draft = f"Subject: {email_result['subject']}\n\n{email_result['body']}"
                st.session_state.email_draft = draft
                st.session_state.hooks_used = email_result.get("personalization_hooks_used", [])
                st.success("✅ Email generated!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Email generation failed: {e}")

    # -----------------------------------------------------------
    # UI — REVIEW & QC
    # -----------------------------------------------------------
    if st.session_state.email_draft:
        draft = st.session_state.email_draft
        hooks = st.session_state.get("hooks_used", [])

        st.divider()
        st.subheader("👁️ Review & Edit Email")

        if hooks:
            st.caption("Personalization hooks used: " + ", ".join(hooks))

        lines = draft.split("\n", 1)
        subject = lines[0].replace("Subject:", "").strip() if lines else ""
        body = lines[1].strip() if len(lines) > 1 else draft

        with st.spinner("Running quality check..."):
            qc = quality_check(draft, p, hooks, st.session_state.sender_info)

        if qc["passed"]:
            st.success("✅ Quality check passed")
        else:
            st.warning("⚠️ Quality check flagged issues")
            st.json(qc["checks"])

        new_subject = st.text_area("Subject:", value=subject, height=50)
        new_body = st.text_area("Email Body:", value=body, height=350)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve & Save"):
                final = f"Subject: {new_subject}\n\n{new_body}"
                st.session_state.email_draft = final
                st.success("✅ Approved! Ready to send (Phase 7)")
                st.text_area("Final Email:", value=final, height=400)
        with col2:
            if st.button("🔄 Regenerate"):
                st.session_state.email_draft = None
                st.rerun()

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
st.markdown("**Next:** Phase 7 (Sending) → add Gmail SMTP or Mailgun/SendGrid API.")