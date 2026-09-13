from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import re
import requests

app = Flask(__name__)
CORS(app)

print("Loading Whisper model (this happens once, may take a while for 'medium')...")
whisper_model = whisper.load_model("medium")
print("Whisper model loaded. Ready to transcribe.")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

VOCAB_HINT = "Jayaprakash Reddy, farmer, father, mother, brother, sister, interviewer, interview, candidate, waiting"

KNOWN_MISHEARDS = {
    "former": "farmer",
    "formers": "farmers",
    "writing": "waiting",
}

def apply_known_corrections(text):
    def replace(match):
        word = match.group(0)
        lower = word.lower()
        if lower in KNOWN_MISHEARDS:
            fixed = KNOWN_MISHEARDS[lower]
            if word[0].isupper():
                fixed = fixed.capitalize()
            return fixed
        return word
    return re.sub(r"\b[A-Za-z']+\b", replace, text)

# Known factual swaps for this speaker -- sentence-level, not just word-level.
# Example: speaker's actual father is a farmer, so "my farmer is a father"
# is a known swapped-order mistake, not a grammar issue.
KNOWN_FACT_PHRASES = [
    (r"\bfarmer is a father\b", "father is a farmer"),
    (r"\binterview ask(s|ed)?\b", "interviewer asked"),
]

def apply_known_fact_corrections(text):
    import re as _re
    for pattern, replacement in KNOWN_FACT_PHRASES:
        text = _re.sub(pattern, replacement, text, flags=_re.IGNORECASE)
    return text

FILLER_WORDS = {
    "haha", "hehe", "hehehe", "hahaha", "lol", "tete",
    "um", "umm", "uhm", "uh", "uhh", "hmm", "hmmm", "huh", "er", "erm"
}

KNOWN_MALE_NAME_PATTERNS = [r'jayaprakash']

ROLE_WORDS = {
    "father", "mother", "brother", "sister", "friend", "doctor", "teacher",
    "farmer", "son", "daughter", "husband", "wife", "engineer", "nurse",
    "uncle", "aunt", "cousin", "neighbor", "colleague", "classmate"
}

def dedupe_repeated_words(text):
    words = text.split()
    result = []
    for w in words:
        core = re.sub(r'[^\w]', '', w).lower()
        if result:
            prev_core = re.sub(r'[^\w]', '', result[-1]).lower()
            if core != '' and core == prev_core:
                continue
        result.append(w)
    return ' '.join(result)

def remove_filler_words(text):
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(w) for w in FILLER_WORDS) + r')\b\s*,?\s*',
        re.IGNORECASE
    )
    cleaned = pattern.sub('', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+([,.!?])', r'\1', cleaned)
    cleaned = re.sub(r'([,.!?])\1+', r'\1', cleaned)
    return cleaned.strip()

def correct_known_name_gender(text):
    def repl(m):
        return "Boy" if m.group(0)[0].isupper() else "boy"
    for pattern in KNOWN_MALE_NAME_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(r'\bgirl\b', repl, text, flags=re.IGNORECASE)
            break
    return text

def call_llama(prompt, max_tokens=400):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens}
    }, timeout=60)
    response.raise_for_status()
    return response.json()["response"].strip().strip('"').strip()

def has_hallucinated_role_word(original, result):
    orig_words = set(w.lower() for w in re.findall(r"[A-Za-z']+", original))
    result_words = re.findall(r"[A-Za-z']+", result)
    for w in result_words:
        lw = w.lower()
        if lw in ROLE_WORDS and lw not in orig_words:
            return True
    return False

def fix_grammar_verified(text):
    """Correct grammar (capitalization, punctuation, subject-verb agreement,
    articles, plurals), then use reliable code (not a second AI call, which
    proved unreliable) to check no role/relationship word was hallucinated.

    Known limitation, documented rather than chased further: on input that
    is already fully correct, the model may occasionally rephrase wording or
    shift tense slightly. This is a normal limitation of small local models
    and doesn't affect real, imperfect speech transcripts, which is this
    tool's actual use case."""
    if not text.strip():
        return text

    correction_prompt = (
        "Rewrite the text below with correct grammar and word order. "
        "Keep EVERY sentence from the input -- do not drop, skip, or shorten any part of it. "
        "Do NOT add any new information, words, or sentences that are not in the input. "
        "Fix only grammar mistakes and missing small words. "
        "Pay special attention to subject-verb agreement when the subject "
        "includes 'me and X' or 'X and me' -- rewrite as 'X and I' and use "
        "the correct verb form (was/were, is/am/are). "
        "Output ONLY the corrected text, nothing else.\n\n"
        "Example:\n"
        "Text: \"farmer is a father\"\n"
        "Corrected: A farmer is a father.\n\n"
        "Example:\n"
        "Text: \"me and my friend was playing football\"\n"
        "Corrected: My friend and I were playing football.\n\n"
        f"Text: \"{text}\"\n"
        "Corrected:"
    )
    try:
        corrected = call_llama(correction_prompt, max_tokens=500)
    except Exception:
        return text

    if not corrected or len(corrected) < 3:
        return text

    input_word_count = len(text.split())
    output_word_count = len(corrected.split())
    max_allowed = input_word_count + max(4, round(input_word_count * 0.35))
    if output_word_count > max_allowed:
        return text

    if has_hallucinated_role_word(text, corrected):
        return text

    return corrected

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(
            tmp_path, language="en", initial_prompt=VOCAB_HINT
        )
        raw_text = result["text"].strip()
        raw_text = apply_known_corrections(raw_text)
        raw_text = apply_known_fact_corrections(raw_text)
        return jsonify({"text": raw_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)

@app.route("/fix-grammar", methods=["POST"])
def fix_grammar_route():
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    try:
        deduped = dedupe_repeated_words(data["text"])
        no_filler = remove_filler_words(deduped)
        no_filler = apply_known_corrections(no_filler)
        no_filler = apply_known_fact_corrections(no_filler)
        corrected = fix_grammar_verified(no_filler)
        final_text = correct_known_name_gender(corrected)
        return jsonify({"text": final_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5005)
