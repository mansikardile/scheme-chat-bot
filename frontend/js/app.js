// ═══════════════════════════════════════════════════════════════
//  SchemeSathi — Frontend Application
//  Premium Edition with Multilingual Support, Voice I/O, Theme Toggle
// ═══════════════════════════════════════════════════════════════

const API_BASE = '';

// ── State ──
let sessionId = null;
let isWaiting = false;
let selectedLanguage = 'en';
let autoSpeak = false;
let recognition = null;
let synth = window.speechSynthesis;
let isRecording = false;
let currentTheme = localStorage.getItem('schemesathi_theme') || 'dark';

// ── Language mappings ──
const SPEECH_LANG_MAP = {
    en: 'en-IN', hi: 'hi-IN', ta: 'ta-IN', te: 'te-IN',
    bn: 'bn-IN', mr: 'mr-IN', gu: 'gu-IN', kn: 'kn-IN',
    ml: 'ml-IN', pa: 'pa-IN', or: 'or-IN', as: 'as-IN', ur: 'ur-IN'
};

const GREETINGS = {
    en: "Hello! I'm SchemeSathi, your AI assistant. Tell me — what kind of help are you looking for? (jobs, education, health, loans, housing...)",
    hi: "नमस्ते! मैं SchemeSathi हूं, आपका AI सहायक। बताइए — आपको किस तरह की मदद चाहिए? (नौकरी, शिक्षा, स्वास्थ्य, लोन, आवास...)",
    mr: "नमस्कार! मी SchemeSathi, तुमचा AI सहाय्यक. सांगा — तुम्हाला कोणत्या प्रकारची मदत हवी आहे?",
    ta: "வணக்கம்! நான் SchemeSathi, உங்கள் AI உதவியாளர். சொல்லுங்கள் — உங்களுக்கு என்ன வகையான உதவி தேவை?",
    te: "నమస్కారం! నేను SchemeSathi, మీ AI సహాయకుడిని. చెప్పండి — మీకు ఏ రకమైన సహాయం కావాలి?",
    bn: "নমস্কার! আমি SchemeSathi, আপনার AI সহায়ক। বলুন — আপনার কী ধরনের সাহায্য দরকার?",
    gu: "નમસ્તે! હું SchemeSathi, તમારો AI સહાયક. કહો — તમારે કયા પ્રકારની મદદ જોઈએ છે?",
    kn: "ನಮಸ್ಕಾರ! ನಾನು SchemeSathi, ನಿಮ್ಮ AI ಸಹಾಯಕ. ಹೇಳಿ — ನಿಮಗೆ ಯಾವ ರೀತಿಯ ಸಹಾಯ ಬೇಕು?",
    ml: "നമസ്കാരം! ഞാൻ SchemeSathi, നിങ്ങളുടെ AI സഹായി. പറയൂ — നിങ്ങൾക്ക് എന്ത് തരത്തിലുള്ള സഹായം വേണം?",
    pa: "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ SchemeSathi ਹਾਂ, ਤੁਹਾਡਾ AI ਸਹਾਇਕ। ਦੱਸੋ — ਤੁਹਾਨੂੰ ਕਿਸ ਤਰ੍ਹਾਂ ਦੀ ਮਦਦ ਚਾਹੀਦੀ ਹੈ?",
    or: "ନମସ୍କାର! ମୁଁ SchemeSathi, ଆପଣଙ୍କ AI ସହାଯ଼କ। କୁହନ୍ତୁ — ଆପଣଙ୍କୁ କେଉଁ ପ୍ରକାର ସାହାଯ଼୍ଯ ଦରକାର?",
    ur: "السلام علیکم! میں SchemeSathi ہوں، آپ کا AI معاون۔ بتائیں — آپ کو کس قسم کی مدد چاہیے؟"
};

// ── Helpers ──
const $ = id => document.getElementById(id);

// ═══ Init ═══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Theme setup
    applyTheme(currentTheme);
    $('themeToggleBtn')?.addEventListener('click', toggleTheme);

    // Form
    $('chatForm').addEventListener('submit', handleSubmit);
    $('messageInput').addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e); }
    });
    $('messageInput').addEventListener('input', () => {
        autoResize($('messageInput'));
        $('sendBtn').disabled = !$('messageInput').value.trim();
    });

    // Header buttons
    $('newChatBtn').addEventListener('click', startNewChat);
    $('audioToggleBtn').addEventListener('click', toggleAutoSpeak);
    $('micBtn').addEventListener('click', toggleRecording);

    // Language chips
    document.querySelectorAll('.lang-chip').forEach(chip => {
        chip.addEventListener('click', () => selectLanguage(chip));
    });

    // Modal
    document.querySelector('.modal-backdrop')?.addEventListener('click', closeModal);
    $('closeModalBtn')?.addEventListener('click', closeModal);

    // Keyboard shortcut: Escape closes modal
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeModal();
    });

    // Init speech
    initSpeechRecognition();

    // Create session
    await createSession();
}

// ═══ Language Selection ════════════════════════════════════
function selectLanguage(chip) {
    selectedLanguage = chip.dataset.lang;

    document.querySelectorAll('.lang-chip').forEach(c => c.classList.remove('selected'));
    chip.classList.add('selected');

    if (recognition) recognition.lang = SPEECH_LANG_MAP[selectedLanguage] || 'en-IN';

    setTimeout(() => {
        $('languageScreen').classList.add('hidden');
        $('messagesContainer').classList.remove('hidden');

        const greeting = GREETINGS[selectedLanguage] || GREETINGS.en;
        renderAIMessage(greeting, []);
        if (autoSpeak) speakText(greeting);

        $('messageInput').focus();
    }, 250);
}
