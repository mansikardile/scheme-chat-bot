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
    ml: "നമസ്കാരം! ഞാൻ SchemeSathi, നിങ്ങളുടെ AI സഹായി. പറയൂ — നിങ്ങൾക്ക് എന്ത് തരത്തിലുള്ള സഹாயം വേണം?",
    pa: "ਸਤ ਸ੍ਰੀ ਅਾਲ! ਮੈਂ SchemeSathi ਹਾਂ, ਤੁਹਾਡਾ AI ਸਹਾਇਕ। ਦੱਸੋ — ਤੁਹਾਨੂੰ ਕਿਸ ਤਰ੍ਹਾਂ ਦੀ ਮਦਦ ਚਾਹੀਦੀ ਹੈ?",
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

// ═══ Speech Recognition ═══════════════════════════════════
function initSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
        $('micBtn').style.display = 'none';
        return;
    }

    recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-IN';

    recognition.onresult = e => {
        const text = e.results[0][0].transcript;
        $('messageInput').value = text;
        $('sendBtn').disabled = false;
        sendMessage(text);
    };
    recognition.onerror = () => stopRecording();
    recognition.onend = () => stopRecording();
}

function toggleRecording() {
    isRecording ? stopRecording() : startRecording();
}

function startRecording() {
    if (!recognition) {
        alert('Voice input requires Chrome or Edge browser.');
        return;
    }
    recognition.lang = SPEECH_LANG_MAP[selectedLanguage] || 'en-IN';
    try { recognition.start(); } catch(e) {}
    isRecording = true;
    $('micBtn').classList.add('recording');
    $('recordingPulse').classList.remove('hidden');
}

function stopRecording() {
    if (recognition && isRecording) try { recognition.stop(); } catch(e) {}
    isRecording = false;
    $('micBtn').classList.remove('recording');
    $('recordingPulse').classList.add('hidden');
}

// ═══ Theme Management ══════════════════════════════════════
function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(currentTheme);
    localStorage.setItem('schemesathi_theme', currentTheme);
}

function applyTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        $('themeMoonIcon')?.classList.add('hidden');
        $('themeSunIcon')?.classList.remove('hidden');
    } else {
        document.documentElement.removeAttribute('data-theme');
        $('themeSunIcon')?.classList.add('hidden');
        $('themeMoonIcon')?.classList.remove('hidden');
    }
}

// ═══ Text-to-Speech ═══════════════════════════════════════
function toggleAutoSpeak() {
    autoSpeak = !autoSpeak;
    $('audioOnIcon').classList.toggle('hidden', !autoSpeak);
    $('audioOffIcon').classList.toggle('hidden', autoSpeak);
    $('audioToggleBtn').classList.toggle('active', autoSpeak);
}

function speakText(text) {
    if (!synth) return;
    synth.cancel();
    const clean = text
        .replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*(.*?)\*/g, '$1')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/#{1,6}\s/g, '').replace(/https?:\/\/[^\s]+/g, '')
        .replace(/[-*•]\s/g, '').trim();

    const utt = new SpeechSynthesisUtterance(clean);
    utt.lang = SPEECH_LANG_MAP[selectedLanguage] || 'en-IN';
    utt.rate = 0.9;
    synth.speak(utt);
}

function speakSingle(btn, text) {
    if (btn.classList.contains('speaking')) {
        synth.cancel(); btn.classList.remove('speaking'); return;
    }
    document.querySelectorAll('.speak-btn').forEach(b => b.classList.remove('speaking'));
    btn.classList.add('speaking');
    speakText(text);
    const poll = setInterval(() => {
        if (!synth.speaking) { btn.classList.remove('speaking'); clearInterval(poll); }
    }, 200);
}

// ═══ Chat Session ═════════════════════════════════════════
async function createSession() {
    try {
        const r = await fetch(`${API_BASE}/api/chat/new`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }
        });
        if (r.ok) sessionId = (await r.json()).session_id;
    } catch(e) {
        console.error('Session error:', e);
        sessionId = 'local-' + Date.now();
    }
}

async function startNewChat() {
    if (synth) synth.cancel();
    await createSession();

    $('messagesContainer').innerHTML = '';
    $('messagesContainer').classList.add('hidden');
    $('languageScreen').classList.remove('hidden');
    $('messageInput').value = '';
    $('messageInput').style.height = 'auto';
    $('sendBtn').disabled = true;
    isWaiting = false;
    $('typingIndicator').classList.add('hidden');
    document.querySelectorAll('.lang-chip').forEach(c => c.classList.remove('selected'));
}

function handleSubmit(e) {
    if (e) e.preventDefault();
    sendMessage();
}

async function sendMessage(text = null) {
    if (isWaiting) return;
    const msg = text || $('messageInput').value.trim();
    if (!msg) return;
    if (!sessionId) await createSession();

    $('languageScreen').classList.add('hidden');
    $('messagesContainer').classList.remove('hidden');

    renderUserMessage(msg);
    $('messageInput').value = '';
    $('messageInput').style.height = 'auto';
    $('sendBtn').disabled = true;
    isWaiting = true;
    showTyping();

    try {
        const r = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message: msg, language: selectedLanguage })
        });
        if (!r.ok) throw new Error('Request failed');
        const data = await r.json();
        hideTyping();
        renderAIMessage(data.reply, data.schemes || []);
        if (autoSpeak) speakText(data.reply);
    } catch(e) {
        hideTyping();
        renderAIMessage('Sorry, something went wrong. Please try again.', []);
    } finally {
        isWaiting = false;
    }
}

// ═══ Message Rendering ════════════════════════════════════
function renderUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user-message';
    div.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    $('messagesContainer').appendChild(div);
    scrollBottom();
}

function renderAIMessage(text, schemes) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message ai-message';

    const header = document.createElement('div');
    header.className = 'ai-header';
    header.innerHTML = `
        <div class="ai-avatar-sm">S</div>
        <span class="ai-name">SchemeSathi</span>
    `;
    const speakBtn = document.createElement('button');
    speakBtn.className = 'speak-btn';
    speakBtn.title = 'Listen';
    speakBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>';
    speakBtn.onclick = () => speakSingle(speakBtn, text);
    header.appendChild(speakBtn);
    wrapper.appendChild(header);

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = renderMd(text);
    wrapper.appendChild(content);

    if (schemes?.length > 0) {
        const grid = document.createElement('div');
        grid.className = 'scheme-cards-grid';
        schemes.forEach(s => grid.appendChild(buildCard(s)));
        wrapper.appendChild(grid);
    }

    $('messagesContainer').appendChild(wrapper);
    scrollBottom();
}

function buildCard(scheme) {
    const card = document.createElement('div');
    card.className = 'scheme-card';
    const lvl = (scheme.level || 'central').toLowerCase();
    card.innerHTML = `
        <span class="level-badge ${lvl}">${scheme.level || 'Central'}</span>
        <h4>${escapeHtml(scheme.name)}</h4>
        ${scheme.brief ? `<p>${escapeHtml(scheme.brief)}</p>` : ''}
        ${scheme.tags?.length ? `<div class="tags">${scheme.tags.slice(0, 3).map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>` : ''}
    `;
    const btn = document.createElement('button');
    btn.className = 'btn-details';
    btn.textContent = 'View Details →';
    btn.onclick = () => openDetail(scheme.slug);
    card.appendChild(btn);
    return card;
}

// ═══ Comprehensive Scheme Detail Modal ════════════════════
async function openDetail(slug) {
    $('schemeModal').classList.remove('hidden');
    $('schemeDetailContent').innerHTML = '<div class="loading-spinner"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';

    try {
        const r = await fetch(`${API_BASE}/api/schemes/${slug}`);
        if (!r.ok) throw new Error();
        const s = await r.json();

        let html = `
            <div class="modal-header">
                <span class="level-badge ${(s.level||'central').toLowerCase()}">${s.level||'Central'}</span>
                <h2>${escapeHtml(s.name)}</h2>
                ${s.ministry ? `<span class="ministry-tag">🏛️ ${escapeHtml(s.ministry)}</span>` : ''}
            </div>

            <!-- Overview Section -->
            <div class="detail-section">
                <h3>📋 Overview</h3>
                <p>${escapeHtml(s.brief || 'No summary available.')}</p>
                <div style="margin-top:0.75rem; display:flex; flex-wrap:wrap; gap:0.5rem;">
                    ${s.states?.length ? `<span class="tag"><strong>States:</strong> ${escapeHtml(s.states.join(', '))}</span>` : ''}
                    ${s.categories?.length ? `<span class="tag"><strong>Categories:</strong> ${escapeHtml(s.categories.join(', '))}</span>` : ''}
                </div>
            </div>
        `;

        // Detailed Description
        if (s.detailed_description) {
            html += `
                <div class="detail-section">
                    <h3>📖 Detailed Description</h3>
                    <div>${renderMd(s.detailed_description)}</div>
                </div>
            `;
        }

        // Benefits
        if (s.benefits) {
            html += `
                <div class="detail-section">
                    <h3>💰 Key Benefits</h3>
                    <div>${renderMd(s.benefits)}</div>
                </div>
            `;
        }

        // Eligibility
        if (s.eligibility) {
            html += `
                <div class="detail-section">
                    <h3>📝 Eligibility Criteria</h3>
                    <div>${renderMd(s.eligibility)}</div>
                </div>
            `;
        }

        // Application Process
        if (s.application_process && s.application_process.length > 0) {
            html += `<div class="detail-section"><h3>✅ Step-by-Step Application Process</h3>`;
            s.application_process.forEach(p => {
                if (!p) return;
                html += `<div style="margin-bottom:1rem; padding:0.5rem; border-left:3px solid var(--accent); background:var(--bg-elevated); border-radius:4px;">`;
                html += `<h4>Mode: ${escapeHtml(p.mode || 'Process')}</h4>`;
                if (p.url) html += `<a href="${p.url}" target="_blank" class="tag" style="display:inline-block; margin:0.4rem 0;">Apply Portal Link ↗</a>`;
                if (p.process_md) html += `<div>${renderMd(p.process_md)}</div>`;
                html += `</div>`;
            });
            html += `</div>`;
        }

        // Documents Required
        if (s.documents && s.documents.length > 0) {
            html += `
                <div class="detail-section">
                    <h3>📄 Documents Required</h3>
                    <ul>
                        ${s.documents.map(d => `<li>${escapeHtml(typeof d === 'string' ? d : d.document_name || d.name || JSON.stringify(d))}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // FAQs
        if (s.faqs && s.faqs.length > 0) {
            html += `
                <div class="detail-section">
                    <h3>❓ Frequently Asked Questions (FAQs)</h3>
                    ${s.faqs.map(faq => `
                        <div style="margin-bottom:0.75rem;">
                            <strong>Q: ${escapeHtml(faq.question || faq.q || '')}</strong>
                            <p style="margin-top:0.25rem;">${escapeHtml(faq.answer || faq.a || '')}</p>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Definitions
        if (s.definitions && s.definitions.length > 0) {
            html += `
                <div class="detail-section">
                    <h3>💡 Definitions & Terms</h3>
                    ${s.definitions.map(d => `
                        <div style="margin-bottom:0.5rem;">
                            <strong>${escapeHtml(d.name || '')}</strong>
                            <div>${renderMd(d.definition || '')}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Apply Button
        if (s.url) {
            html += `<a href="${s.url}" target="_blank" class="apply-btn">View Official Page on myScheme.gov.in →</a>`;
        }

        $('schemeDetailContent').innerHTML = html;

    } catch(e) {
        console.error("Modal detail error:", e);
        $('schemeDetailContent').innerHTML = '<div style="text-align:center;padding:3rem"><h3>Could not load details</h3><p style="color:var(--text-muted);margin-top:0.5rem">Please try again later.</p></div>';
    }
}

function closeModal() {
    $('schemeModal').classList.add('hidden');
}

// ═══ Utilities ════════════════════════════════════════════
function showTyping() { $('typingIndicator').classList.remove('hidden'); scrollBottom(); }
function hideTyping() { $('typingIndicator').classList.add('hidden'); }
function scrollBottom() { $('chatArea').scrollTop = $('chatArea').scrollHeight; }

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                       .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}

function renderMd(text) {
    if (!text) return '';
    let h = escapeHtml(text);
    // Strip any raw URLs if present so chat text remains clean
    h = h.replace(/https?:\/\/[^\s<)]+/g, '');
    h = h.replace(/^### (.*$)/gim, '<h4>$1</h4>');
    h = h.replace(/^## (.*$)/gim, '<h3>$1</h3>');
    h = h.replace(/^# (.*$)/gim, '<h2>$1</h2>');
    h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.*?)\*/g, '<em>$1</em>');
    h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<strong>$1</strong>');
    h = h.replace(/^\s*\d+\.\s(.*)$/gim, '<li>$1</li>');
    h = h.replace(/^\s*[-*]\s(.*)$/gim, '<li>$1</li>');
    h = h.replace(/\n\n/g, '</p><p>');
    h = h.replace(/\n/g, '<br>');
    return `<p>${h}</p>`.replace(/<p><\/p>/g, '');
}
