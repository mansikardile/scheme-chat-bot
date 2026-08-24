// ═══════════════════════════════════════════════════════════════
//  SchemeSathi — Frontend Application
//  Premium Edition with Multilingual Support, Voice I/O, Theme Toggle,
//  and Dynamic Local/Remote LLM Model Management
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

// Default built-in model is Gemini Flash
const DEFAULT_MODELS = [
    {
        id: 'gemini-flash',
        name: 'Google Gemini Flash',
        provider: 'gemini',
        is_default: true,
        model_name: 'gemini-flash-latest',
        description: 'Default Google Cloud AI model'
    }
];

// Custom models loaded from localStorage
let customModels = [];
try {
    const savedCustom = localStorage.getItem('schemesathi_custom_models');
    if (savedCustom) customModels = JSON.parse(savedCustom);
} catch (e) {
    customModels = [];
}

let selectedModel = localStorage.getItem('schemesathi_selected_model') ||
                    sessionStorage.getItem('schemesathi_selected_model') || 'gemini-flash';
let userApiKey = localStorage.getItem('schemesathi_api_key') ||
                 sessionStorage.getItem('schemesathi_api_key') || '';
let localOllamaModels = [];


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
    pa: "ਸਤ ਸ੍ਰੀ ਅਾਲ! ਮੈਂ SchemeSathi ਹਾਂ, ਤੁਹਾਡਾ AI ਸਹਾਇਕ। ਦੱਸੋ — ਤੁਹਾਨੂੰ ਕਿਸ ਤਰ੍ਹਾਂ ਦੀ ਮਦਦ ਚਾਹੀਦੀ ਹੈ?",
    or: "ନମସ୍କାର! ମୁଁ SchemeSathi, ଆପଣଙ୍କ AI ସହାଯ଼କ। କୁହନ୍ତୁ — ଆପଣଙ୍କୁ କେଉଁ ପ୍ରକାର ସାହାଯ଼୍ଯ ଦରକାର?",
    ur: "السلام علیکم! میں SchemeSathi ہوں, آپ کا AI معاون। بتائیں — آپ کو کس قسم کی مدد چاہیے؟"
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
    $('settingsBtn')?.addEventListener('click', openSettingsModal);

    // Language chips
    document.querySelectorAll('.lang-chip').forEach(chip => {
        chip.addEventListener('click', () => selectLanguage(chip));
    });

    // Scheme Modal
    document.querySelector('#schemeModal .modal-backdrop')?.addEventListener('click', closeModal);
    $('closeModalBtn')?.addEventListener('click', closeModal);

    // Settings Modal
    $('closeSettingsBtn')?.addEventListener('click', closeSettingsModal);
    $('btnCancelSettings')?.addEventListener('click', closeSettingsModal);
    $('btnSaveSettings')?.addEventListener('click', handleSaveSettings);
    $('btnShowAddModel')?.addEventListener('click', showAddModelPanel);

    // Keyboard shortcut: Escape closes modals
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            closeModal();
            closeSettingsModal();
        }
    });

    // Header Model Change Listener
    $('headerModelSelect')?.addEventListener('change', e => onModelChange(e.target.value));

    // Populate header model select
    populateModelSelects();

    // Init speech
    initSpeechRecognition();

    // Scan local Ollama in background
    scanLocalOllamaModels();

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
    }, 200);
}

// ═══ Session Management ════════════════════════════════════
async function createSession() {
    try {
        const r = await fetch(`${API_BASE}/api/chat/new`, { method: 'POST' });
        const data = await r.json();
        sessionId = data.session_id;
    } catch {
        sessionId = 'session_' + Date.now();
    }
}

async function startNewChat() {
    if (synth.speaking) synth.cancel();
    await createSession();
    $('messagesContainer').innerHTML = '';
    $('messagesContainer').classList.add('hidden');
    $('languageScreen').classList.remove('hidden');
    $('messageInput').value = '';
    $('messageInput').style.height = 'auto';
    $('sendBtn').disabled = true;
    $('typingIndicator').classList.add('hidden');
    document.querySelectorAll('.lang-chip').forEach(c => c.classList.remove('selected'));
}

// ═══ Theme Management ══════════════════════════════════════
function applyTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('schemesathi_theme', theme);

    const moon = $('themeMoonIcon');
    const sun = $('themeSunIcon');
    if (theme === 'light') {
        moon?.classList.add('hidden');
        sun?.classList.remove('hidden');
    } else {
        sun?.classList.add('hidden');
        moon?.classList.remove('hidden');
    }
}

function toggleTheme() {
    applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
}

// ═══ Speech & Voice ════════════════════════════════════════
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        if ($('micBtn')) $('micBtn').style.display = 'none';
        return;
    }
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = SPEECH_LANG_MAP[selectedLanguage] || 'en-IN';

    recognition.onstart = () => {
        isRecording = true;
        $('micBtn')?.classList.add('recording');
    };
    recognition.onresult = e => {
        const text = e.results[0][0].transcript;
        $('messageInput').value = text;
        autoResize($('messageInput'));
        $('sendBtn').disabled = false;
        sendMessage(text);
    };
    recognition.onerror = () => stopRecording();
    recognition.onend = () => stopRecording();
}

function toggleRecording() {
    if (!recognition) return;
    if (isRecording) {
        recognition.stop();
    } else {
        try {
            recognition.lang = SPEECH_LANG_MAP[selectedLanguage] || 'en-IN';
            recognition.start();
        } catch (e) {
            console.error('Speech recognition error:', e);
        }
    }
}

function stopRecording() {
    isRecording = false;
    $('micBtn')?.classList.remove('recording');
}

function toggleAutoSpeak() {
    autoSpeak = !autoSpeak;
    $('audioOffIcon')?.classList.toggle('hidden', autoSpeak);
    $('audioOnIcon')?.classList.toggle('hidden', !autoSpeak);
    $('audioToggleBtn')?.classList.toggle('active', autoSpeak);
    if (!autoSpeak && synth.speaking) synth.cancel();
}

function speakText(text) {
    if (!synth) return;
    synth.cancel();
    const clean = text.replace(/<[^>]*>/g, '').replace(/[*_#`[\]()]/g, '');
    const utter = new SpeechSynthesisUtterance(clean);
    utter.lang = SPEECH_LANG_MAP[selectedLanguage] || 'en-IN';
    utter.rate = 0.95;
    synth.speak(utter);
}

function speakSingle(btn, text) {
    if (!synth) return;
    if (synth.speaking) {
        synth.cancel();
        btn.classList.remove('speaking');
        return;
    }
    const clean = text.replace(/<[^>]*>/g, '').replace(/[*_#`[\]()]/g, '');
    const utter = new SpeechSynthesisUtterance(clean);
    utter.lang = SPEECH_LANG_MAP[selectedLanguage] || 'en-IN';
    utter.rate = 0.95;
    btn.classList.add('speaking');
    utter.onend = () => btn.classList.remove('speaking');
    utter.onerror = () => btn.classList.remove('speaking');
    synth.speak(utter);
}

// ═══ Chat Submission & Streaming ═══════════════════════════
function handleSubmit(e) {
    if (e) e.preventDefault();
    sendMessage();
}

async function sendMessage(text = null) {
    if (isWaiting) return;
    const msg = text || $('messageInput').value.trim();
    if (!msg) return;
    if (!sessionId) await createSession();

    // Get selected model config object
    const allModels = getAllModels();
    const activeModelObj = allModels.find(m => m.id === selectedModel) || DEFAULT_MODELS[0];

    $('languageScreen').classList.add('hidden');
    $('messagesContainer').classList.remove('hidden');

    renderUserMessage(msg);
    $('messageInput').value = '';
    $('messageInput').style.height = 'auto';
    $('sendBtn').disabled = true;
    isWaiting = true;
    showTyping();

    try {
        const payload = {
            session_id: sessionId,
            message: msg,
            language: selectedLanguage,
            model: activeModelObj.id,
            api_key: activeModelObj.is_default ? (userApiKey || null) : (activeModelObj.api_key || userApiKey || null),
            model_config: activeModelObj.is_default ? null : activeModelObj
        };

        const r = await fetch(`${API_BASE}/api/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!r.ok) throw new Error('Request failed');

        hideTyping();
        const reader = r.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let messageObj = null;
        let fullText = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep partial line in buffer

            for (const line of lines) {
                const cleaned = line.trim();
                if (!cleaned.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(cleaned.substring(6));
                    if (data.type === 'text') {
                        if (!messageObj) {
                            messageObj = createStreamingAIMessage();
                        }
                        fullText += data.content;
                        messageObj.contentDiv.innerHTML = renderMd(fullText);
                        scrollBottom();
                    } else if (data.type === 'cards') {
                        if (!messageObj) {
                            messageObj = createStreamingAIMessage();
                        }
                        if (data.content && data.content.length > 0) {
                            const grid = document.createElement('div');
                            grid.className = 'scheme-cards-grid';
                            data.content.forEach(s => grid.appendChild(buildCard(s)));
                            messageObj.wrapper.appendChild(grid);
                            scrollBottom();
                        }
                    } else if (data.type === 'error') {
                        throw new Error(data.content);
                    }
                } catch (e) {
                    console.error('Failed to parse SSE line:', e);
                }
            }
        }

        // Finalize speech & speaker button
        if (messageObj && fullText) {
            messageObj.speakBtn.style.display = 'inline-flex';
            messageObj.speakBtn.onclick = () => speakSingle(messageObj.speakBtn, fullText);
            if (autoSpeak) speakText(fullText);
        }
    } catch(e) {
        hideTyping();
        renderAIMessage('Sorry, something went wrong with the model. Please check your model settings.', []);
    } finally {
        isWaiting = false;
        $('sendBtn').disabled = false;
    }
}

// ═══ Model Management & Settings ═══════════════════════════
function getAllModels() {
    return [...DEFAULT_MODELS, ...customModels];
}

// Track which tab is active in settings (model id, or 'add')
let activeSettingsTab = null;

function populateModelSelects() {
    const allModels = getAllModels();
    const headerSelect = $('headerModelSelect');
    if (!headerSelect) return;

    const exists = allModels.some(m => m.id === selectedModel);
    if (!exists && allModels.length > 0) {
        selectedModel = allModels[0].id;
    }

    const optionsHtml = allModels.map(m => {
        let badge = m.is_default ? '☁️ Default' : (m.is_local ? '🟢 Local Ollama' : '🌐 Remote');
        return `<option value="${m.id}">${escapeHtml(m.name)} (${badge})</option>`;
    }).join('');

    if (headerSelect) {
        headerSelect.innerHTML = optionsHtml;
        headerSelect.value = selectedModel;
    }

    updateModelUI();
}

function onModelChange(newModelId) {
    selectedModel = newModelId;
    localStorage.setItem('schemesathi_selected_model', selectedModel);
    sessionStorage.setItem('schemesathi_selected_model', selectedModel);

    if ($('headerModelSelect')) $('headerModelSelect').value = selectedModel;

    updateModelUI();
}

function updateModelUI() {
    const allModels = getAllModels();
    const model = allModels.find(m => m.id === selectedModel) || DEFAULT_MODELS[0];
    const headerBadge = $('headerModelBadge');
    if (headerBadge) {
        const isLocal = model.is_local;
        const isDefault = model.is_default;
        headerBadge.className = `header-model-badge ${isDefault ? 'badge-cloud' : (isLocal ? 'badge-local' : 'badge-server')}`;
        headerBadge.textContent = isDefault ? 'Cloud' : (isLocal ? 'Local' : 'Remote');
    }
}

// ── Sidebar rendering ──
function renderModelSidebar() {
    const list = $('modelTabList');
    if (!list) return;

    const allModels = getAllModels();
    let html = `<div class="settings-sidebar-label">LLM Models</div>`;

    html += allModels.map(m => {
        const isDefault = m.is_default;
        const badgeClass = isDefault ? 'badge-cloud' : (m.is_local ? 'badge-local' : 'badge-server');
        const badgeLabel = isDefault ? 'Cloud' : (m.is_local ? 'Local' : 'Remote');
        const isActive = activeSettingsTab === m.id ? 'active' : '';
        return `
            <div class="model-tab-item ${isActive}" data-model-id="${m.id}" onclick="selectModelTab('${m.id}')">
                <span class="model-tab-name">${escapeHtml(m.name)}</span>
                <span class="model-tab-badge header-model-badge ${badgeClass}">${badgeLabel}</span>
            </div>
        `;
    }).join('');

    const isEmbActive = activeSettingsTab === 'embedding' ? 'active' : '';
    html += `
        <div class="settings-sidebar-label" style="margin-top:1rem;">Embedding Model</div>
        <div class="model-tab-item ${isEmbActive}" data-model-id="embedding" onclick="selectModelTab('embedding')">
            <span class="model-tab-name">Embedding Model</span>
        </div>
    `;

    list.innerHTML = html;
}


window.selectModelTab = function(id) {
    if (activeSettingsTab && activeSettingsTab !== 'gemini-flash' && activeSettingsTab !== 'add' && activeSettingsTab !== 'embedding') {
        saveCurrentCustomModelEdits();
    }
    activeSettingsTab = id;
    renderModelSidebar();
    if (id === 'embedding') {
        renderEmbeddingDetailPanel();
    } else if (id === 'add') {
        showAddModelPanel();
    } else {
        renderModelDetailPanel(id);
    }
};

function saveCurrentCustomModelEdits() {
    if (!activeSettingsTab || activeSettingsTab === 'gemini-flash' || activeSettingsTab === 'add' || activeSettingsTab === 'embedding') {
        return true;
    }

    const m = customModels.find(item => item.id === activeSettingsTab);
    if (!m) return true;

    const nameInp = $('editModelName');
    if (!nameInp) return true;

    const name = nameInp.value.trim();
    if (!name) return false;

    const isLocal = $('editModelIsLocal')?.checked ?? true;
    let modelTag = '';
    let endpoint = '';
    let apiKey = '';

    if (isLocal) {
        modelTag = $('editModelLocalTag')?.value.trim() || $('editLocalDownloadedSelect')?.value;
        if (!modelTag) return false;
    } else {
        modelTag = $('editModelRemoteTag')?.value.trim();
        endpoint = $('editModelEndpoint')?.value.trim();
        apiKey = $('editModelApiKey')?.value.trim() || '';
        if (!modelTag || !endpoint) return false;
    }

    m.name = name;
    m.is_local = isLocal;
    m.model_name = modelTag;
    m.base_url = isLocal ? null : endpoint;
    m.api_key = isLocal ? null : apiKey;
    m.description = isLocal ? 'Local Ollama Model' : `Remote (${endpoint})`;

    localStorage.setItem('schemesathi_custom_models', JSON.stringify(customModels));

    // Update DOM indicators
    const headerName = $('editHeaderModelName');
    if (headerName) headerName.textContent = name;

    const headerTag = $('editHeaderModelTag');
    if (headerTag) headerTag.textContent = modelTag;

    const sidebarTab = document.querySelector(`.model-tab-item[data-model-id="${m.id}"] .model-tab-name`);
    if (sidebarTab) sidebarTab.textContent = name;

    const sidebarBadge = document.querySelector(`.model-tab-item[data-model-id="${m.id}"] .model-tab-badge`);
    if (sidebarBadge) {
        sidebarBadge.className = `model-tab-badge header-model-badge ${isLocal ? 'badge-local' : 'badge-server'}`;
        sidebarBadge.textContent = isLocal ? 'Local' : 'Remote';
    }

    return true;
}

function renderModelDetailPanel(id) {
    const panel = $('settingsMainPanel');
    if (!panel) return;

    const allModels = getAllModels();
    const m = allModels.find(m => m.id === id);
    if (!m) return;

    const isDefault = m.is_default;
    const badgeClass = isDefault ? 'badge-cloud' : (m.is_local ? 'badge-local' : 'badge-server');
    const badgeLabel = isDefault ? 'Cloud (Default)' : (m.is_local ? 'Local Ollama' : 'Remote Endpoint');

    let html = `
        <div class="model-detail-panel">
            <div class="model-detail-header">
                <div>
                    <h2 class="model-detail-name" id="editHeaderModelName">${escapeHtml(m.name)}</h2>
                    <div class="model-detail-meta">
                        <span class="header-model-badge ${badgeClass}" id="editHeaderModelBadge">${badgeLabel}</span>
                        <span class="model-detail-tag" id="editHeaderModelTag">${escapeHtml(m.model_name || m.id)}</span>
                    </div>
                </div>
                ${!isDefault ? `
                <button type="button" class="btn-delete-model-detail" onclick="deleteCustomModel('${m.id}')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                    Delete Model
                </button>` : ''}
            </div>
    `;

    // Gemini default model: show API key section
    if (isDefault && m.provider === 'gemini') {
        const savedKey = userApiKey || '';
        html += `
            <div class="setting-section">
                <label for="settingsApiKeyInput" class="section-label">Gemini API Key <span class="label-optional">(Optional)</span></label>
                <p class="setting-desc">SchemeSathi uses Google Gemini Flash by default. Enter your own API key if you hit rate limits.</p>
                <div class="api-key-input-wrapper">
                    <input type="password" id="settingsApiKeyInput" class="settings-input" placeholder="Leave empty to use server default key" value="${escapeHtml(savedKey)}">
                    <button type="button" id="toggleApiKeyBtn" class="btn-toggle-eye" title="Show / Hide Key" aria-label="Toggle API key visibility">
                        <svg id="eyeIconShow" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                        <svg id="eyeIconHide" class="hidden" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"></path>
                            <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"></path>
                            <line x1="1" y1="1" x2="23" y2="23"></line>
                        </svg>
                    </button>
                </div>
            </div>
        `;
    } else if (!isDefault) {
        // Custom model — editable form fields directly visible
        html += `
            <div class="form-grid">
                <div class="form-group">
                    <label for="editModelName">Model Display Name *</label>
                    <input type="text" id="editModelName" class="settings-input" value="${escapeHtml(m.name)}" placeholder="e.g. My Local Gemma, DeepSeek R1">
                </div>

                <div class="form-group">
                    <label class="checkbox-container" for="editModelIsLocal">
                        <input type="checkbox" id="editModelIsLocal" ${m.is_local ? 'checked' : ''}>
                        <span class="checkbox-box"></span>
                        <span class="checkbox-label-text">
                            <strong>Local Model (uses Ollama)</strong>
                            <small>Runs directly on your machine at localhost:11434</small>
                        </span>
                    </label>
                </div>

                <!-- Local Ollama options -->
                <div id="editLocalOllamaOptions" class="form-group local-options-box ${m.is_local ? '' : 'hidden'}">
                    <div class="local-select-header">
                        <label for="editLocalDownloadedSelect">Select Downloaded Model</label>
                        <button type="button" id="btnRefreshEditOllama" class="btn-refresh-ollama" title="Scan local Ollama models">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                            <span>Scan Models</span>
                        </button>
                    </div>
                    <select id="editLocalDownloadedSelect" class="settings-select">
                        <option value="">Scanning local models...</option>
                    </select>
                    <div class="manual-tag-wrap">
                        <label for="editModelLocalTag">Or enter Model Tag manually *</label>
                        <input type="text" id="editModelLocalTag" class="settings-input" value="${escapeHtml(m.is_local ? m.model_name || '' : '')}" placeholder="e.g. gemma3:4b, llama3:8b">
                    </div>
                </div>

                <!-- Remote options -->
                <div id="editRemoteModelOptions" class="form-group remote-options-box ${!m.is_local ? '' : 'hidden'}">
                    <div class="form-group">
                        <label for="editModelRemoteTag">Model Tag / Identifier *</label>
                        <input type="text" id="editModelRemoteTag" class="settings-input" value="${escapeHtml(!m.is_local ? m.model_name || '' : '')}" placeholder="e.g. gemma3:4b, llama3:8b">
                    </div>
                    <div class="form-group">
                        <label for="editModelEndpoint">Endpoint URL *</label>
                        <input type="url" id="editModelEndpoint" class="settings-input" value="${escapeHtml(m.base_url || '')}" placeholder="e.g. https://ai.example.com">
                    </div>
                    <div class="form-group">
                        <label for="editModelApiKey">API Key (Optional)</label>
                        <div class="api-key-input-wrapper">
                            <input type="password" id="editModelApiKey" class="settings-input" value="${escapeHtml(m.api_key || '')}" placeholder="Authorization bearer token or key">
                            <button type="button" id="toggleEditModelKeyBtn" class="btn-toggle-eye" title="Show/Hide">
                                <svg id="editKeyIconShow" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                    <circle cx="12" cy="12" r="3"></circle>
                                </svg>
                                <svg id="editKeyIconHide" class="hidden" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"></path>
                                    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"></path>
                                    <line x1="1" y1="1" x2="23" y2="23"></line>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Built-in lock notice
    if (isDefault) {
        html += `
            <div class="built-in-notice">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
                This is a built-in model and cannot be deleted.
            </div>
        `;
    }

    html += `</div>`;
    panel.innerHTML = html;

    // Wire up Gemini API key eye toggle
    const toggleBtn = $('toggleApiKeyBtn');
    const apiInput = $('settingsApiKeyInput');
    if (toggleBtn && apiInput) {
        toggleBtn.addEventListener('click', () => {
            const isHidden = apiInput.type === 'password';
            apiInput.type = isHidden ? 'text' : 'password';
            $('eyeIconShow')?.classList.toggle('hidden', isHidden);
            $('eyeIconHide')?.classList.toggle('hidden', !isHidden);
        });
    }

    // Wire up Custom Model Edit fields
    if (!isDefault) {
        $('editModelIsLocal')?.addEventListener('change', (e) => {
            const isLocal = e.target.checked;
            $('editLocalOllamaOptions')?.classList.toggle('hidden', !isLocal);
            $('editRemoteModelOptions')?.classList.toggle('hidden', isLocal);
            const badge = $('editHeaderModelBadge');
            if (badge) {
                badge.className = `header-model-badge ${isLocal ? 'badge-local' : 'badge-server'}`;
                badge.textContent = isLocal ? 'Local Ollama' : 'Remote Endpoint';
            }
            saveCurrentCustomModelEdits();
        });

        $('btnRefreshEditOllama')?.addEventListener('click', scanLocalOllamaModels);

        $('editLocalDownloadedSelect')?.addEventListener('change', (e) => {
            const val = e.target.value;
            if (val && $('editModelLocalTag')) {
                $('editModelLocalTag').value = val;
                saveCurrentCustomModelEdits();
            }
        });

        const editKeyInput = $('editModelApiKey');
        $('toggleEditModelKeyBtn')?.addEventListener('click', () => {
            if (editKeyInput) {
                const isHidden = editKeyInput.type === 'password';
                editKeyInput.type = isHidden ? 'text' : 'password';
                $('editKeyIconShow')?.classList.toggle('hidden', isHidden);
                $('editKeyIconHide')?.classList.toggle('hidden', !isHidden);
            }
        });

        ['editModelName', 'editModelLocalTag', 'editModelRemoteTag', 'editModelEndpoint', 'editModelApiKey'].forEach(inputId => {
            $(inputId)?.addEventListener('input', saveCurrentCustomModelEdits);
            $(inputId)?.addEventListener('change', saveCurrentCustomModelEdits);
        });

        if (m.is_local) {
            if (localOllamaModels.length > 0) {
                const select = $('editLocalDownloadedSelect');
                if (select) {
                    select.innerHTML = '<option value="">-- Select Downloaded Model (' + localOllamaModels.length + ' found) --</option>' +
                        localOllamaModels.map(lm => `<option value="${escapeHtml(lm.name)}" ${lm.name === m.model_name ? 'selected' : ''}>${escapeHtml(lm.name)} ${lm.size ? '(' + lm.size + ')' : ''}</option>`).join('');
                }
            } else {
                scanLocalOllamaModels();
            }
        }
    }
}

window.toggleRemoteKey = function(btn) {
    const inp = $('remoteModelKeyDisplay');
    if (inp) inp.type = inp.type === 'password' ? 'text' : 'password';
};

// ── Show Add Model form in right panel ──
function showAddModelPanel() {
    activeSettingsTab = 'add';
    renderModelSidebar();

    const panel = $('settingsMainPanel');
    if (!panel) return;

    panel.innerHTML = `
        <div class="add-model-form-panel">
            <h2 class="add-model-form-title">Add New Model</h2>

            <div class="form-grid">
                <div class="form-group">
                    <label for="newModelName">Model Display Name *</label>
                    <input type="text" id="newModelName" class="settings-input" placeholder="e.g. My Local Gemma, DeepSeek R1">
                </div>

                <div class="form-group">
                    <label class="checkbox-container" for="newModelIsLocal">
                        <input type="checkbox" id="newModelIsLocal" checked>
                        <span class="checkbox-box"></span>
                        <span class="checkbox-label-text">
                            <strong>Local Model (uses Ollama)</strong>
                            <small>Runs directly on your machine at localhost:11434</small>
                        </span>
                    </label>
                </div>

                <!-- Local Ollama options -->
                <div id="localOllamaOptions" class="form-group local-options-box">
                    <div class="local-select-header">
                        <label for="localDownloadedSelect">Select Downloaded Model</label>
                        <button type="button" id="btnRefreshOllama" class="btn-refresh-ollama" title="Scan local Ollama models">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                            <span>Scan Models</span>
                        </button>
                    </div>
                    <select id="localDownloadedSelect" class="settings-select">
                        <option value="">Scanning local models...</option>
                    </select>
                    <div class="manual-tag-wrap">
                        <label for="newModelLocalTag">Or enter Model Tag manually *</label>
                        <input type="text" id="newModelLocalTag" class="settings-input" placeholder="e.g. gemma3:4b, llama3:8b">
                    </div>
                </div>

                <!-- Remote options -->
                <div id="remoteModelOptions" class="form-group remote-options-box hidden">
                    <div class="form-group">
                        <label for="newModelRemoteTag">Model Tag / Identifier *</label>
                        <input type="text" id="newModelRemoteTag" class="settings-input" placeholder="e.g. gemma3:4b, llama3:8b">
                    </div>
                    <div class="form-group">
                        <label for="newModelEndpoint">Endpoint URL *</label>
                        <input type="url" id="newModelEndpoint" class="settings-input" placeholder="e.g. https://ai.example.com">
                    </div>
                    <div class="form-group">
                        <label for="newModelApiKey">API Key (Optional)</label>
                        <div class="api-key-input-wrapper">
                            <input type="password" id="newModelApiKey" class="settings-input" placeholder="Authorization bearer token or key">
                            <button type="button" id="toggleNewModelKeyBtn" class="btn-toggle-eye" title="Show/Hide">
                                <svg id="newKeyIconShow" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                    <circle cx="12" cy="12" r="3"></circle>
                                </svg>
                                <svg id="newKeyIconHide" class="hidden" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"></path>
                                    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"></path>
                                    <line x1="1" y1="1" x2="23" y2="23"></line>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>

                <div class="form-actions-row">
                    <button type="button" id="btnSubmitAddModel" class="btn-save-model">Save Model</button>
                </div>
            </div>
        </div>
    `;

    // Wire events for dynamically rendered form
    $('newModelIsLocal')?.addEventListener('change', onNewModelIsLocalChange);
    $('btnRefreshOllama')?.addEventListener('click', scanLocalOllamaModels);
    $('localDownloadedSelect')?.addEventListener('change', onLocalDownloadedModelSelect);
    $('btnSubmitAddModel')?.addEventListener('click', handleAddModelSubmit);

    const newKeyInput = $('newModelApiKey');
    $('toggleNewModelKeyBtn')?.addEventListener('click', () => {
        if (newKeyInput) {
            const isHidden = newKeyInput.type === 'password';
            newKeyInput.type = isHidden ? 'text' : 'password';
            $('newKeyIconShow')?.classList.toggle('hidden', isHidden);
            $('newKeyIconHide')?.classList.toggle('hidden', !isHidden);
        }
    });

    // Scan Ollama in background
    scanLocalOllamaModels();
}

// ── Add Model Form Handlers ──
function onNewModelIsLocalChange(e) {
    const isLocal = e.target.checked;
    $('localOllamaOptions')?.classList.toggle('hidden', !isLocal);
    $('remoteModelOptions')?.classList.toggle('hidden', isLocal);
}

async function scanLocalOllamaModels() {
    const refreshBtn = $('btnRefreshOllama');
    const select = $('localDownloadedSelect');
    if (refreshBtn) refreshBtn.classList.add('spinning');
    if (select) select.innerHTML = '<option value="">Scanning local Ollama...</option>';

    try {
        const res = await fetch(`${API_BASE}/api/models/local-ollama`);
        if (res.ok) {
            const data = await res.json();
            if (data.running && data.models?.length > 0) {
                localOllamaModels = data.models;
                if (select) select.innerHTML = '<option value="">-- Select Downloaded Model (' + data.models.length + ' found) --</option>' +
                    data.models.map(m => `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)} ${m.size ? '(' + m.size + ')' : ''}</option>`).join('');
            } else if (data.running) {
                if (select) select.innerHTML = '<option value="">Ollama is running (No downloaded models found)</option>';
            } else {
                if (select) select.innerHTML = '<option value="">Ollama daemon offline (enter tag below)</option>';
            }
        } else {
            if (select) select.innerHTML = '<option value="">Could not check Ollama</option>';
        }
    } catch {
        if (select) select.innerHTML = '<option value="">Ollama offline or unreachable</option>';
    } finally {
        if (refreshBtn) refreshBtn.classList.remove('spinning');
    }
}

function onLocalDownloadedModelSelect(e) {
    const val = e.target.value;
    if (val && $('newModelLocalTag')) {
        $('newModelLocalTag').value = val;
        if (!$('newModelName').value.trim()) {
            const clean = val.replace(':', ' (').replace(/$/, val.includes(':') ? ')' : '');
            $('newModelName').value = clean.charAt(0).toUpperCase() + clean.slice(1);
        }
    }
}

function handleAddModelSubmit() {
    const name = $('newModelName')?.value.trim();
    const isLocal = $('newModelIsLocal')?.checked ?? true;

    if (!name) {
        showToast('Please enter a Display Name for the model.');
        $('newModelName')?.focus();
        return;
    }

    let modelTag = '';
    let endpoint = '';
    let apiKey = '';

    if (isLocal) {
        modelTag = $('newModelLocalTag')?.value.trim() || $('localDownloadedSelect')?.value;
        if (!modelTag) {
            showToast('Please specify a local model tag (e.g. gemma3:4b).');
            $('newModelLocalTag')?.focus();
            return;
        }
    } else {
        modelTag = $('newModelRemoteTag')?.value.trim();
        endpoint = $('newModelEndpoint')?.value.trim();
        apiKey = $('newModelApiKey')?.value.trim() || '';

        if (!modelTag) {
            showToast('Please specify the Model Tag / Identifier.');
            $('newModelRemoteTag')?.focus();
            return;
        }
        if (!endpoint) {
            showToast('Please enter the Endpoint URL / Link.');
            $('newModelEndpoint')?.focus();
            return;
        }
    }

    const newModel = {
        id: (isLocal ? 'ollama-' : 'remote-') + Date.now(),
        name: name,
        provider: 'ollama',
        is_local: isLocal,
        model_name: modelTag,
        base_url: isLocal ? null : endpoint,
        api_key: isLocal ? null : apiKey,
        description: isLocal ? 'Local Ollama Model' : `Remote (${endpoint})`
    };

    customModels.push(newModel);
    localStorage.setItem('schemesathi_custom_models', JSON.stringify(customModels));

    selectedModel = newModel.id;
    localStorage.setItem('schemesathi_selected_model', selectedModel);

    // Select the new model's tab in the sidebar
    activeSettingsTab = newModel.id;
    renderModelSidebar();
    renderModelDetailPanel(newModel.id);

    populateModelSelects();
    showToast(`Model "${name}" added successfully!`);
}

window.deleteCustomModel = function(id) {
    const model = customModels.find(m => m.id === id);
    customModels = customModels.filter(m => m.id !== id);
    localStorage.setItem('schemesathi_custom_models', JSON.stringify(customModels));

    if (selectedModel === id) {
        selectedModel = 'gemini-flash';
        localStorage.setItem('schemesathi_selected_model', selectedModel);
    }

    // Switch sidebar to first available model after deletion
    activeSettingsTab = getAllModels()[0]?.id || null;
    renderModelSidebar();
    if (activeSettingsTab) renderModelDetailPanel(activeSettingsTab);

    populateModelSelects();
    showToast(`Model ${model?.name ? `"${model.name}"` : ''} deleted.`);
};

// ── Embedding Model Management ──
async function renderEmbeddingDetailPanel() {
    const panel = $('settingsMainPanel');
    if (!panel) return;

    panel.innerHTML = '<div class="loading-spinner" style="margin:4rem auto;"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';

    let embeddingInfo = null;
    try {
        const res = await fetch(`${API_BASE}/api/embedding/info`);
        if (res.ok) {
            embeddingInfo = await res.json();
        }
    } catch (e) {
        console.error('Failed to fetch embedding info:', e);
    }

    const dbInfo = embeddingInfo?.db_info || { embedding_model: 'bge-m3:latest' };
    const currentCfg = embeddingInfo?.current_config || { model: 'bge-m3:latest', is_local: true, base_url: '' };
    const localModels = embeddingInfo?.local_ollama?.models || [];

    const dbModelTag = dbInfo.embedding_model || 'bge-m3:latest';
    const activeModelTag = currentCfg.model || dbModelTag;
    const isLocal = currentCfg.is_local ?? true;

    let html = `
        <div class="model-detail-panel">
            <div class="model-detail-header">
                <div>
                    <h2 class="model-detail-name">Embedding Model Settings</h2>
                    <div class="model-detail-meta" style="margin-top:0.4rem;">
                        <span class="model-detail-tag" style="font-size:0.8rem; padding:3px 10px;">Vector DB trained with: <strong style="color:var(--accent); font-family:monospace;">${escapeHtml(dbModelTag)}</strong></span>
                    </div>
                </div>
            </div>



            <!-- Host Location Selection -->
            <div class="setting-section">
                <label class="section-label">Embedding Host Location</label>
                <p class="setting-desc">Select whether to run the embedding model on your local Ollama daemon or a remote endpoint.</p>
                <div class="radio-option-group">
                    <label class="radio-option-wrap">
                        <input type="radio" name="embHostChoice" value="local" ${isLocal ? 'checked' : ''} onchange="onEmbHostChange(true)">
                        <div class="radio-option-text">
                            <strong>Local Ollama Host</strong>
                            <small>Runs directly on your machine at <code>http://localhost:11434</code></small>
                        </div>
                    </label>
                    <label class="radio-option-wrap">
                        <input type="radio" name="embHostChoice" value="remote" ${!isLocal ? 'checked' : ''} onchange="onEmbHostChange(false)">
                        <div class="radio-option-text">
                            <strong>Remote Server Endpoint</strong>
                            <small>Use a remote server or API endpoint for embeddings</small>
                        </div>
                    </label>
                </div>
            </div>

            <!-- Remote Endpoint Options -->
            <div id="embRemoteSection" class="setting-section ${isLocal ? 'hidden' : ''}">
                <div class="form-group" style="margin-bottom:0.75rem;">
                    <label for="embEndpointUrl" class="section-label">Endpoint URL *</label>
                    <input type="url" id="embEndpointUrl" class="settings-input" value="${escapeHtml(currentCfg.base_url || '')}" placeholder="e.g. https://ai.11022006.xyz">
                </div>
                <div class="form-group">
                    <label for="embApiKey" class="section-label">API Key / Token (Optional)</label>
                    <input type="password" id="embApiKey" class="settings-input" value="${escapeHtml(currentCfg.api_key || '')}" placeholder="Authorization bearer key">
                </div>
            </div>

            <!-- Embedding Model Selection -->
            <div class="setting-section">
                <label for="embModelTagInput" class="section-label">Embedding Model Tag</label>
                <p class="setting-desc">The model identifier used to calculate scheme query embeddings.</p>

                <div id="embLocalModelWrapper" class="${isLocal && localModels.length > 0 ? '' : 'hidden'}">
                    <select id="embLocalModelSelect" class="settings-select" onchange="onEmbModelSelectChange(this.value)">
                        <option value="">-- Select Downloaded Local Model (${localModels.length} found) --</option>
                        ${localModels.map(m => {
                            const isSelected = m.name.toLowerCase() === activeModelTag.toLowerCase() || 
                                               m.name.toLowerCase().startsWith(activeModelTag.toLowerCase().split(':')[0]);
                            return `<option value="${escapeHtml(m.name)}" ${isSelected ? 'selected' : ''}>${escapeHtml(m.name)} ${m.size ? '(' + m.size + ')' : ''}</option>`;
                        }).join('')}
                    </select>
                </div>

                <div id="embTextInputWrapper" class="api-key-input-wrapper ${isLocal && localModels.length > 0 ? 'hidden' : ''}">
                    <input type="text" id="embModelTagInput" class="settings-input" value="${escapeHtml(activeModelTag)}" placeholder="e.g. bge-m3:latest">
                </div>
            </div>
        </div>
    `;

    panel.innerHTML = html;
}

window.onEmbHostChange = function(isLocal) {
    const remoteSec = $('embRemoteSection');
    const localSec = $('embLocalModelWrapper');
    const textSec = $('embTextInputWrapper');
    const localSelect = $('embLocalModelSelect');
    const hasLocalModels = localSelect && localSelect.options.length > 1;

    if (remoteSec) remoteSec.classList.toggle('hidden', isLocal);
    
    if (isLocal && hasLocalModels) {
        if (localSec) localSec.classList.remove('hidden');
        if (textSec) textSec.classList.add('hidden');
    } else {
        if (localSec) localSec.classList.add('hidden');
        if (textSec) textSec.classList.remove('hidden');
    }
};

window.onEmbModelSelectChange = function(val) {
    if (val && $('embModelTagInput')) {
        $('embModelTagInput').value = val;
    }
};

async function handleSaveEmbeddingSubmit() {
    const isLocal = document.querySelector('input[name="embHostChoice"]:checked')?.value === 'local';
    const localSelect = $('embLocalModelSelect');
    const localSec = $('embLocalModelWrapper');
    
    let modelTag = '';
    if (isLocal && localSelect && localSec && !localSec.classList.contains('hidden')) {
        modelTag = localSelect.value || $('embModelTagInput')?.value?.trim() || 'bge-m3:latest';
    } else {
        modelTag = $('embModelTagInput')?.value?.trim() || 'bge-m3:latest';
    }
    
    const endpoint = $('embEndpointUrl')?.value?.trim() || '';
    const apiKey = $('embApiKey')?.value?.trim() || '';

    if (!isLocal && !endpoint) {
        showToast('Please enter the Remote Endpoint URL.');
        $('embEndpointUrl')?.focus();
        return;
    }

    try {
        const payload = {
            model: modelTag,
            is_local: isLocal,
            base_url: isLocal ? null : endpoint,
            api_key: isLocal ? null : apiKey
        };

        const res = await fetch(`${API_BASE}/api/embedding/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            showToast(`Embedding settings saved (${modelTag})!`);
            renderEmbeddingDetailPanel();
        } else {
            showToast('Failed to save embedding configuration.');
        }
    } catch (e) {
        console.error('Save embedding error:', e);
        showToast('Error saving embedding configuration.');
    }
}



// ── Save & Apply Settings ──
function handleSaveSettings() {
    if (activeSettingsTab === 'embedding') {
        handleSaveEmbeddingSubmit();
        closeSettingsModal();
        return;
    }

    // Read API key if the Gemini model detail panel is currently showing it
    const apiKeyInput = $('settingsApiKeyInput');
    if (apiKeyInput) {
        userApiKey = apiKeyInput.value.trim();
        localStorage.setItem('schemesathi_api_key', userApiKey);
        sessionStorage.setItem('schemesathi_api_key', userApiKey);
    }

    // Save custom model edits if viewing a custom model tab
    if (activeSettingsTab && activeSettingsTab !== 'gemini-flash' && activeSettingsTab !== 'add' && activeSettingsTab !== 'embedding') {
        const ok = saveCurrentCustomModelEdits();
        if (!ok) {
            showToast('Please fill in all required model fields (Name and Model Tag / Endpoint).');
            return;
        }
        selectedModel = activeSettingsTab;
    } else if (activeSettingsTab && activeSettingsTab !== 'add' && activeSettingsTab !== 'embedding') {
        selectedModel = activeSettingsTab;
    }

    localStorage.setItem('schemesathi_selected_model', selectedModel);
    sessionStorage.setItem('schemesathi_selected_model', selectedModel);

    populateModelSelects();
    closeSettingsModal();
    showToast('Settings saved & applied!');
}


function showToast(msg, duration = 3000) {
    const toast = $('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.remove('hidden');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
        toast.classList.add('hidden');
    }, duration);
}

function openSettingsModal() {
    // Default to the currently active model's tab
    activeSettingsTab = selectedModel || getAllModels()[0]?.id || null;
    renderModelSidebar();
    if (activeSettingsTab) renderModelDetailPanel(activeSettingsTab);
    $('settingsModal')?.classList.remove('hidden');
}

function closeSettingsModal() {
    $('settingsModal')?.classList.add('hidden');
}



// ═══ Message Rendering ════════════════════════════════════
function renderUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user-message';
    div.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    $('messagesContainer').appendChild(div);
    scrollBottom();
}

function createStreamingAIMessage() {
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
    speakBtn.style.display = 'none';
    header.appendChild(speakBtn);
    wrapper.appendChild(header);

    const content = document.createElement('div');
    content.className = 'message-content';
    wrapper.appendChild(content);

    $('messagesContainer').appendChild(wrapper);
    scrollBottom();

    return {
        contentDiv: content,
        speakBtn: speakBtn,
        wrapper: wrapper
    };
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
