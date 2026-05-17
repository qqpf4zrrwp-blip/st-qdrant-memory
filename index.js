// Qdrant Memory Extension for SillyTavern
// This extension retrieves relevant memories from Qdrant and injects them into conversations
// Version 3.1.3 - fixed partial memory storage during streaming

const extensionName = "qdrant-memory"

// Default settings
const defaultSettings = {
  enabled: true,
  qdrantUrl: "http://localhost:6333",
  qdrantApiKey: "",
  collectionName: "mem",
  embeddingProvider: "openai",
  openaiApiKey: "",
  openRouterApiKey: "",
  googleApiKey: "",
  googleEmbeddingModels: [], // cached list fetched from Google API: [{value,label,outputDimensionality}]
  localEmbeddingUrl: "",
  localEmbeddingApiKey: "",
  embeddingModel: "text-embedding-3-large",
  customEmbeddingDimensions: null,
  detectedDimensions: {}, // auto-detected dimensions per model name (for fetched / unknown models)
  memoryLimit: 5,
  scoreThreshold: 0.3,
  memoryPosition: 2,
  debugMode: false,
  // New v3.0 settings
  usePerCharacterCollections: true,
  autoSaveMemories: true,
  saveUserMessages: true,
  saveCharacterMessages: true,
  minMessageLength: 5,
  showMemoryNotifications: true,
  retainRecentMessages: 5,
  chunkMinSize: 1200,
  chunkMaxSize: 1500,
  chunkTimeout: 30000, // 30 seconds - save chunk if no new messages
  // NEW v3.1.2 settings
  dedupeThreshold: 0.92, // Similarity threshold for chunk deduplication
  preventDuplicateInjection: true, // Prevent inserting memories multiple times
  streamFinalizePollMs: 250,
  streamFinalizeStableMs: 1200,
  streamFinalizeMaxWaitMs: 300000,
  flushAfterAssistant: true,
  // NEW: when true, do not auto-splice memories into chat; instead expose
  // them through the {{qdrant}} macro so the user controls placement.
  useMacro: false,
}

let settings = { ...defaultSettings }
const saveQueue = []
let processingSaveQueue = false

let messageBuffer = []
let lastMessageTime = 0
let chunkTimer = null
let pendingAssistantFinalize = null

// NEW: Track which chats have had memories injected to prevent duplicates
const memoryInjectionTracker = new Set()

// Holds the most recently retrieved memory text so the {{qdrant}} macro
// (which is evaluated synchronously during prompt building) can return the
// result of the async Qdrant search performed by the generation interceptor.
let lastQdrantMemoryText = ""

// Helper to create a unique hash for a chat state
function getChatHash(chat) {
  // Create a hash based on the last few messages to identify unique chat states
  const lastMessages = chat.slice(-5).map(msg => {
    return `${msg.is_user ? 'U' : 'A'}_${msg.mes?.substring(0, 50) || ''}_${msg.send_date || ''}`
  }).join('|')
  
  return lastMessages
}

const EMBEDDING_MODEL_OPTIONS = {
  openai: [
    {
      value: "text-embedding-3-large",
      label: "text-embedding-3-large (best quality)",
    },
    {
      value: "text-embedding-3-small",
      label: "text-embedding-3-small (faster)",
    },
    {
      value: "text-embedding-ada-002",
      label: "text-embedding-ada-002 (legacy)",
    },
  ],
  openrouter: [
    {
      value: "openai/text-embedding-3-large",
      label: "OpenAI: Text Embedding 3 Large",
    },
    {
      value: "openai/text-embedding-3-small",
      label: "OpenAI: Text Embedding 3 Small",
    },
    {
      value: "openai/text-embedding-ada-002",
      label: "OpenAI: Text Embedding Ada 002",
    },
    {
      value: "qwen/qwen3-embedding-8b",
      label: "Qwen: Qwen3 Embedding 8B",
    },
    {
      value: "mistralai/mistral-embed-2312",
      label: "Mistral: Mistral Embed 2312",
    },
    {
      value: "google/gemini-embedding-001",
      label: "Google: Gemini Embedding 001",
    },
  ],
  google: [
    // populated dynamically from settings.googleEmbeddingModels (fetched via API)
    // fallback options shown before first fetch:
    {
      value: "models/text-embedding-004",
      label: "models/text-embedding-004 (768 dim, default)",
    },
    {
      value: "models/gemini-embedding-001",
      label: "models/gemini-embedding-001 (3072 dim)",
    },
  ],
}

const DEFAULT_MODEL_BY_PROVIDER = {
  openai: "text-embedding-3-large",
  openrouter: EMBEDDING_MODEL_OPTIONS.openrouter[0].value,
  google: "models/text-embedding-004",
}

const OPENROUTER_MODEL_ALIASES = {
  "text-embedding-3-large": "openai/text-embedding-3-large",
  "text-embedding-3-small": "openai/text-embedding-3-small",
  "text-embedding-ada-002": "openai/text-embedding-ada-002",
}

const OPENAI_MODEL_ALIASES = {
  "openai/text-embedding-3-large": "text-embedding-3-large",
  "openai/text-embedding-3-small": "text-embedding-3-small",
  "openai/text-embedding-ada-002": "text-embedding-ada-002",
}

// ============================================================================
// DATE/TIMESTAMP NORMALIZATION
// ============================================================================

/**
 * Normalizes various date formats to Unix timestamp in milliseconds
 */
function normalizeTimestamp(date) {
  // Already a valid millisecond timestamp
  if (typeof date === 'number' && date > 1000000000000) {
    return date;
  }
  
  // Timestamp in seconds - convert to milliseconds
  if (typeof date === 'number' && date > 1000000000 && date < 1000000000000) {
    return date * 1000;
  }
  
  // Date object
  if (date instanceof Date) {
    const timestamp = date.getTime();
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }
  
  // String date - try to parse it
  if (typeof date === 'string' && date.trim()) {
    const parsed = new Date(date);
    const timestamp = parsed.getTime();
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }
  
  // Fallback to current time
  if (settings.debugMode) {
    console.warn('[Qdrant Memory] Could not normalize timestamp, using current time. Input:', date);
  }
  return Date.now();
}

/**
 * Formats a timestamp as YYYY-MM-DD for display in memory chunks
 */
function formatDateForChunk(timestamp) {
  try {
    const dateObj = new Date(timestamp);
    if (isNaN(dateObj.getTime())) {
      throw new Error('Invalid date');
    }
    return dateObj.toISOString().split('T')[0]; // YYYY-MM-DD format
  } catch (e) {
    console.warn('[Qdrant Memory] Error formatting date:', e, 'timestamp:', timestamp);
    return new Date().toISOString().split('T')[0]; // Fallback to today
  }
}

// ============================================================================
// SETTINGS MANAGEMENT
// ============================================================================

// Returns SillyTavern's extension settings store, or null if unavailable
function getStSettingsStore() {
  try {
    const ctx = getContext()
    if (ctx && ctx.extensionSettings && typeof ctx.saveSettingsDebounced === "function") {
      return ctx
    }
  } catch (e) {
    /* fall through to localStorage */
  }
  return null
}

// Load settings from SillyTavern (extension_settings), with one-time
// migration from the old localStorage location. Falls back to
// localStorage only if the SillyTavern settings store is unavailable.
function loadSettings() {
  const store = getStSettingsStore()

  if (store) {
    const ext = store.extensionSettings
    let stored = ext[extensionName]

    // One-time migration: pull old config out of localStorage so existing
    // API keys / URLs are not lost when switching storage backend.
    if (!stored || Object.keys(stored).length === 0) {
      const legacy = localStorage.getItem(extensionName)
      if (legacy) {
        try {
          stored = JSON.parse(legacy)
          console.log("[Qdrant Memory] Migrated settings from localStorage to SillyTavern")
        } catch (e) {
          console.error("[Qdrant Memory] Failed to parse legacy localStorage settings:", e)
        }
      }
    }

    settings = { ...defaultSettings, ...(stored || {}) }
    // Write the merged object back so newly-added default keys persist.
    ext[extensionName] = settings
    store.saveSettingsDebounced()

    // Clean up the old localStorage copy after a successful migration.
    try {
      localStorage.removeItem(extensionName)
    } catch (e) {
      /* ignore */
    }

    console.log("[Qdrant Memory] Settings loaded from SillyTavern:", settings)
    return
  }

  // Fallback: SillyTavern store not ready — keep working via localStorage.
  const saved = localStorage.getItem(extensionName)
  if (saved) {
    try {
      settings = { ...defaultSettings, ...JSON.parse(saved) }
    } catch (e) {
      console.error("[Qdrant Memory] Failed to load settings:", e)
    }
  }
  console.log("[Qdrant Memory] Settings loaded (localStorage fallback):", settings)
}

// Save settings into SillyTavern's persistent settings. Falls back to
// localStorage only if the SillyTavern settings store is unavailable.
function saveSettings() {
  const store = getStSettingsStore()

  if (store) {
    store.extensionSettings[extensionName] = settings
    store.saveSettingsDebounced()
    console.log("[Qdrant Memory] Settings saved to SillyTavern")
    return
  }

  localStorage.setItem(extensionName, JSON.stringify(settings))
  console.log("[Qdrant Memory] Settings saved (localStorage fallback)")
}

// Get collection name for a character
function getCollectionName(characterName) {
  if (!settings.usePerCharacterCollections) {
    return settings.collectionName
  }

  // Qdrant 集合名稱規則（v1.x）：長度 1–255，不可包含 < > : " / \\ | ? * \\0 \\u{1F}
  // 來源：https://github.com/qdrant/qdrant/issues/6073
  // 中文、日文等 Unicode 字元完全合法，只需把禁用字元替換為底線即可。
  let sanitized = characterName
    .replace(/[<>:"/\\|?*\u0000\u001F\s]+/g, "_") // 禁用字元 + 空白 → _
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "")

  if (!sanitized) sanitized = "default"

  // 截斷以確保 `${prefix}_${sanitized}` 仍在 255 字元內
  const maxLen = 255 - settings.collectionName.length - 1
  if (sanitized.length > maxLen) sanitized = sanitized.slice(0, maxLen)

  return `${settings.collectionName}_${sanitized}`
}

// Get embedding dimensions for the selected model
function getEmbeddingDimensions() {
  const dimensions = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-ada-002": 1536,
    "qwen/qwen3-embedding-8b": 4096,
    "mistralai/mistral-embed-2312": 1024,
    "google/gemini-embedding-001": 3072,
    // Google native API model names (returned by list-models API as "models/xxx")
    "models/text-embedding-004": 768,
    "models/embedding-001": 768,
    "models/gemini-embedding-001": 3072,
    "models/gemini-embedding-exp-03-07": 3072,
    "text-embedding-004": 768,
    "embedding-001": 768,
    "gemini-embedding-001": 3072,
  }

  const customDimensions = Number.parseInt(settings.customEmbeddingDimensions, 10)
  const isCustomValid = Number.isFinite(customDimensions) && customDimensions > 0

  if (settings.embeddingProvider === "local") {
    if (isCustomValid) {
      return customDimensions
    }
    return null
  }

  // Google: prefer fetched-model metadata if available
  if (settings.embeddingProvider === "google") {
    const fetched = (settings.googleEmbeddingModels || []).find(
      (m) => m.value === settings.embeddingModel
    )
    if (fetched && Number.isFinite(fetched.outputDimensionality) && fetched.outputDimensionality > 0) {
      return fetched.outputDimensionality
    }
  }

  if (dimensions[settings.embeddingModel]) {
    return dimensions[settings.embeddingModel]
  }

  // Auto-detected from a previous embed call
  const detected = settings.detectedDimensions?.[settings.embeddingModel]
  if (Number.isFinite(detected) && detected > 0) {
    return detected
  }

  if (isCustomValid) {
    return customDimensions
  }

  return 1536
}

function updateLocalEmbeddingDimensions(vector) {
  if (!Array.isArray(vector)) {
    return
  }
  const vectorSize = vector.length
  if (!Number.isFinite(vectorSize) || vectorSize <= 0) {
    return
  }

  // Google: store per-model detected dimension (multiple models can be used)
  if (settings.embeddingProvider === "google") {
    if (!settings.detectedDimensions || typeof settings.detectedDimensions !== "object") {
      settings.detectedDimensions = {}
    }
    const model = settings.embeddingModel
    if (model && settings.detectedDimensions[model] !== vectorSize) {
      settings.detectedDimensions[model] = vectorSize
      saveSettings()
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Auto-detected google model dimension: ${model} = ${vectorSize}`)
      }
    }
    return
  }

  if (settings.embeddingProvider !== "local") {
    return
  }

  const currentDimensions = Number.parseInt(settings.customEmbeddingDimensions, 10)
  if (Number.isFinite(currentDimensions) && currentDimensions === vectorSize) {
    return
  }

  settings.customEmbeddingDimensions = vectorSize

  try {
    const $ = window.$
    if ($) {
      const $input = $("#qdrant_local_dimensions")
      if ($input && $input.length) {
        $input.val(vectorSize)
      }
    }
  } catch (error) {
    if (settings.debugMode) {
      console.warn("[Qdrant Memory] Unable to update local dimensions input:", error)
    }
  }

  saveSettings()

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] Auto-detected local embedding dimensions: ${vectorSize}`)
  }
}

function getEmbeddingProviderError() {
  const provider = settings.embeddingProvider || "openai"

  const validProviders = ["openai", "openrouter", "google", "local"]
  if (!validProviders.includes(provider)) {
    return `不支援的嵌入提供者: ${provider}`
  }

  if (provider === "openai") {
    if (!settings.openaiApiKey || !settings.openaiApiKey.trim()) {
      return "未設定 OpenAI API 金鑰"
    }
  }

  if (provider === "openrouter") {
    if (!settings.openRouterApiKey || !settings.openRouterApiKey.trim()) {
      return "未設定 OpenRouter API 金鑰"
    }
  }

  if (provider === "google") {
    if (!settings.googleApiKey || !settings.googleApiKey.trim()) {
      return "未設定 Google AI API 金鑰"
    }
    if (!settings.embeddingModel || !settings.embeddingModel.trim()) {
      return "未選擇 Google 嵌入模型，請點「重新整理模型清單」"
    }
  }

  if (provider === "local") {
    if (!settings.localEmbeddingUrl || !settings.localEmbeddingUrl.trim()) {
      return "未設定本地嵌入 URL"
    }

    if (settings.customEmbeddingDimensions != null && settings.customEmbeddingDimensions !== "") {
      const customDimensions = Number.parseInt(settings.customEmbeddingDimensions, 10)
      if (!Number.isFinite(customDimensions) || customDimensions <= 0) {
        return "嵌入維度必須為正整數"
      }
    }
  }

  if (!provider) {
    return "尚未設定嵌入提供者"
  }

  return null
}

// ============================================================================
// HTTP HEADERS AND CSRF TOKEN HANDLING
// ============================================================================

// Helper to safely call potential CSRF token providers
function tryGetCSRFTokenFromHelpers() {
  const helperCandidates = [
    () => (typeof window.getCSRFToken === "function" ? window.getCSRFToken() : null),
    () => (typeof window.getCsrfToken === "function" ? window.getCsrfToken() : null),
    () =>
      typeof window.SillyTavern?.getCSRFToken === "function"
        ? window.SillyTavern.getCSRFToken()
        : null,
    () =>
      typeof window.SillyTavern?.getCsrfToken === "function"
        ? window.SillyTavern.getCsrfToken()
        : null,
    () =>
      typeof window.SillyTavern?.extensions?.webui?.getCSRFToken === "function"
        ? window.SillyTavern.extensions.webui.getCSRFToken()
        : null,
    () =>
      typeof window.SillyTavern?.extensions?.webui?.getCsrfToken === "function"
        ? window.SillyTavern.extensions.webui.getCsrfToken()
        : null,
  ]

  for (const helper of helperCandidates) {
    try {
      const token = helper()
      if (typeof token === "string" && token.trim().length > 0) {
        return token.trim()
      }
    } catch (error) {
      console.warn("[Qdrant Memory] Failed to read CSRF token from helper:", error)
    }
  }

  return null
}

// Helper to read a cookie value by name
function getCookie(name) {
  const cookies = document.cookie ? document.cookie.split(";") : []

  for (const cookie of cookies) {
    const [cookieName, ...rest] = cookie.trim().split("=")
    if (cookieName === name) {
      return decodeURIComponent(rest.join("="))
    }
  }

  return null
}

function pickFirstCSRFToken() {
  const tokenCandidates = [
    document.querySelector('meta[name="csrf-token"]')?.content,
    document.querySelector('meta[name="csrfToken"]')?.content,
    window.CSRF_TOKEN,
    window.CSRFToken,
    window.csrfToken,
    window.csrf_token,
    tryGetCSRFTokenFromHelpers(),
    getCookie("csrftoken"),
    getCookie("csrf_token"),
    getCookie("XSRF-TOKEN"),
    getCookie("XSRF_TOKEN"),
  ]

  for (const token of tokenCandidates) {
    if (typeof token === "string" && token.trim().length > 0) {
      return token.trim()
    }
  }

  return null
}

function getHeadersFromSillyTavernContext() {
  try {
    const context = window.SillyTavern?.getContext?.()
    const getRequestHeaders = context?.getRequestHeaders

    if (typeof getRequestHeaders === "function") {
      const headers = getRequestHeaders.call(context)

      if (headers && typeof headers === "object") {
        return headers
      }
    }
  } catch (error) {
    console.warn("[Qdrant Memory] Failed to read headers from SillyTavern context:", error)
  }

  return null
}

// Get headers for SillyTavern API requests (with CSRF token if available)
function getSillyTavernHeaders() {
  if (settings.debugMode) {
    console.log("[Qdrant Memory] === Checking available ST methods ===")
    console.log("[Qdrant Memory] window.SillyTavern exists?", typeof SillyTavern !== "undefined")
    if (typeof SillyTavern !== "undefined") {
      console.log("[Qdrant Memory] SillyTavern keys:", Object.keys(SillyTavern))
      console.log("[Qdrant Memory] SillyTavern.getContext exists?", typeof SillyTavern.getContext === "function")
      if (typeof SillyTavern.getContext === "function") {
        const ctx = SillyTavern.getContext()
        console.log("[Qdrant Memory] Context keys:", Object.keys(ctx))
        console.log("[Qdrant Memory] getRequestHeaders exists?", typeof ctx.getRequestHeaders === "function")
      }
    }
  }

  // Try multiple possible locations for the header builder
  const headerBuilders = [
    () => SillyTavern?.getContext?.()?.getRequestHeaders?.(),
    () => SillyTavern?.getRequestHeaders?.(),
    () => window.getRequestHeaders?.(),
    () => getContext()?.getRequestHeaders?.(),
  ]

  for (const builder of headerBuilders) {
    try {
      const headers = builder()
      if (headers && typeof headers === "object") {
        if (settings.debugMode) {
          console.log("[Qdrant Memory] ✓ Found working header builder!")
          console.log("[Qdrant Memory] Headers:", headers)
        }
        return headers
      }
    } catch (error) {
      // Continue to next method
    }
  }
  
  // None of the built-in methods worked
  if (settings.debugMode) {
    console.warn("[Qdrant Memory] No built-in header builder found, using manual fallback")
  }
  
  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
    Origin: window.location.origin,
    "X-Requested-With": "XMLHttpRequest",
  }

  const csrfToken = pickFirstCSRFToken()

  if (csrfToken) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Using manual CSRF token:", csrfToken.substring(0, 10) + "...")
    }
    headers["X-CSRF-Token"] = csrfToken
    headers["X-CSRFToken"] = csrfToken
    headers["csrf-token"] = csrfToken
  } else {
    console.warn("[Qdrant Memory] No CSRF token found - requests may fail")
  }

  return headers
}

// Get headers for Qdrant requests (with optional API key)
function getQdrantHeaders() {
  const headers = {
    "Content-Type": "application/json",
  }
  
  if (settings.qdrantApiKey) {
    headers["api-key"] = settings.qdrantApiKey
  }
  
  return headers
}

// ============================================================================
// QDRANT COLLECTION MANAGEMENT
// ============================================================================

// Check if collection exists
async function collectionExists(collectionName) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeURIComponent(collectionName)}`, {
      headers: getQdrantHeaders(),
    })

    if (response.status === 404) {
      return { exists: false, vectorSize: null }
    }

    if (!response.ok) {
      console.error(
        `[Qdrant Memory] Failed to fetch collection info: ${collectionName} (${response.status} ${response.statusText})`,
      )
      return { exists: false, vectorSize: null }
    }

    const data = await response.json().catch(() => null)
    const vectorSize =
      data?.result?.config?.params?.vectors?.size ??
      data?.result?.config?.params?.vectors?.default?.size ??
      data?.result?.vectors?.size ??
      null

    return { exists: true, vectorSize }
  } catch (error) {
    console.error("[Qdrant Memory] Error checking collection:", error)
    return { exists: false, vectorSize: null }
  }
}

// Create collection for a character
async function createCollection(collectionName, vectorSize) {
  try {
    const dimensions = Number.isFinite(vectorSize) && vectorSize > 0 ? vectorSize : getEmbeddingDimensions()

    if (!Number.isFinite(dimensions) || dimensions <= 0) {
      console.error(`[Qdrant Memory] Cannot create collection ${collectionName} - invalid embedding dimensions`)
      return false
    }

    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeURIComponent(collectionName)}`, {
      method: "PUT",
      headers: getQdrantHeaders(),
      body: JSON.stringify({
        vectors: {
          size: dimensions,
          distance: "Cosine",
        },
      }),
    })

    if (response.ok) {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Created collection: ${collectionName}`)
      }
      return true
    } else {
      console.error(`[Qdrant Memory] Failed to create collection: ${collectionName}`)
      return false
    }
  } catch (error) {
    console.error("[Qdrant Memory] Error creating collection:", error)
    return false
  }
}

// Ensure collection exists (create if needed)
async function ensureCollection(characterName, vectorSize) {
  const collectionName = getCollectionName(characterName)
  const { exists, vectorSize: existingSize } = await collectionExists(collectionName)

  if (exists) {
    if (
      Number.isFinite(existingSize) &&
      Number.isFinite(vectorSize) &&
      existingSize > 0 &&
      vectorSize > 0 &&
      existingSize !== vectorSize
    ) {
      console.error(
        `[Qdrant Memory] Collection ${collectionName} has dimension ${existingSize}, but embedding returned ${vectorSize}. Please recreate the collection to match the model.`,
      )
      return false
    }

    return true
  }

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] Collection doesn't exist, creating: ${collectionName}`)
  }

  return await createCollection(collectionName, vectorSize)
}

// ============================================================================
// EMBEDDING GENERATION
// ============================================================================

// Generate embedding using the configured provider
async function generateEmbedding(text) {
  const providerError = getEmbeddingProviderError()
  if (providerError) {
    console.error(`[Qdrant Memory] ${providerError}`)
    return null
  }

  try {
    const provider = settings.embeddingProvider || "openai"

    // Google uses native (non-OpenAI-compatible) API: POST /v1beta/{model}:embedContent?key=KEY
    if (provider === "google") {
      const model = settings.embeddingModel
      // Ensure "models/" prefix; Google list-models API returns names as "models/xxx"
      const modelPath = model.startsWith("models/") ? model : `models/${model}`
      const apiKey = settings.googleApiKey.trim()
      const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:embedContent?key=${encodeURIComponent(apiKey)}`

      const gBody = {
        model: modelPath,
        content: { parts: [{ text }] },
      }

      const gResp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(gBody),
      })

      if (!gResp.ok) {
        const errorData = await gResp.json().catch(() => ({}))
        console.error("[Qdrant Memory] google embedding API error:", gResp.statusText, errorData)
        return null
      }

      const gData = await gResp.json()
      const gVector = gData?.embedding?.values || gData?.embedding?.value || null

      if (!Array.isArray(gVector)) {
        console.error("[Qdrant Memory] Unable to parse google embedding response", gData)
        return null
      }

      updateLocalEmbeddingDimensions(gVector)
      return gVector
    }

    let url = "https://api.openai.com/v1/embeddings"
    const headers = {
      "Content-Type": "application/json",
    }
    const body = {
      model: settings.embeddingModel,
      input: text,
    }

    if (provider === "openai") {
      headers.Authorization = `Bearer ${settings.openaiApiKey}`
    } else if (provider === "openrouter") {
      url = "https://openrouter.ai/api/v1/embeddings"
      headers.Authorization = `Bearer ${settings.openRouterApiKey}`
      if (window?.location?.origin) {
        headers["HTTP-Referer"] = window.location.origin
      }
      if (document?.title) {
        headers["X-Title"] = document.title
      }
    } else if (provider === "local") {
      url = settings.localEmbeddingUrl.trim()
      if (settings.localEmbeddingApiKey && settings.localEmbeddingApiKey.trim()) {
        headers.Authorization = `Bearer ${settings.localEmbeddingApiKey.trim()}`
      }
    } else {
      console.error(`[Qdrant Memory] Unsupported embedding provider: ${provider}`)
      return null
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      console.error(
        `[Qdrant Memory] ${provider} embedding API error:`,
        response.statusText,
        errorData
      )
      return null
    }

    const data = await response.json()
    let embeddingVector = null

    if (Array.isArray(data?.data) && Array.isArray(data.data[0]?.embedding)) {
      embeddingVector = data.data[0].embedding
    } else if (Array.isArray(data?.data) && Array.isArray(data.data[0]?.vector)) {
      embeddingVector = data.data[0].vector
    } else if (Array.isArray(data?.embedding)) {
      embeddingVector = data.embedding
    } else if (Array.isArray(data?.embeddings)) {
      embeddingVector = data.embeddings[0]
    }

    if (!Array.isArray(embeddingVector)) {
      console.error("[Qdrant Memory] Unable to parse embedding response", data)
      return null
    }

    updateLocalEmbeddingDimensions(embeddingVector)

    return embeddingVector
  } catch (error) {
    console.error("[Qdrant Memory] Error generating embedding:", error)
    return null
  }
}

// ============================================================================
// GOOGLE AI: FETCH AVAILABLE EMBEDDING MODELS
// ============================================================================

/**
 * Fetch list of embedding-capable models from Google AI API.
 * Filters out models that don't declare embedContent in supportedGenerationMethods.
 * Handles pagination via nextPageToken.
 * @param {string} apiKey
 * @returns {Promise<Array<{value:string,label:string,outputDimensionality:number|null}>>}
 */
async function fetchGoogleEmbeddingModels(apiKey) {
  if (!apiKey || !apiKey.trim()) {
    throw new Error("缺少 Google API 金鑰")
  }
  const trimmed = apiKey.trim()
  const collected = []
  let pageToken = ""
  let safetyCounter = 0

  while (safetyCounter < 10) {
    safetyCounter += 1
    const params = new URLSearchParams({
      key: trimmed,
      pageSize: "200",
    })
    if (pageToken) params.set("pageToken", pageToken)

    const url = `https://generativelanguage.googleapis.com/v1beta/models?${params.toString()}`
    const resp = await fetch(url, { method: "GET" })

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}))
      const msg = errData?.error?.message || resp.statusText || "Unknown error"
      throw new Error(`Google API 錯誤 (${resp.status}): ${msg}`)
    }

    const data = await resp.json()
    const models = Array.isArray(data?.models) ? data.models : []

    for (const m of models) {
      const methods = Array.isArray(m.supportedGenerationMethods) ? m.supportedGenerationMethods : []
      if (!methods.includes("embedContent")) continue

      const name = m.name || ""
      const displayName = m.displayName || name.replace(/^models\//, "")
      // Try several possible field names; if absent, leave null and auto-detect on first call
      const dimRaw = m.outputDimensionality ?? m.outputDimension ?? null
      const dim = Number.isFinite(dimRaw) ? dimRaw : null

      collected.push({
        value: name,
        label: dim ? `${displayName} (${dim} dim)` : displayName,
        outputDimensionality: dim,
      })
    }

    pageToken = data?.nextPageToken || ""
    if (!pageToken) break
  }

  // Sort: newer "gemini-embedding-*" first, then "text-embedding-*", then rest
  collected.sort((a, b) => {
    const score = (v) => (/gemini-embedding/i.test(v) ? 0 : /text-embedding/i.test(v) ? 1 : 2)
    const sa = score(a.value)
    const sb = score(b.value)
    if (sa !== sb) return sa - sb
    return a.value.localeCompare(b.value)
  })

  return collected
}

// ============================================================================
// MEMORY SEARCH AND RETRIEVAL
// ============================================================================

// NEW: Check if chunk already exists (deduplication)
async function chunkExistsInCollection(collectionName, embedding, text, dedupeThreshold) {
  try {
    const searchPayload = {
      vector: embedding,
      limit: 1,
      score_threshold: dedupeThreshold,
      with_payload: true,
    }

    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeURIComponent(collectionName)}/points/search`, {
      method: "POST",
      headers: getQdrantHeaders(),
      body: JSON.stringify(searchPayload),
    })

    if (!response.ok) {
      return false
    }

    const data = await response.json()
    const results = data.result || []

    if (results.length > 0) {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Found similar chunk with score: ${results[0].score.toFixed(4)}`)
        console.log(`[Qdrant Memory] Existing: "${results[0].payload?.text?.substring(0, 80)}..."`)
        console.log(`[Qdrant Memory] New: "${text.substring(0, 80)}..."`)
      }
      return true
    }

    return false
  } catch (error) {
    console.warn('[Qdrant Memory] Deduplication check failed:', error)
    return false
  }
}

// Search Qdrant for relevant memories
async function searchMemories(query, characterName) {
  if (!settings.enabled) return []

  try {
    const collectionName = getCollectionName(characterName)

    const embedding = await generateEmbedding(query)
    if (!embedding) return []

    const collectionReady = await ensureCollection(characterName, embedding.length)
    if (!collectionReady) {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Collection not ready: ${collectionName}`)
      }
      return []
    }

    // FIXED: Improved retain logic - get ALL message IDs that should be excluded
    const context = getContext()
    const chat = context.chat || []
    const excludedMessageIds = new Set()

    if (settings.retainRecentMessages > 0) {
      // Get the last N messages
      const recentMessages = chat.slice(-settings.retainRecentMessages)
      
      recentMessages.forEach(msg => {
        // Create all possible message ID formats this message might have been saved as
        const normalizedDate = normalizeTimestamp(msg.send_date || Date.now())
        const msgIndex = chat.indexOf(msg)
        
        // Add multiple ID formats to catch all variations
        excludedMessageIds.add(`${characterName}_${normalizedDate}_${msgIndex}`)
        excludedMessageIds.add(`${characterName}_${msg.send_date}_${msgIndex}`)
        
        if (settings.debugMode && excludedMessageIds.size <= 5) {
          console.log(`[Qdrant Memory] Excluding message ID: ${characterName}_${normalizedDate}_${msgIndex}`)
        }
      })

      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Excluding ${excludedMessageIds.size} recent message IDs from search`)
      }
    }

    const searchPayload = {
      vector: embedding,
      limit: settings.memoryLimit * 2, // Get more results for filtering
      score_threshold: settings.scoreThreshold,
      with_payload: true,
    }

    const filterConditions = []

    // Add character filter if using shared collection
    if (!settings.usePerCharacterCollections) {
      filterConditions.push({
        key: "character",
        match: { value: characterName },
      })
    }

    // Only add filter if we have conditions
    if (filterConditions.length > 0) {
      searchPayload.filter = {
        must: filterConditions,
      }
    }

    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeURIComponent(collectionName)}/points/search`, {
      method: "POST",
      headers: getQdrantHeaders(),
      body: JSON.stringify(searchPayload),
    })

    if (!response.ok) {
      console.error("[Qdrant Memory] Search failed:", response.statusText)
      return []
    }

    const data = await response.json()
    let results = data.result || []

    // FIXED: Filter out chunks that contain any excluded message IDs
    if (excludedMessageIds.size > 0) {
      const beforeFilterCount = results.length
      
      results = results.filter(memory => {
        const messageIds = memory.payload.messageIds || ""
        const chunkMessageIds = messageIds.split(",")
        
        // Check if any of the chunk's message IDs are in the excluded set
        const hasExcludedMessage = chunkMessageIds.some(id => excludedMessageIds.has(id.trim()))
        
        if (hasExcludedMessage && settings.debugMode) {
          console.log(`[Qdrant Memory] Filtered out chunk containing recent message: ${messageIds}`)
        }
        
        return !hasExcludedMessage
      })

      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Filtered ${beforeFilterCount - results.length} chunks with recent messages`)
      }
    }

    // Deduplicate results based on text similarity
const uniqueResults = []
const seenTexts = new Set()

for (const result of results) {
  const text = result.payload?.text || ""
  
  // Create a normalized version for comparison (remove dates, extra whitespace)
  const normalizedText = text
    .replace(/\[[\d-]+\]/g, '') // Remove date markers
    .replace(/\s+/g, ' ')        // Normalize whitespace
    .trim()
    .substring(0, 200)           // Compare first 200 chars
  
  // Only add if we haven't seen very similar text
  if (!seenTexts.has(normalizedText)) {
    seenTexts.add(normalizedText)
    uniqueResults.push(result)
  } else if (settings.debugMode) {
    console.log(`[Qdrant Memory] Filtered duplicate search result: "${normalizedText.substring(0, 50)}..."`)  // ← FIXED: use () not backticks
  }
  
  // Stop if we have enough unique results
  if (uniqueResults.length >= settings.memoryLimit) {
    break
  }
}

results = uniqueResults

if (settings.debugMode) {
  console.log(`[Qdrant Memory] Found ${results.length} unique memories (after deduplication)`)  // ← FIXED: use () not backticks
}

    return results
  } catch (error) {
    console.error("[Qdrant Memory] Error searching memories:", error)
    return []
  }
}

// Format memories for display
function formatMemories(memories) {
  if (!memories || memories.length === 0) return ""

  let formatted = "\n[Past chat memories]\n\n"
  
  // Get persona name for display
  const personaName = getPersonaName()

  memories.forEach((memory) => {
    const payload = memory.payload

    let speakerLabel
    if (payload.isChunk) {
      // For conversation chunks, show all speakers
      speakerLabel = `Conversation (${payload.speakers})`
    } else {
      // For individual messages (legacy format), use persona name
      speakerLabel = payload.speaker === "user" 
        ? `${personaName} said`   // ← CHANGED: Use personaName instead of "User"
        : "Character said"
    }

    let text = payload.text.replace(/\n/g, " ") // flatten newlines

    const score = (memory.score * 100).toFixed(0)

    formatted += `• ${speakerLabel}: "${text}" (score: ${score}%)\n\n`
  })

  return formatted
}

// ============================================================================
// MESSAGE CHUNKING AND BUFFERING
// ============================================================================

function getChatParticipants() {
  const context = getContext()
  const characterName = context.name2

  // Check if this is a group chat
  const characters = context.characters || []
  const chat = context.chat || []

  // For group chats, get all unique character names from recent messages
  if (characters.length > 1) {
    const participants = new Set()

    // Add the main character
    if (characterName) {
      participants.add(characterName)
    }

    // Look through recent messages to find all participants
    chat.slice(-50).forEach((msg) => {
      if (!msg.is_user && msg.name && msg.name !== "System") {
        participants.add(msg.name)
      }
    })

    return Array.from(participants)
  }

  // Single character chat
  return characterName ? [characterName] : []
}

function createChunkFromBuffer() {
  if (messageBuffer.length === 0) return null

  let chunkText = ""
  const speakers = new Set()
  const messageIds = []
  let totalLength = 0
  const currentTimestamp = Date.now()
  
  // NEW: Get the persona name once for this chunk
  const personaName = getPersonaName()

  // Build chunk text with speaker labels
  messageBuffer.forEach((msg) => {
    const speaker = msg.isUser ? personaName : msg.characterName  // ← CHANGED: Use personaName
    speakers.add(speaker)
    messageIds.push(msg.messageId)

    const line = `${speaker}: ${msg.text}\n`
    chunkText += line
    totalLength += line.length
  })

  // Format date prefix
  let finalText = chunkText.trim()
  const dateStr = formatDateForChunk(currentTimestamp)
  finalText = `[${dateStr}]\n${finalText}`

  return {
    text: finalText,
    speakers: Array.from(speakers),
    messageIds: messageIds,
    messageCount: messageBuffer.length,
    timestamp: currentTimestamp,
  }
}

async function saveChunkToQdrant(chunk, participants) {
  if (!settings.enabled) return false
  if (!chunk || !participants || participants.length === 0) return false

  try {
    // Generate embedding for the chunk text
    const embedding = await generateEmbedding(chunk.text)
    if (!embedding) {
      console.error("[Qdrant Memory] Cannot save chunk - embedding generation failed")
      return false
    }

    // NEW: Check for duplicates before saving
    let alreadyExists = false
    
    for (const characterName of participants) {
      const collectionName = getCollectionName(characterName)
      const collectionReady = await ensureCollection(characterName, embedding.length)
      
      if (!collectionReady) {
        console.error(`[Qdrant Memory] Cannot check duplicates - collection creation failed for ${characterName}`)
        continue
      }

      const exists = await chunkExistsInCollection(
        collectionName, 
        embedding, 
        chunk.text, 
        settings.dedupeThreshold
      )
      
      if (exists) {
        alreadyExists = true
        if (settings.debugMode) {
          console.log(`[Qdrant Memory] Duplicate chunk detected in ${characterName}'s collection, skipping save`)
        }
        break
      }
    }

    if (alreadyExists) {
      if (settings.showMemoryNotifications) {
        const toastr = window.toastr
        toastr.info("已儲存過類似對話", "Qdrant Memory", { timeOut: 1500 })
      }
      return false
    }

    const pointId = generateUUID()

    // Prepare payload
    const payload = {
      text: chunk.text,
      speakers: chunk.speakers.join(", "),
      messageCount: chunk.messageCount,
      timestamp: chunk.timestamp,
      messageIds: chunk.messageIds.join(","),
      isChunk: true,
    }

    // Save to all participant collections
    const savePromises = participants.map(async (characterName) => {
      const collectionName = getCollectionName(characterName)

      // Ensure collection exists
      const collectionReady = await ensureCollection(characterName, embedding.length)
      if (!collectionReady) {
        console.error(`[Qdrant Memory] Cannot save chunk - collection creation failed for ${characterName}`)
        return false
      }

      // Add character name to payload only if using shared collection
      const characterPayload = settings.usePerCharacterCollections 
        ? payload 
        : { ...payload, character: characterName }

      // Save to Qdrant
      const response = await fetch(`${settings.qdrantUrl}/collections/${encodeURIComponent(collectionName)}/points`, {
        method: "PUT",
        headers: getQdrantHeaders(),
        body: JSON.stringify({
          points: [
            {
              id: pointId,
              vector: embedding,
              payload: characterPayload,
            },
          ],
        }),
      })

      if (!response.ok) {
        console.error(
          `[Qdrant Memory] Failed to save chunk to ${characterName}: ${response.status} ${response.statusText}`,
        )
        return false
      }

      if (settings.debugMode) {
        console.log(
          `[Qdrant Memory] Saved chunk to ${characterName}'s collection (${chunk.messageCount} messages, ${chunk.text.length} chars)`,
        )
      }

      return true
    })

    const results = await Promise.all(savePromises)
    const successCount = results.filter((r) => r).length

    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Chunk saved to ${successCount}/${participants.length} collections`)
    }

    return successCount > 0
  } catch (err) {
    console.error("[Qdrant Memory] Error saving chunk:", err)
    return false
  }
}

async function processMessageBuffer() {
  if (!settings.enabled) return
  if (messageBuffer.length === 0) return

  const chunk = createChunkFromBuffer()
  if (!chunk) return

  // Get all participants (for group chats)
  const participants = getChatParticipants()

  if (participants.length === 0) {
    console.error("[Qdrant Memory] No participants found for chunk")
    messageBuffer = []
    return
  }

  // Save chunk to all participant collections
  await saveChunkToQdrant(chunk, participants)

  // Clear buffer after saving
  messageBuffer = []
}

function bufferMessage(text, characterName, isUser, messageId) {
  if (!settings.enabled) return
  if (!settings.autoSaveMemories) return
  if (getEmbeddingProviderError()) return
  if (text.length < settings.minMessageLength) return

  // Check if we should save this type of message
  if (isUser && !settings.saveUserMessages) return
  if (!isUser && !settings.saveCharacterMessages) return

  // Add to buffer
  messageBuffer.push({ text, characterName, isUser, messageId })
  lastMessageTime = Date.now()

  // Calculate current buffer size
  let bufferSize = 0
  messageBuffer.forEach((msg) => {
    bufferSize += msg.text.length + msg.characterName.length + 4 // +4 for ": " and "\n"
  })

  if (settings.debugMode) {
    console.log(`[Qdrant Memory] Buffer: ${messageBuffer.length} messages, ${bufferSize} chars`)
  }

  // Clear existing timer
  if (chunkTimer) {
    clearTimeout(chunkTimer)
  }

  // If buffer exceeds max size, process it now
  if (bufferSize >= settings.chunkMaxSize) {
    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Buffer reached max size (${bufferSize}), processing chunk`)
    }
    processMessageBuffer()
  }
  // If buffer is at least min size, set a short timer
  else if (bufferSize >= settings.chunkMinSize) {
    chunkTimer = setTimeout(() => {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Buffer reached min size and timeout, processing chunk`)
      }
      processMessageBuffer()
    }, 5000) // 5 seconds after reaching min size
  }
  // Otherwise, set a longer timer
  else {
    chunkTimer = setTimeout(() => {
      if (settings.debugMode) {
        console.log(`[Qdrant Memory] Buffer timeout reached, processing chunk`)
      }
      processMessageBuffer()
    }, settings.chunkTimeout)
  }
}

// ============================================================================
// CHAT INDEXING FUNCTIONS
// ============================================================================

async function getCharacterChats(characterName) {
  try {
    const context = getContext()

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Getting chats for character:", characterName)
    }

    // Try to get the character's avatar URL
    let avatar_url = `${characterName}.png`
    if (context.characters && Array.isArray(context.characters)) {
      const char = context.characters.find((c) => c.name === characterName)
      if (char && char.avatar) {
        avatar_url = char.avatar
      }
    }

    const response = await fetch("/api/characters/chats", {
      method: "POST",
      headers: getSillyTavernHeaders(),
      credentials: "include",
      body: JSON.stringify({
        avatar_url: avatar_url,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("[Qdrant Memory] Failed to get chat list:", response.status, response.statusText)
      console.error("[Qdrant Memory] Error response:", errorText)
      return []
    }

    const data = await response.json()

    // Handle different response formats - extract just the filenames
    let chatFiles = []
    
    if (Array.isArray(data)) {
      if (typeof data[0] === 'string') {
        chatFiles = data
      } else if (data[0] && data[0].file_name) {
        chatFiles = data.map(item => item.file_name)
      } else {
        chatFiles = data.map(item => {
          if (typeof item === 'string') return item
          if (item.file_name) return item.file_name
          if (item.filename) return item.filename
          return null
        }).filter(f => f !== null)
      }
    } else if (data && Array.isArray(data.files)) {
      chatFiles = data.files.map(item => {
        if (typeof item === 'string') return item
        if (item.file_name) return item.file_name
        if (item.filename) return item.filename
        return null
      }).filter(f => f !== null)
    } else if (data && Array.isArray(data.chats)) {
      chatFiles = data.chats.map(item => {
        if (typeof item === 'string') return item
        if (item.file_name) return item.file_name
        if (item.filename) return item.filename
        return null
      }).filter(f => f !== null)
    }

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Extracted filenames:", chatFiles)
    }

    return chatFiles
  } catch (error) {
    console.error("[Qdrant Memory] Error getting character chats:", error)
    return []
  }
}

async function loadChatFile(characterName, chatFile) {
  try {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Loading chat file:", chatFile, "for character:", characterName)
    }

    // Ensure chatFile is a string
    if (typeof chatFile !== 'string') {
      console.error("[Qdrant Memory] chatFile is not a string:", chatFile)
      if (chatFile && chatFile.file_name) {
        chatFile = chatFile.file_name
      } else {
        return null
      }
    }

    // Remove .jsonl extension as the API adds it back
    const fileNameWithoutExt = chatFile.replace(/\.jsonl$/, '')

    const context = getContext()

    

    // Try to get the character's avatar URL
    let avatar_url = `${characterName}.png`
    if (context.characters && Array.isArray(context.characters)) {
      const char = context.characters.find((c) => c.name === characterName)
      if (char && char.avatar) {
        avatar_url = char.avatar
      }
    }

    const response = await fetch("/api/chats/get", {
      method: "POST",
      headers: getSillyTavernHeaders(),
      credentials: "include",
      body: JSON.stringify({
        ch_name: characterName,
        file_name: fileNameWithoutExt,
        avatar_url: avatar_url,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("[Qdrant Memory] Failed to load chat file:", response.status, response.statusText)
      console.error("[Qdrant Memory] Error response:", errorText)
      return null
    }

    const chatData = await response.json()
    
    // Handle different response formats
    let messages = null
    if (Array.isArray(chatData)) {
      messages = chatData
    } else if (chatData && Array.isArray(chatData.chat)) {
      messages = chatData.chat
    } else if (chatData && Array.isArray(chatData.messages)) {
      messages = chatData.messages
    } else if (chatData && typeof chatData === 'object') {
      messages = [chatData]
    }
    
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Loaded chat with", messages?.length || 0, "messages")
    }
    
    return messages
  } catch (error) {
    console.error("[Qdrant Memory] Error loading chat file:", error)
    return null
  }
}

async function chunkExists(collectionName, messageIds) {
  try {
    // Search for any of the message IDs in the chunk
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeURIComponent(collectionName)}/points/scroll`, {
      method: "POST",
      headers: getQdrantHeaders(),
      body: JSON.stringify({
        filter: {
          should: messageIds.map((id) => ({
            key: "messageIds",
            match: { text: id },
          })),
        },
        limit: 1,
        with_payload: false,
      }),
    })

    if (!response.ok) return false

    const data = await response.json()
    return data.result?.points?.length > 0
  } catch (error) {
    console.error("[Qdrant Memory] Error checking chunk existence:", error)
    return false
  }
}

function createChunksFromChat(messages, characterName) {
  const chunks = []
  let currentChunk = []
  let currentSize = 0

  for (const msg of messages) {
    // Skip system messages
    if (msg.is_system) continue

    const text = msg.mes?.trim()
    if (!text || text.length < settings.minMessageLength) continue

    // Check if we should save this type of message
    const isUser = msg.is_user || false
    if (isUser && !settings.saveUserMessages) continue
    if (!isUser && !settings.saveCharacterMessages) continue

    // Normalize send_date before using it
    const normalizedDate = normalizeTimestamp(msg.send_date || Date.now())
    
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Message date - raw:", msg.send_date, "normalized:", normalizedDate, "formatted:", formatDateForChunk(normalizedDate))
    }

    // Create message object
    const messageObj = {
      text: text,
      characterName: characterName,
      isUser: isUser,
      messageId: `${characterName}_${normalizedDate}_${messages.indexOf(msg)}`,
      timestamp: normalizedDate,
    }

    const messageSize = text.length + characterName.length + 4

    // If adding this message would exceed max size, save current chunk
    if (currentSize + messageSize > settings.chunkMaxSize && currentChunk.length > 0) {
      chunks.push(createChunkFromMessages(currentChunk))
      currentChunk = []
      currentSize = 0
    }

    currentChunk.push(messageObj)
    currentSize += messageSize

    // If we've reached min size and have a good number of messages, consider chunking
    if (currentSize >= settings.chunkMinSize && currentChunk.length >= 3) {
      chunks.push(createChunkFromMessages(currentChunk))
      currentChunk = []
      currentSize = 0
    }
  }

  // Save any remaining messages
  if (currentChunk.length > 0) {
    chunks.push(createChunkFromMessages(currentChunk))
  }

  return chunks
}

function createChunkFromMessages(messages) {
  let chunkText = ""
  const speakers = new Set()
  const messageIds = []
  let oldestTimestamp = Number.POSITIVE_INFINITY
  
  // NEW: Get the persona name once for all messages
  const personaName = getPersonaName()

  messages.forEach((msg) => {
    const speaker = msg.isUser ? personaName : msg.characterName  // ← CHANGED: Use personaName
    speakers.add(speaker)
    messageIds.push(msg.messageId)

    const line = `${speaker}: ${msg.text}\n`
    chunkText += line

    if (msg.timestamp < oldestTimestamp) {
      oldestTimestamp = msg.timestamp
    }
  })

  // Format date prefix for the chunk
  let finalText = chunkText.trim()
  if (oldestTimestamp !== Number.POSITIVE_INFINITY) {
    const dateStr = formatDateForChunk(oldestTimestamp)
    finalText = `[${dateStr}]\n${finalText}`
  }

  return {
    text: finalText,
    speakers: Array.from(speakers),
    messageIds: messageIds,
    messageCount: messages.length,
    timestamp: oldestTimestamp !== Number.POSITIVE_INFINITY ? oldestTimestamp : Date.now(),
  }
}

async function indexCharacterChats() {
  const context = getContext()
  const characterName = context.name2
  const toastr = window.toastr
  const $ = window.$

  if (!characterName) {
    toastr.warning("尚未選擇角色", "Qdrant Memory")
    return
  }

  const providerError = getEmbeddingProviderError()
  if (providerError) {
    toastr.error(providerError, "Qdrant Memory")
    return
  }

  // Create progress modal.
  // The modal sits inside a full-screen flex container that handles
  // centering. This avoids the mobile bug where transform-centering a modal
  // taller than the (dynamic) viewport pushes its top off-screen with no way
  // to scroll up. Height uses dvh (dynamic viewport) so the collapsing mobile
  // address bar does not cause clipping; svh/vh are fallbacks for old browsers.
  const modalHtml = `
    <div id="qdrant_index_overlay" style="
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.5);
      z-index: 9999;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      box-sizing: border-box;
      padding: 5vh 16px calc(5vh + env(safe-area-inset-bottom)) 16px;
      text-align: center;
    ">
      <div id="qdrant_index_modal" style="
        display: inline-block;
        text-align: left;
        vertical-align: top;
        background: white;
        padding: clamp(16px, 5vw, 28px);
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        width: 100%;
        max-width: 500px;
        box-sizing: border-box;
      ">
        <div style="color: #333;">
          <h3 style="margin-top: 0;">索引對話 - ${characterName}</h3>
          <p id="qdrant_index_status">正在掃描對話檔案...</p>
          <div style="background: #f0f0f0; border-radius: 5px; height: 20px; margin: 15px 0; overflow: hidden;">
            <div id="qdrant_index_progress" style="background: #4CAF50; height: 100%; width: 0%; transition: width 0.3s;"></div>
          </div>
          <p id="qdrant_index_details" style="font-size: 0.9em; color: #666;"></p>
          <button id="qdrant_index_cancel" class="menu_button" style="margin-top: 15px; min-height: 44px; padding: 10px 16px; width: 100%;">取消</button>
        </div>
      </div>
    </div>
  `

  // Attach to body so position:fixed + inset:0 resolves against the viewport.
  // No flexbox: the overlay is a full-screen scroll area; the white box is a
  // normal inline-block sized purely by its content (cannot collapse to a
  // strip). If content is taller than the screen, the OVERLAY scrolls.
  $("body").append(modalHtml)

  const closeModal = () => {
    $("#qdrant_index_modal").remove()
    $("#qdrant_index_overlay").remove()
  }

  const setCancelButtonToClose = () => {
    $("#qdrant_index_cancel")
      .prop("disabled", false)
      .text("關閉")
      .off("click")
      .on("click", closeModal)
  }

  let cancelled = false
  $("#qdrant_index_cancel").on("click", () => {
    cancelled = true
    $("#qdrant_index_cancel").text("取消中...").prop("disabled", true)
  })

  try {
    // Get all chat files for this character
    const chatFiles = await getCharacterChats(characterName)

    if (chatFiles.length === 0) {
      $("#qdrant_index_status").text("找不到對話檔案")
      setCancelButtonToClose()
      setTimeout(() => {
        closeModal()
      }, 2000)
      return
    }

    $("#qdrant_index_status").text(`找到 ${chatFiles.length} 個對話檔案`)

    const collectionName = getCollectionName(characterName)

    let totalChunks = 0
    let savedChunks = 0
    let skippedChunks = 0

    // Process each chat file
    for (let i = 0; i < chatFiles.length; i++) {
      if (cancelled) break

      const chatFile = chatFiles[i]
      const progress = ((i / chatFiles.length) * 100).toFixed(0)

      $("#qdrant_index_progress").css("width", `${progress}%`)
      $("#qdrant_index_status").text(`處理對話 ${i + 1}/${chatFiles.length}`)
      $("#qdrant_index_details").text(`檔案: ${chatFile}`)

      // Load chat file
      const chatData = await loadChatFile(characterName, chatFile)
      if (!chatData || !Array.isArray(chatData)) continue

      // Create chunks from messages
      const chunks = createChunksFromChat(chatData, characterName)
      totalChunks += chunks.length

      // Save each chunk
      for (const chunk of chunks) {
        if (cancelled) break

        // Check if chunk already exists
        const exists = await chunkExists(collectionName, chunk.messageIds)
        if (exists) {
          skippedChunks++
          continue
        }

        // Get participants (for group chats)
        const participants = [characterName]

        // Save chunk
        const success = await saveChunkToQdrant(chunk, participants)
        if (success) {
          savedChunks++
        }

        $("#qdrant_index_details").text(`已儲存: ${savedChunks} | 已跳過: ${skippedChunks} | 總計: ${totalChunks}`)
      }
    }

    // Complete
    $("#qdrant_index_progress").css("width", "100%")

    if (cancelled) {
      $("#qdrant_index_status").text("索引已取消")
      toastr.info(`取消前已索引 ${savedChunks} 個區塊`, "Qdrant Memory")
    } else {
      $("#qdrant_index_status").text("索引完成！")
      toastr.success(`已索引 ${savedChunks} 個新區塊，跳過 ${skippedChunks} 個既有區塊`, "Qdrant Memory")
    }

    setCancelButtonToClose()
  } catch (error) {
    console.error("[Qdrant Memory] Error indexing chats:", error)
    $("#qdrant_index_status").text("索引過程發生錯誤")
    $("#qdrant_index_details").text(error.message)
    toastr.error("對話索引失敗", "Qdrant Memory")
    setCancelButtonToClose()
  }
}

// ============================================================================
// GENERATION INTERCEPTOR
// ============================================================================

// FIXED: Prevent duplicate memory injection
globalThis.qdrantMemoryInterceptor = async (chat, contextSize, abort, type) => {
  if (!settings.enabled) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Extension disabled, skipping")
    }
    return
  }

  // NEW: Use chat hash instead of WeakMap
  if (settings.preventDuplicateInjection) {
    const chatHash = getChatHash(chat)
    
    if (memoryInjectionTracker.has(chatHash)) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Memories already injected for this chat state, skipping")
      }
      return
    }
    
    // Mark this chat state as having memories injected
    memoryInjectionTracker.add(chatHash)
    
    // Clean up old hashes to prevent memory leaks (keep last 50)
    if (memoryInjectionTracker.size > 50) {
      const oldestHash = memoryInjectionTracker.values().next().value
      memoryInjectionTracker.delete(oldestHash)
    }
  }

  try {
    const context = getContext()
    const characterName = context.name2

    // Skip if no character is selected
    if (!characterName) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No character selected, skipping")
      }
      return
    }

    // Find the last user message to use as the query
    const lastUserMsg = chat
      .slice()
      .reverse()
      .find((msg) => msg.is_user)
    if (!lastUserMsg || !lastUserMsg.mes) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No user message found, skipping")
      }
      return
    }

    const query = lastUserMsg.mes

    if (settings.debugMode) {
      console.log("[Qdrant Memory] Generation interceptor triggered")
      console.log("[Qdrant Memory] Type:", type)
      console.log("[Qdrant Memory] Context size:", contextSize)
      console.log("[Qdrant Memory] Searching for:", query)
      console.log("[Qdrant Memory] Character:", characterName)
    }

    // Search for relevant memories
    const memories = await searchMemories(query, characterName)

    if (memories.length > 0) {
      const memoryText = formatMemories(memories)

      // Cache for the {{qdrant}} macro (evaluated synchronously later).
      lastQdrantMemoryText = memoryText

      if (settings.debugMode) {
        console.log("[Qdrant Memory] Retrieved memories:", memoryText)
      }

      if (settings.useMacro) {
        // Macro mode: do NOT splice into chat. The user places {{qdrant}}
        // wherever they want (preset / character card / author's note) and
        // it is filled in from the cache above.
        if (settings.debugMode) {
          console.log(
            `[Qdrant Memory] Macro mode: ${memories.length} memories exposed via {{qdrant}} (no chat injection)`,
          )
        }
      } else {
        // Create memory entry
        const memoryEntry = {
          name: "System",
          is_user: false,
          is_system: true,
          mes: memoryText,
          send_date: Date.now(),
        }

        // Insert memories at the specified position from the end
        const insertIndex = Math.max(0, chat.length - settings.memoryPosition)
        chat.splice(insertIndex, 0, memoryEntry)

        if (settings.debugMode) {
          console.log(`[Qdrant Memory] Injected ${memories.length} memories at position ${insertIndex}`)
        }
      }

      const toastr = window.toastr
      if (settings.showMemoryNotifications) {
        toastr.info(`已擷取 ${memories.length} 條相關記憶`, "Qdrant Memory", { timeOut: 2000 })
      }
    } else {
      // No relevant memories: clear the macro cache so a stale result from a
      // previous turn is not reused by {{qdrant}}.
      lastQdrantMemoryText = ""
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No relevant memories found")
      }
    }
  } catch (error) {
    console.error("[Qdrant Memory] Error in generation interceptor:", error)
  }
}

// ============================================================================
// AUTOMATIC MEMORY CREATION
// ============================================================================

function clearPendingAssistantFinalize() {
  if (pendingAssistantFinalize?.pollTimerId) {
    clearInterval(pendingAssistantFinalize.pollTimerId)
  }
  pendingAssistantFinalize = null
}

function scheduleFinalizeLastAssistantMessage(messageId, characterName) {
  const context = getContext()
  const chat = context.chat || []

  if (chat.length === 0) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] No chat messages to finalize")
    }
    return
  }

  const lastMessage = chat[chat.length - 1]
  
  // Safety check: make sure this is actually a character message
  if (lastMessage.is_user) {
    if (settings.debugMode) {
      console.log("[Qdrant Memory] Last message is user message, skipping finalize")
    }
    return
  }
  
  const initialText = lastMessage?.mes || ""

  clearPendingAssistantFinalize()

  const pollInterval = settings.streamFinalizePollMs || 250
  const stableMs = settings.streamFinalizeStableMs || 1200
  const maxWaitMs = settings.streamFinalizeMaxWaitMs || 300000

  if (settings.debugMode) {
    console.log("[Qdrant Memory] Starting stream finalize poll for character message")
    console.log("[Qdrant Memory] Initial text length:", initialText.length)
  }

  pendingAssistantFinalize = {
    messageId,
    characterName,
    startedAt: Date.now(),
    lastText: initialText,
    lastChangeAt: Date.now(),
    pollTimerId: null,
  }

  const finalizeAssistant = (text, reason) => {
    if (!text || text.trim().length === 0) {
      if (settings.debugMode) {
        console.warn("[Qdrant Memory] Attempted to finalize empty message, skipping")
      }
      clearPendingAssistantFinalize()
      return
    }

    if (settings.debugMode) {
      console.log(`[Qdrant Memory] Finalizing character message (${reason})`)
      console.log(`[Qdrant Memory] Final text length: ${text.length}`)
      console.log(`[Qdrant Memory] Text preview: "${text.substring(0, 100)}..."`)
    }

    // Buffer the complete message
    bufferMessage(text, characterName, false, messageId)

    // Optionally flush the buffer if we have enough messages
    if (settings.flushAfterAssistant && messageBuffer.length >= 2) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Flushing buffer after assistant message")
      }
      processMessageBuffer()
    }

    clearPendingAssistantFinalize()
  }

  let pollCount = 0
  
  pendingAssistantFinalize.pollTimerId = setInterval(() => {
    pollCount++
    
    const currentContext = getContext()
    const currentChat = currentContext.chat || []
    
    if (currentChat.length === 0) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Chat is empty, cancelling finalize")
      }
      clearPendingAssistantFinalize()
      return
    }
    
    const currentLastMessage = currentChat[currentChat.length - 1] || {}
    const currentText = currentLastMessage.mes || ""
    const now = Date.now()

    // Log every 4 seconds (16 polls at 250ms) in debug mode
    if (settings.debugMode && pollCount % 16 === 0) {
      console.log(`[Qdrant Memory] Stream poll check #${pollCount}: length=${currentText.length}`)
    }

    // Detect if the text has changed
    if (currentText !== pendingAssistantFinalize.lastText) {
      if (settings.debugMode && pollCount % 4 === 0) {
        console.log(`[Qdrant Memory] Text changed: ${pendingAssistantFinalize.lastText.length} → ${currentText.length}`)
      }
      pendingAssistantFinalize.lastText = currentText
      pendingAssistantFinalize.lastChangeAt = now
    }

    const stableDuration = now - pendingAssistantFinalize.lastChangeAt
    const totalDuration = now - pendingAssistantFinalize.startedAt

    // Check if message switched to user (conversation continued)
    if (currentLastMessage.is_user) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Detected new user message, finalizing previous assistant message")
      }
      finalizeAssistant(pendingAssistantFinalize.lastText, "new user message detected")
      return
    }

    // Check if text has been stable for the required duration
    if (stableDuration >= stableMs) {
      finalizeAssistant(pendingAssistantFinalize.lastText, `stable for ${stableDuration}ms`)
      return
    }

    // Safety: max wait time exceeded
    if (totalDuration >= maxWaitMs) {
      if (settings.debugMode) {
        console.warn(`[Qdrant Memory] Max wait (${maxWaitMs}ms) reached while finalizing assistant message`)
      }
      finalizeAssistant(pendingAssistantFinalize.lastText, "max wait reached")
    }
  }, pollInterval)
}

function onMessageSent() {
  if (!settings.enabled) return
  if (!settings.autoSaveMemories) return

  try {
    const context = getContext()
    const chat = context.chat || []
    const characterName = context.name2

    if (!characterName || chat.length === 0) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] No character or empty chat, skipping save")
      }
      return
    }

    // Get the last message
    const lastMessage = chat[chat.length - 1]

    // Normalize send_date for messageId
    const normalizedDate = normalizeTimestamp(lastMessage.send_date || Date.now())
    
    // Create a unique ID for this message
    const messageId = `${characterName}_${normalizedDate}_${chat.length - 1}`

    if (!lastMessage.mes || lastMessage.mes.trim().length === 0) {
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Empty message, skipping")
      }
      return
    }

    const isUser = lastMessage.is_user || false
    
    if (settings.debugMode) {
      console.log(`[Qdrant Memory] onMessageSent - isUser: ${isUser}, length: ${lastMessage.mes.length}`)
    }
    
    if (isUser) {
      // User messages are saved immediately
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Buffering user message immediately")
      }
      bufferMessage(lastMessage.mes, characterName, true, messageId)
    } else {
      // Character messages need to wait for streaming to complete
      if (settings.debugMode) {
        console.log("[Qdrant Memory] Character message detected, scheduling finalize poll")
      }
      scheduleFinalizeLastAssistantMessage(messageId, characterName)
    }
  } catch (error) {
    console.error("[Qdrant Memory] Error in onMessageSent:", error)
  }
}

// ============================================================================
// MEMORY VIEWER FUNCTIONS
// ============================================================================

async function getCollectionInfo(collectionName) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeURIComponent(collectionName)}`, {
      headers: getQdrantHeaders(),
    })
    if (response.ok) {
      const data = await response.json()
      return data.result
    }
    return null
  } catch (error) {
    console.error("[Qdrant Memory] Error getting collection info:", error)
    return null
  }
}

async function deleteCollection(collectionName) {
  try {
    const response = await fetch(`${settings.qdrantUrl}/collections/${encodeURIComponent(collectionName)}`, {
      method: "DELETE",
      headers: getQdrantHeaders(),
    })
    return response.ok
  } catch (error) {
    console.error("[Qdrant Memory] Error deleting collection:", error)
    return false
  }
}

async function showMemoryViewer() {
  const context = getContext()
  const characterName = context.name2

  if (!characterName) {
    const toastr = window.toastr
    toastr.warning("尚未選擇角色", "Qdrant Memory")
    return
  }

  const collectionName = getCollectionName(characterName)
  const info = await getCollectionInfo(collectionName)

  if (!info) {
    const toastr = window.toastr
    toastr.warning(`找不到 ${characterName} 的記憶`, "Qdrant Memory")
    return
  }

  const count = info.points_count || 0

  const modalHtml = `
        <div id="qdrant_overlay" style="
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.5);
            z-index: 9999;
            overflow-y: auto;
            -webkit-overflow-scrolling: touch;
            box-sizing: border-box;
            padding: 5vh 16px calc(5vh + env(safe-area-inset-bottom)) 16px;
            text-align: center;
        ">
            <div id="qdrant_modal" style="
                display: inline-block;
                text-align: left;
                vertical-align: top;
                background: white;
                padding: clamp(16px, 5vw, 28px);
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                width: 100%;
                max-width: 500px;
                box-sizing: border-box;
            ">
                <div style="color: #333; word-break: break-all; overflow-wrap: anywhere;">
                    <h3 style="margin-top: 0;">記憶檢視 - ${characterName}</h3>
                    <p><strong>集合名稱:</strong> ${collectionName}</p>
                    <p><strong>記憶總數:</strong> ${count}</p>
                    <div style="margin-top: 20px;">
                        <button id="qdrant_delete_collection_btn" class="menu_button" style="background-color: #dc3545; color: white; min-height: 44px; padding: 10px 16px; width: 100%; margin-bottom: 10px;">
                            刪除所有記憶
                        </button>
                        <button id="qdrant_close_modal" class="menu_button" style="min-height: 44px; padding: 10px 16px; width: 100%;">
                            關閉
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `

  const $ = window.$
  // Attach to body so position:fixed + inset:0 resolves against the viewport.
  // No flexbox: overlay is a full-screen scroll area, the white box is a
  // content-sized inline-block that cannot collapse to a strip.
  $("body").append(modalHtml)

  // Close on the close button, or on tapping the dark backdrop — but NOT when
  // tapping inside the modal itself (stop the bubble to the overlay).
  $("#qdrant_modal").on("click", (e) => e.stopPropagation())
  $("#qdrant_close_modal, #qdrant_overlay").on("click", () => {
    $("#qdrant_overlay").remove()
  })

  $("#qdrant_delete_collection_btn").on("click", async function () {
    const confirmed = confirm(
      `確定要刪除 ${characterName} 的所有記憶嗎？此操作無法復原！`,
    )
    if (confirmed) {
      $(this).prop("disabled", true).text("刪除中...")
      const success = await deleteCollection(collectionName)
      if (success) {
        const toastr = window.toastr
        toastr.success(`已刪除 ${characterName} 的所有記憶`, "Qdrant Memory")
        $("#qdrant_overlay").remove()
      } else {
        const toastr = window.toastr
        toastr.error("記憶刪除失敗", "Qdrant Memory")
        $(this).prop("disabled", false).text("刪除所有記憶")
      }
    }
  })
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function getContext() {
  const SillyTavern = window.SillyTavern
  
  if (typeof SillyTavern !== "undefined" && SillyTavern.getContext) {
    return SillyTavern.getContext()
  }
  return {
    chat: window.chat || [],
    name2: window.name2 || "",
    characters: window.characters || [],
  }
}

function getPersonaName() {
  const context = getContext()
  
  // Try multiple possible locations for persona name in order of preference
  const personaName = 
    context.name1 ||                              // Standard SillyTavern location
    context.persona?.name ||                      // Alternative location
    window.name1 ||                               // Direct window access
    window.SillyTavern?.getContext?.()?.name1 ||  // Through ST API
    "User"                                        // Fallback to generic "User"
  
  if (settings.debugMode && personaName !== "User") {
    console.log(`[Qdrant Memory] Using persona name: ${personaName}`)
  }
  
  return personaName
}

function generateUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    var r = (Math.random() * 16) | 0,
      v = c == "x" ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

async function processSaveQueue() {
  if (processingSaveQueue || saveQueue.length === 0) return
  processingSaveQueue = true
  while (saveQueue.length > 0) {
    saveQueue.shift()
  }
  processingSaveQueue = false
}

// ============================================================================
// SETTINGS UI
// ============================================================================

function createSettingsUI() {
  const settingsHtml = `
        <div class="qdrant-memory-settings">
            <div class="inline-drawer">
                <div class="inline-drawer-toggle inline-drawer-header">
                    <b>Qdrant 記憶（Qdrant Memory）</b>
                    <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
                </div>
                <div class="inline-drawer-content">
                    <p style="margin: 10px 0; color: #666; font-size: 0.9em;">
                        自動建立帶有時間脈絡的記憶
                    </p>
                    
                    <div style="margin: 15px 0;">
                        <label style="display: flex; align-items: center; gap: 10px;">
                            <input type="checkbox" id="qdrant_enabled" ${settings.enabled ? "checked" : ""} />
                            <strong>啟用 Qdrant Memory</strong>
                        </label>
                    </div>
            
            <hr style="margin: 15px 0;" />
            
            <h4>連線設定</h4>
            
            <div style="margin: 10px 0;">
                <label><strong>Qdrant URL:</strong></label>
                <input type="text" id="qdrant_url" class="text_pole" value="${settings.qdrantUrl}" 
                       style="width: 100%; margin-top: 5px;" 
                       placeholder="http://localhost:6333" />
                <small style="color: #666;">你的 Qdrant 實例網址</small>
            </div>

             <div style="margin: 10px 0;">
                <label><strong>Qdrant API 金鑰:</strong></label>
                <input type="password" id="qdrant_api_key" class="text_pole" value="${settings.qdrantApiKey || ""}"
                       style="width: 100%; margin-top: 5px;"
                       placeholder="選填，不需要請留空" />
                <small style="color: #666;">Qdrant 驗證用 API 金鑰（選填）</small>
            </div>
            
            <div style="margin: 10px 0;">
                <label><strong>基本集合名稱:</strong></label>
                <input type="text" id="qdrant_collection" class="text_pole" value="${settings.collectionName}"
                       style="width: 100%; margin-top: 5px;"
                       placeholder="sillytavern_memories" />
                <small style="color: #666;">集合的基本名稱（角色名會附加在後）</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>嵌入提供者:</strong></label>
                <select id="qdrant_embedding_provider" class="text_pole" style="width: 100%; margin-top: 5px;">
                    <option value="openai" ${settings.embeddingProvider === "openai" ? "selected" : ""}>OpenAI</option>
                    <option value="openrouter" ${settings.embeddingProvider === "openrouter" ? "selected" : ""}>OpenRouter</option>
                    <option value="google" ${settings.embeddingProvider === "google" ? "selected" : ""}>Google AI</option>
                    <option value="local" ${settings.embeddingProvider === "local" ? "selected" : ""}>本地 / 自訂端點</option>
                </select>
                <small style="color: #666;">選擇用於產生嵌入向量的 API</small>
            </div>

            <div id="qdrant_openai_key_group" style="margin: 10px 0;">
                <label><strong>OpenAI API 金鑰:</strong></label>
                <input type="password" id="qdrant_openai_key" class="text_pole" value="${settings.openaiApiKey}"
                       placeholder="sk-..." style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">使用 OpenAI 時必填</small>
            </div>

            <div id="qdrant_openrouter_key_group" style="margin: 10px 0; display: none;">
                <label><strong>OpenRouter API 金鑰:</strong></label>
                <input type="password" id="qdrant_openrouter_key" class="text_pole" value="${settings.openRouterApiKey}"
                       placeholder="or-..." style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">使用 OpenRouter 時必填</small>
            </div>

            <div id="qdrant_google_key_group" style="margin: 10px 0; display: none;">
                <label><strong>Google AI API 金鑰:</strong></label>
                <input type="password" id="qdrant_google_key" class="text_pole" value="${settings.googleApiKey}"
                       placeholder="AIza..." style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">從 <a href="https://aistudio.google.com/apikey" target="_blank">Google AI Studio</a> 取得，可免費使用</small>
                <div style="margin-top: 8px;">
                    <button id="qdrant_google_fetch_models" class="menu_button" type="button" style="width: 100%;">
                        🔄 重新整理可用模型清單
                    </button>
                    <small id="qdrant_google_fetch_status" style="color: #666; display: block; margin-top: 4px;">
                        ${
                          Array.isArray(settings.googleEmbeddingModels) && settings.googleEmbeddingModels.length
                            ? `已快取 ${settings.googleEmbeddingModels.length} 個模型`
                            : "尚未抓取，將使用內建預設清單"
                        }
                    </small>
                </div>
            </div>

            <div id="qdrant_local_url_group" style="margin: 10px 0; display: none;">
                <label><strong>嵌入 URL:</strong></label>
                <input type="text" id="qdrant_local_url" class="text_pole" value="${settings.localEmbeddingUrl}"
                       placeholder="http://localhost:11434/api/embeddings"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">可接受 OpenAI 相容嵌入請求的端點</small>
            </div>

            <div id="qdrant_local_api_key_group" style="margin: 10px 0; display: none;">
                <label><strong>嵌入 API 金鑰（選填）:</strong></label>
                <input type="password" id="qdrant_local_api_key" class="text_pole" value="${settings.localEmbeddingApiKey}"
                       placeholder="本地端點的 Bearer token"
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">若本地 / 自訂端點需要驗證則填入</small>
            </div>

            <div id="qdrant_local_dimensions_group" style="margin: 10px 0; display: none;">
                <label><strong>嵌入維度:</strong></label>
                <input type="number" id="qdrant_local_dimensions" class="text_pole"
                       value="${settings.customEmbeddingDimensions ?? ""}"
                       min="1" step="1" style="width: 100%; margin-top: 5px;" placeholder="首次呼叫後自動偵測" />
                <small style="color: #666;">自訂嵌入模型回傳的向量大小（留空則自動偵測）</small>
            </div>

            <div id="qdrant_embedding_model_group" style="margin: 10px 0;">
                <label><strong>嵌入模型:</strong></label>
                <select id="qdrant_embedding_model" class="text_pole" style="width: 100%; margin-top: 5px;">
                    <option value="text-embedding-3-large" ${settings.embeddingModel === "text-embedding-3-large" ? "selected" : ""}>text-embedding-3-large（品質最佳）</option>
                    <option value="text-embedding-3-small" ${settings.embeddingModel === "text-embedding-3-small" ? "selected" : ""}>text-embedding-3-small（速度較快）</option>
                    <option value="text-embedding-ada-002" ${settings.embeddingModel === "text-embedding-ada-002" ? "selected" : ""}>text-embedding-ada-002（舊版）</option>
                </select>
            </div>
            
            <hr style="margin: 15px 0;" />
            
            <h4>記憶擷取設定</h4>
            
            <div style="margin: 10px 0;">
                <label><strong>記憶數量:</strong> <span id="memory_limit_display">${settings.memoryLimit}</span></label>
                <input type="range" id="qdrant_memory_limit" min="1" max="50" value="${settings.memoryLimit}" 
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">每次生成擷取的最大記憶數</small>
            </div>
            
            <div style="margin: 10px 0;">
                <label><strong>相關度閾值:</strong> <span id="score_threshold_display">${settings.scoreThreshold}</span></label>
                <input type="range" id="qdrant_score_threshold" min="0" max="1" step="0.05" value="${settings.scoreThreshold}" 
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">最低相似度分數（0.0 - 1.0）</small>
            </div>
            
            <div style="margin: 10px 0;">
                <label><strong>記憶插入位置:</strong> <span id="memory_position_display">${settings.memoryPosition}</span></label>
                <input type="range" id="qdrant_memory_position" min="1" max="30" value="${settings.memoryPosition}" 
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">從對話尾端往前數第幾則訊息插入記憶</small>
            </div>

            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 8px;">
                    <input type="checkbox" id="qdrant_use_macro" ${settings.useMacro ? "checked" : ""} />
                    <strong>改用 {{qdrant}} 巨集插入記憶</strong>
                </label>
                <small style="color: #666; display: block; margin-top: 4px;">
                    勾選後不再自動插入聊天，改由你在預設／角色卡／作者註記裡放的
                    <code>{{qdrant}}</code> 標記決定記憶出現的位置（上方的「記憶插入位置」會被忽略）。
                    無相關記憶時 <code>{{qdrant}}</code> 會替換成空字串。
                </small>
            </div>
            
            <div style="margin: 10px 0;">
                <label><strong>保留最近訊息數:</strong> <span id="retain_recent_display">${settings.retainRecentMessages}</span></label>
                <input type="range" id="qdrant_retain_recent" min="0" max="50" value="${settings.retainRecentMessages}" 
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">從擷取結果中排除最後 N 則訊息（0 = 不排除）</small>
            </div>
            
            <hr style="margin: 15px 0;" />
            
            <h4>自動建立記憶</h4>
            
            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_per_character" ${settings.usePerCharacterCollections ? "checked" : ""} />
                    <strong>每個角色使用獨立集合</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">每個角色擁有專屬的集合（建議啟用）</small>
            </div>
            
            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_auto_save" ${settings.autoSaveMemories ? "checked" : ""} />
                    <strong>自動儲存記憶</strong>
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">在對話進行時自動儲存訊息到 Qdrant</small>
            </div>
            
            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_save_user" ${settings.saveUserMessages ? "checked" : ""} />
                    儲存使用者訊息
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_save_character" ${settings.saveCharacterMessages ? "checked" : ""} />
                    儲存角色訊息
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label><strong>最小訊息長度:</strong> <span id="min_message_length_display">${settings.minMessageLength}</span></label>
                <input type="range" id="qdrant_min_length" min="5" max="50" value="${settings.minMessageLength}" 
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">儲存訊息所需的最少字元數</small>
            </div>

            <div style="margin: 10px 0;">
                <label><strong>去重閾值:</strong> <span id="dedupe_threshold_display">${settings.dedupeThreshold}</span></label>
                <input type="range" id="qdrant_dedupe_threshold" min="0.80" max="1.00" step="0.01" value="${settings.dedupeThreshold}" 
                       style="width: 100%; margin-top: 5px;" />
                <small style="color: #666;">避免儲存重複區塊（值越高越嚴格）</small>
            </div>
            
            <hr style="margin: 15px 0;" />
            
            <h4>其他設定</h4>
            
            <div style="margin: 15px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_prevent_duplicate" ${settings.preventDuplicateInjection ? "checked" : ""} />
                    避免重複注入記憶
                </label>
                <small style="color: #666; display: block; margin-left: 30px;">避免同一筆記憶被多次加入上下文</small>
            </div>
            
            <div style="margin: 15px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_notifications" ${settings.showMemoryNotifications ? "checked" : ""} />
                    顯示記憶提示通知
                </label>
            </div>
            
            <div style="margin: 15px 0;">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="qdrant_debug" ${settings.debugMode ? "checked" : ""} />
                    除錯模式（請查看主控台）
                </label>
            </div>
            
            <hr style="margin: 15px 0;" />
            
            <div style="margin: 15px 0; display: flex; gap: 10px; flex-wrap: wrap;">
                <button id="qdrant_test" class="menu_button">測試連線</button>
                <button id="qdrant_save" class="menu_button">儲存設定</button>
                <button id="qdrant_view_memories" class="menu_button">檢視記憶</button>
                <button id="qdrant_index_chats" class="menu_button" style="background-color: #28a745; color: white;">索引角色對話</button>
            </div>
            
            <div id="qdrant_status" style="margin-top: 10px; padding: 10px; border-radius: 5px;"></div>
                </div>
            </div>
        </div>
    `

  const $ = window.$
  $("#extensions_settings2").append(settingsHtml)

  if (typeof window.applyInlineDrawerListeners === "function") {
    window.applyInlineDrawerListeners()
  }

  function updateEmbeddingModelOptions(provider) {
    let models = EMBEDDING_MODEL_OPTIONS[provider] || EMBEDDING_MODEL_OPTIONS.openai

    // For google, prefer dynamically-fetched models if cached
    if (provider === "google" && Array.isArray(settings.googleEmbeddingModels) && settings.googleEmbeddingModels.length) {
      models = settings.googleEmbeddingModels
    }

    const $modelSelect = $("#qdrant_embedding_model")
    let previousValue = settings.embeddingModel
    if (provider === "openrouter" && OPENROUTER_MODEL_ALIASES[previousValue]) {
      previousValue = OPENROUTER_MODEL_ALIASES[previousValue]
      settings.embeddingModel = previousValue
    } else if (provider === "openai" && OPENAI_MODEL_ALIASES[previousValue]) {
      previousValue = OPENAI_MODEL_ALIASES[previousValue]
      settings.embeddingModel = previousValue
    }
    let matched = false

    $modelSelect.empty()

    models.forEach((model) => {
      const isSelected = model.value === previousValue
      if (isSelected) {
        matched = true
      }

      const optionHtml = `<option value="${model.value}"${
        isSelected ? " selected" : ""
      }>${model.label}</option>`
      $modelSelect.append(optionHtml)
    })

    if (!matched && models.length > 0) {
      const fallback = DEFAULT_MODEL_BY_PROVIDER[provider] || models[0].value
      // If fallback isn't in the list, use first available
      const finalFallback = models.find((m) => m.value === fallback) ? fallback : models[0].value
      settings.embeddingModel = finalFallback
      $modelSelect.val(settings.embeddingModel)
    }
  }

  function updateEmbeddingProviderUI() {
    const provider = settings.embeddingProvider || "openai"
    const $openAIGroup = $("#qdrant_openai_key_group")
    const $openRouterGroup = $("#qdrant_openrouter_key_group")
    const $googleGroup = $("#qdrant_google_key_group")
    const $localGroup = $("#qdrant_local_url_group")
    const $localApiKeyGroup = $("#qdrant_local_api_key_group")
    const $localDimensionsGroup = $("#qdrant_local_dimensions_group")
    const $localDimensionsInput = $("#qdrant_local_dimensions")
    const $modelGroup = $("#qdrant_embedding_model_group")

    $openAIGroup.toggle(provider === "openai")
    $openRouterGroup.toggle(provider === "openrouter")
    $googleGroup.toggle(provider === "google")
    $localGroup.toggle(provider === "local")
    $localApiKeyGroup.toggle(provider === "local")
    $localDimensionsGroup.toggle(provider === "local")

    if (provider === "local") {
      $localDimensionsInput.val(settings.customEmbeddingDimensions ?? "")
    }

    const showModelSelect = provider !== "local"
    $modelGroup.toggle(showModelSelect)

    if (showModelSelect) {
      updateEmbeddingModelOptions(provider)
    }
  }

  // Event handlers
  $("#qdrant_enabled").on("change", function () {
    settings.enabled = $(this).is(":checked")
  })

  $("#qdrant_url").on("input", function () {
    settings.qdrantUrl = $(this).val()
  })

  $("#qdrant_api_key").on("input", function () {
    settings.qdrantApiKey = $(this).val()
  })

  $("#qdrant_collection").on("input", function () {
    settings.collectionName = $(this).val()
  })

  $("#qdrant_embedding_provider").on("change", function () {
    settings.embeddingProvider = $(this).val()
    updateEmbeddingProviderUI()
  })

  $("#qdrant_openai_key").on("input", function () {
    settings.openaiApiKey = $(this).val()
  })

  $("#qdrant_openrouter_key").on("input", function () {
    settings.openRouterApiKey = $(this).val()
  })

  $("#qdrant_google_key").on("input", function () {
    settings.googleApiKey = $(this).val()
  })

  $("#qdrant_google_fetch_models").on("click", async function () {
    const $btn = $(this)
    const $status = $("#qdrant_google_fetch_status")
    const apiKey = ($("#qdrant_google_key").val() || "").trim()

    if (!apiKey) {
      $status.text("✗ 請先填入 Google API 金鑰").css("color", "#721c24")
      return
    }

    $btn.prop("disabled", true)
    const originalText = $btn.text()
    $btn.text("抓取中…")
    $status.text("正在向 Google AI 查詢可用模型…").css("color", "#004085")

    try {
      const fetched = await fetchGoogleEmbeddingModels(apiKey)
      if (!fetched.length) {
        $status.text("⚠️ 未找到支援 embedContent 的模型").css("color", "#856404")
      } else {
        settings.googleEmbeddingModels = fetched
        // also sync key (user might have typed but not blurred)
        settings.googleApiKey = apiKey
        saveSettings()
        // re-render model select if currently on google
        if (settings.embeddingProvider === "google") {
          updateEmbeddingModelOptions("google")
        }
        $status
          .text(`✓ 抓到 ${fetched.length} 個可用模型，已更新清單`)
          .css("color", "green")
      }
    } catch (err) {
      console.error("[Qdrant Memory] fetchGoogleEmbeddingModels error:", err)
      $status.text(`✗ ${err.message || err}`).css("color", "#721c24")
    } finally {
      $btn.prop("disabled", false).text(originalText)
    }
  })

  $("#qdrant_local_url").on("input", function () {
    settings.localEmbeddingUrl = $(this).val()
  })

  $("#qdrant_local_api_key").on("input", function () {
    settings.localEmbeddingApiKey = $(this).val()
  })

  $("#qdrant_local_dimensions").on("input", function () {
    const value = Number.parseInt($(this).val(), 10)
    settings.customEmbeddingDimensions = Number.isFinite(value) && value > 0 ? value : null
  })

  $("#qdrant_embedding_model").on("change", function () {
    settings.embeddingModel = $(this).val()
  })

  $("#qdrant_memory_limit").on("input", function () {
    settings.memoryLimit = Number.parseInt($(this).val())
    $("#memory_limit_display").text(settings.memoryLimit)
  })

  $("#qdrant_score_threshold").on("input", function () {
    settings.scoreThreshold = Number.parseFloat($(this).val())
    $("#score_threshold_display").text(settings.scoreThreshold)
  })

  $("#qdrant_memory_position").on("input", function () {
    settings.memoryPosition = Number.parseInt($(this).val())
    $("#memory_position_display").text(settings.memoryPosition)
  })

  $("#qdrant_use_macro").on("change", function () {
    settings.useMacro = $(this).is(":checked")
  })

  $("#qdrant_retain_recent").on("input", function () {
    settings.retainRecentMessages = Number.parseInt($(this).val())
    $("#retain_recent_display").text(settings.retainRecentMessages)
  })

  $("#qdrant_per_character").on("change", function () {
    settings.usePerCharacterCollections = $(this).is(":checked")
  })

  $("#qdrant_auto_save").on("change", function () {
    settings.autoSaveMemories = $(this).is(":checked")
  })

  $("#qdrant_save_user").on("change", function () {
    settings.saveUserMessages = $(this).is(":checked")
  })

  $("#qdrant_save_character").on("change", function () {
    settings.saveCharacterMessages = $(this).is(":checked")
  })

  $("#qdrant_min_length").on("input", function () {
    settings.minMessageLength = Number.parseInt($(this).val())
    $("#min_message_length_display").text(settings.minMessageLength)
  })

  $("#qdrant_dedupe_threshold").on("input", function () {
    settings.dedupeThreshold = Number.parseFloat($(this).val())
    $("#dedupe_threshold_display").text(settings.dedupeThreshold.toFixed(2))
  })

  $("#qdrant_prevent_duplicate").on("change", function () {
    settings.preventDuplicateInjection = $(this).is(":checked")
  })

  $("#qdrant_notifications").on("change", function () {
    settings.showMemoryNotifications = $(this).is(":checked")
  })

  $("#qdrant_debug").on("change", function () {
    settings.debugMode = $(this).is(":checked")
  })

  updateEmbeddingProviderUI()

  $("#qdrant_save").on("click", () => {
    saveSettings()
    $("#qdrant_status")
      .text("✓ 設定已儲存！")
      .css({ color: "green", background: "#d4edda", border: "1px solid green" })
    setTimeout(() => $("#qdrant_status").text("").css({ background: "", border: "" }), 3000)
  })

  $("#qdrant_test").on("click", async () => {
    $("#qdrant_status")
      .text("正在測試連線...")
      .css({ color: "#004085", background: "#cce5ff", border: "1px solid #004085" })

    try {
      const response = await fetch(`${settings.qdrantUrl}/collections`, {
        headers: getQdrantHeaders(),
      })

      if (response.ok) {
        const data = await response.json()
        const collections = data.result?.collections || []
        $("#qdrant_status")
          .text(`✓ 連線成功！找到 ${collections.length} 個集合。`)
          .css({ color: "green", background: "#d4edda", border: "1px solid green" })
      } else {
        $("#qdrant_status")
          .text("✗ 連線失敗，請檢查 URL。")
          .css({ color: "#721c24", background: "#f8d7da", border: "1px solid #721c24" })
      }
    } catch (error) {
      $("#qdrant_status")
        .text(`✗ 錯誤: ${error.message}`)
        .css({ color: "#721c24", background: "#f8d7da", border: "1px solid #721c24" })
    }
  })

  $("#qdrant_view_memories").on("click", () => {
    showMemoryViewer()
  })

  $("#qdrant_index_chats").on("click", () => {
    indexCharacterChats()
  })
}

// ============================================================================
// EXTENSION INITIALIZATION
// ============================================================================

window.jQuery(async () => {
  loadSettings()
  createSettingsUI()

  // Register the {{qdrant}} macro. It returns the memory text cached by the
  // generation interceptor (which performs the async Qdrant search before
  // the prompt is built, so the value is ready by macro-evaluation time).
  try {
    const ctx = getContext()
    const macroValue = () => (settings.enabled ? lastQdrantMemoryText || "" : "")

    if (ctx && typeof ctx.registerMacro === "function") {
      ctx.registerMacro("qdrant", macroValue)
      console.log("[Qdrant Memory] Registered {{qdrant}} macro via context")
    } else if (window.MacrosParser && typeof window.MacrosParser.registerMacro === "function") {
      window.MacrosParser.registerMacro("qdrant", macroValue)
      console.log("[Qdrant Memory] Registered {{qdrant}} macro via MacrosParser")
    } else {
      console.warn(
        "[Qdrant Memory] registerMacro not available in this SillyTavern version; {{qdrant}} macro disabled",
      )
    }
  } catch (e) {
    console.error("[Qdrant Memory] Failed to register {{qdrant}} macro:", e)
  }

  // Hook into message events for automatic saving
  const eventSource = window.eventSource
  if (typeof eventSource !== "undefined" && eventSource.on) {
    const handleMessageEvent = () => {
      if (!settings.enabled || !settings.autoSaveMemories) return
      onMessageSent()
    }

    eventSource.on("MESSAGE_RECEIVED", handleMessageEvent)
    eventSource.on("USER_MESSAGE_RENDERED", handleMessageEvent)
    console.log("[Qdrant Memory] Using eventSource hooks")
  } else {
    // Fallback: poll for new messages
    console.log("[Qdrant Memory] Using polling fallback for auto-save")
    let lastChatLength = 0
    setInterval(() => {
      const context = getContext()
      const chat = context.chat || []

      if (!settings.enabled || !settings.autoSaveMemories) {
        lastChatLength = chat.length
        return
      }

      if (chat.length > lastChatLength) {
        onMessageSent()
      }
      lastChatLength = chat.length
    }, 2000)
  }

  console.log("[Qdrant Memory] Extension loaded successfully (v3.1.3 - fixed partial memory storage during streaming)")
})
