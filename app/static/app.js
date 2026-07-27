// Web Search Agent // Frontend Logic

document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatMessages = document.getElementById("chat-messages");
    const newSessionBtn = document.getElementById("new-session-btn");

    // Initialize or load Session ID from LocalStorage
    let currentThreadId = localStorage.getItem("wsa_thread_id");
    if (!currentThreadId) {
        currentThreadId = generateUUID();
        localStorage.setItem("wsa_thread_id", currentThreadId);
    }

    // ── New Chat Button ─────────────────────────────────────────────────────
    newSessionBtn.addEventListener("click", () => {
        currentThreadId = generateUUID();
        localStorage.setItem("wsa_thread_id", currentThreadId);
        
        // Reset chat display with minimal welcome message
        chatMessages.innerHTML = `
            <div class="message agent-message">
                <div class="avatar agent-avatar">WS</div>
                <div class="message-content">
                    <div class="markdown-body">
                        <p><strong>Welcome to Web Search Agent.</strong></p>
                        <p>Started a new research session. Ask any question to search and analyze web content.</p>
                    </div>
                </div>
            </div>
        `;
    });

    // ── Form Submission ─────────────────────────────────────────────────────
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = userInput.value.trim();
        if (!question) return;

        // Clear input and append user message
        userInput.value = "";
        appendUserMessage(question);

        // Show minimal loading indicator
        const loadingId = appendLoadingIndicator();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    thread_id: currentThreadId
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server error occurred");
            }

            const data = await response.json();
            
            // Remove loading indicator
            removeMessage(loadingId);

            // Append agent response with minimal Working accordion
            appendAgentMessage(data);

        } catch (error) {
            removeMessage(loadingId);
            appendErrorMessage(error.message);
        }
    });

    // ── UI Helper Functions ─────────────────────────────────────────────────

    function appendUserMessage(text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message user-message";
        msgDiv.innerHTML = `
            <div class="message-content">
                <p>${escapeHtml(text)}</p>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendLoadingIndicator() {
        const id = "loading-" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.className = "message agent-message";
        msgDiv.id = id;
        msgDiv.innerHTML = `
            <div class="avatar agent-avatar">WS</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <span>Working on your research query...</span>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function appendAgentMessage(data) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message agent-message";

        const accordionId = "acc-" + Date.now();
        const wasRewritten = data.rewritten_query && data.rewritten_query !== data.original_query;
        const memoryText = wasRewritten 
            ? `Resolved reference from conversation history: "${escapeHtml(data.rewritten_query)}"`
            : `Query executed directly without modification.`;

        const sourcesCount = data.rag_results ? data.rag_results.length : 0;
        const sourcesText = sourcesCount > 0
            ? `<pre><code>${escapeHtml(data.rag_results)}</code></pre>`
            : `<p>No external sources retrieved.</p>`;

        // Build the minimal Working accordion
        const accordionHTML = `
            <div class="working-accordion">
                <div class="working-toggle" onclick="toggleWorking('${accordionId}')">
                    <span>⚙️ Working (2 steps)</span>
                    <span>▾</span>
                </div>
                <div class="working-content" id="${accordionId}">
                    <div class="working-step">
                        <strong>1. Conversational Memory</strong>
                        <p>${memoryText}</p>
                    </div>
                    <div class="working-step">
                        <strong>2. Web Search & Content Extraction (${sourcesCount} chars)</strong>
                        ${sourcesText}
                    </div>
                </div>
            </div>
        `;

        // Render Markdown content
        const parsedMarkdown = marked.parse(data.final_answer || "No response generated.");

        msgDiv.innerHTML = `
            <div class="avatar agent-avatar">WS</div>
            <div class="message-content">
                ${accordionHTML}
                <div class="markdown-body">
                    ${parsedMarkdown}
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendErrorMessage(errText) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message agent-message";
        msgDiv.innerHTML = `
            <div class="avatar agent-avatar" style="background: #EF4444;">!</div>
            <div class="message-content">
                <div class="markdown-body">
                    <p style="color: #EF4444;"><strong>Execution Error:</strong> <code>${escapeHtml(errText)}</code></p>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function escapeHtml(unsafe) {
        if (!unsafe) return "";
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
});

// Global function for toggle accordion
window.toggleWorking = function(id) {
    const el = document.getElementById(id);
    if (el) {
        const isHidden = el.style.display === "none" || !el.style.display;
        el.style.display = isHidden ? "block" : "none";
    }
};
