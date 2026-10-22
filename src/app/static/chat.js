"use strict";

(() => {
  const config = JSON.parse(document.getElementById("chat-config").textContent);
  const form = document.getElementById("chat-form");
  const question = document.getElementById("question");
  const messages = document.getElementById("messages");
  const welcome = document.getElementById("welcome");
  const scroll = document.getElementById("chat-scroll");
  const mode = document.getElementById("search-mode");
  const topK = document.getElementById("top-k");
  const banner = document.getElementById("setup-banner");
  let busy = false;
  let messageNumber = 0;

  const node = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const icon = (name) => {
    const element = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    element.setAttribute("class", "icon");
    element.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#icon-${name}`);
    element.append(use);
    return element;
  };
  const scrollDown = () => { scroll.scrollTop = scroll.scrollHeight; };
  const resizeInput = () => {
    question.style.height = "auto";
    question.style.height = `${Math.min(question.scrollHeight, 140)}px`;
  };
  const fillQuestion = (text) => {
    if (busy) return;
    question.value = text;
    resizeInput();
    question.focus();
  };

  const modeLabels = { bm25: "Keyword search", hybrid: "Hybrid search", semantic: "Semantic search" };
  for (const value of config.modes) {
    const option = node("option", "", modeLabels[value]);
    option.value = value;
    option.selected = value === config.default_mode;
    mode.append(option);
  }

  const setBusy = (value) => {
    busy = value;
    messages.setAttribute("aria-busy", String(value));
    for (const control of document.querySelectorAll("#send, #question, #search-mode, #top-k, [data-prompt], [data-new-chat], .retry-button")) {
      control.disabled = value;
    }
  };

  async function checkHealth() {
    const label = document.getElementById("connection-label");
    const status = document.getElementById("connection-status");
    const recheck = document.getElementById("recheck");
    recheck.disabled = true;
    label.textContent = "Checking setup";
    try {
      const response = await fetch(config.health_url, { cache: "no-store" });
      const health = await response.json();
      const ready = response.ok && health.status === "ready";
      status.dataset.state = ready ? "ready" : "setup";
      label.textContent = ready ? "Ready to answer" : "Setup required";
      banner.hidden = ready;
      document.getElementById("setup-message").textContent = health.message || "The server is not ready. Check its configuration.";
    } catch {
      status.dataset.state = "offline";
      label.textContent = "Connection unavailable";
      banner.hidden = false;
      document.getElementById("setup-message").textContent = "Could not connect to the server. Make sure it is running, then check again.";
    } finally {
      recheck.disabled = false;
    }
  }

  function message(role, text) {
    const entry = node("article", `message-entry ${role}-message`);
    const label = node("div", "message-label");
    if (role === "assistant") label.append(icon("mark"));
    label.append(node("span", "", role === "user" ? "You" : "Grounded"));
    const body = node("div", "message-body", text);
    entry.append(label, body);
    messages.append(entry);
    return { entry, body };
  }

  function renderAnswer(answer, body) {
    // Only the server's final GroundedAnswer is rendered. All content is text,
    // including model output, source filenames, and quotes; no HTML is trusted.
    const supported = answer.status === "answered" && answer.evidence_status === "supported";
    body.append(node("span", `answer-badge${supported ? "" : " low"}`, supported ? "Evidence verified" : "Not enough evidence"));
    const prefix = `answer-${++messageNumber}`;
    const sourceTargets = new Map();
    let sourceList;

    if (answer.citations.length) {
      sourceList = node("details", "sources");
      sourceList.open = true;
      sourceList.append(node("summary", "", `${answer.citations.length} verified source${answer.citations.length === 1 ? "" : "s"}`));
      answer.citations.forEach((citation, index) => {
        const source = node("details", "source-card");
        source.id = `${prefix}-source-${index}`;
        const file = citation.source_path.split(/[\\/]/).pop();
        source.append(node("summary", "", `[${citation.citation_id}] ${file} · Page ${citation.source_page}`));
        source.append(node("blockquote", "", citation.quote));
        source.append(node("p", "source-metadata", `FAQ ${citation.faq_id} · Version ${citation.source_version}\n${citation.source_path}\nChunk ${citation.chunk_id}`));
        sourceTargets.set(citation.citation_id, source);
        sourceList.append(source);
      });
    }

    if (answer.claims.length) {
      for (const claim of answer.claims) {
        const paragraph = node("p", "claim", claim.statement);
        for (const citationId of claim.citation_ids) {
          const target = sourceTargets.get(citationId);
          if (!target) continue;
          const link = node("button", "citation-link", `[${citationId}]`);
          link.type = "button";
          link.setAttribute("aria-label", `Read source ${citationId}`);
          link.addEventListener("click", () => {
            sourceList.open = true;
            target.open = true;
            target.scrollIntoView({ block: "nearest", behavior: "smooth" });
            target.querySelector("summary").focus({ preventScroll: true });
          });
          paragraph.append(link);
        }
        body.append(paragraph);
      }
    } else {
      body.append(node("p", "answer-text", answer.answer));
    }
    if (answer.reason) body.append(node("p", "answer-reason", answer.reason));
    if (sourceList) body.append(sourceList);
  }

  async function submit() {
    const text = question.value.trim();
    if (busy || !text || !form.reportValidity()) return;
    const payload = { question: text, mode: mode.value, k: Number(topK.value) };
    welcome.hidden = true;
    message("user", text);
    question.value = "";
    resizeInput();
    const pending = message("assistant");
    const progress = node("div", "pending-line");
    const spinner = node("span", "spinner");
    spinner.setAttribute("aria-hidden", "true");
    progress.append(spinner, node("span", "", "Searching FAQs and checking the evidence…"));
    pending.body.append(progress);
    setBusy(true);
    scrollDown();

    try {
      const response = await fetch(config.chat_url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) {
        const detail = result.detail;
        throw new Error(Array.isArray(detail)
          ? detail.map((item) => item.msg).join("; ")
          : (typeof detail === "string" ? detail : detail?.message) || "The request failed. Try again in a moment.");
      }
      pending.body.replaceChildren();
      renderAnswer(result, pending.body);
    } catch (error) {
      pending.entry.classList.add("error-message");
      pending.body.replaceChildren(node("span", "answer-badge low", "Unable to answer"));
      pending.body.append(node("p", "answer-text", error.message || "A connection error occurred. Check the server and try again."));
      const retry = node("button", "retry-button", "Use this question again");
      retry.type = "button";
      retry.addEventListener("click", () => fillQuestion(text));
      pending.body.append(retry);
      void checkHealth();
    } finally {
      setBusy(false);
      scrollDown();
      question.focus({ preventScroll: true });
    }
  }

  form.addEventListener("submit", (event) => { event.preventDefault(); void submit(); });
  question.addEventListener("input", resizeInput);
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void submit();
    }
  });
  for (const button of document.querySelectorAll("[data-prompt]")) {
    button.addEventListener("click", () => fillQuestion(button.dataset.prompt));
  }
  for (const button of document.querySelectorAll("[data-new-chat]")) {
    button.addEventListener("click", () => {
      if (busy) return;
      messages.replaceChildren();
      welcome.hidden = false;
      question.value = "";
      resizeInput();
      scroll.scrollTop = 0;
      question.focus({ preventScroll: true });
    });
  }
  document.getElementById("recheck").addEventListener("click", () => { void checkHealth(); });
  void checkHealth();
})();
