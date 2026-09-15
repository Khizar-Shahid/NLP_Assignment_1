/*
 * Chat client.
 * Connects to /ws/chat, sends user messages as JSON, and renders streamed
 * tokens into the assistant's bubble as they arrive.
 */
(function () {
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const resetBtn = document.getElementById("reset");
  const statusEl = document.getElementById("status");
  const statusText = document.getElementById("status-text");

  let ws = null;
  let sessionId = null;
  let streamingBubble = null; // the assistant bubble currently being filled
  let awaitingReply = false;

  // ---- rendering ----------------------------------------------------------
  function showEmptyState() {
    messagesEl.innerHTML = `
      <div class="empty">
        <h1>Hey, I'm Coach 👋</h1>
        <p>Ask me about membership plans, class schedules, trainers, or freezing
        your membership at PulsePoint Fitness.</p>
      </div>`;
  }

  function clearEmptyState() {
    const empty = messagesEl.querySelector(".empty");
    if (empty) empty.remove();
  }

  function addBubble(role, text) {
    clearEmptyState();
    const row = document.createElement("div");
    row.className = "row " + role;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
    return bubble;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setStatus(state, label) {
    statusEl.className = "status " + state;
    statusText.textContent = label;
  }

  function setBusy(busy) {
    awaitingReply = busy;
    sendBtn.disabled = busy;
  }

  // ---- websocket ----------------------------------------------------------
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/chat`);

    ws.onopen = () => setStatus("online", "online");

    ws.onclose = () => {
      setStatus("offline", "reconnecting…");
      setBusy(false);
      setTimeout(connect, 1500); // auto-reconnect
    };

    ws.onerror = () => setStatus("offline", "offline");

    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      switch (msg.type) {
        case "session":
          sessionId = msg.session_id;
          break;

        case "start":
          streamingBubble = addBubble("assistant", "");
          streamingBubble.innerHTML = '<span class="caret"></span>';
          break;

        case "token":
          if (!streamingBubble) streamingBubble = addBubble("assistant", "");
          // Append text before the blinking caret.
          const caret = streamingBubble.querySelector(".caret");
          const textNode = document.createTextNode(msg.content);
          if (caret) streamingBubble.insertBefore(textNode, caret);
          else streamingBubble.appendChild(textNode);
          scrollToBottom();
          break;

        case "done": {
          if (streamingBubble) {
            streamingBubble.textContent = msg.content;
            streamingBubble = null;
          }
          setBusy(false);
          break;
        }

        case "reset_ack":
          sessionId = msg.session_id;
          messagesEl.innerHTML = "";
          showEmptyState();
          setBusy(false);
          break;

        case "error":
          if (streamingBubble) {
            streamingBubble.parentElement.remove();
            streamingBubble = null;
          }
          addBubble("error", "⚠ " + msg.message);
          setBusy(false);
          break;
      }
    };
  }

  // ---- actions ------------------------------------------------------------
  function send() {
    const text = inputEl.value.trim();
    if (!text || awaitingReply || !ws || ws.readyState !== WebSocket.OPEN) return;
    addBubble("user", text);
    ws.send(
      JSON.stringify({ type: "user_message", content: text, session_id: sessionId })
    );
    inputEl.value = "";
    autoGrow();
    setBusy(true);
  }

  function reset() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reset", session_id: sessionId }));
    } else {
      messagesEl.innerHTML = "";
      showEmptyState();
    }
  }

  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  }

  // ---- wiring -------------------------------------------------------------
  sendBtn.addEventListener("click", send);
  resetBtn.addEventListener("click", reset);
  inputEl.addEventListener("input", autoGrow);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  showEmptyState();
  connect();
})();
