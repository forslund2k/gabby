/* Gabby embeddable chat widget — no dependencies.
 * Served by /embed/<public_key>.js with __PUBLIC_KEY__, __API_BASE__,
 * __AGENT_NAME__ and __AGENT_COLOR__ injected server-side. */
(function () {
  var PUBLIC_KEY = "__PUBLIC_KEY__";
  var API_BASE = "__API_BASE__";
  var AGENT_NAME = "__AGENT_NAME__";
  var COLOR = "__AGENT_COLOR__" || "#4F46E5";
  var SUGGESTIONS = __SUGGESTIONS__;
  var GREETING = __GREETING__;

  var sessionToken = Math.random().toString(36).slice(2) + Date.now().toString(36);

  // --- styles ---------------------------------------------------------------
  var css = [
    "#gabby-bubble{position:fixed;bottom:20px;right:20px;width:60px;height:60px;border-radius:50%;",
    "background:" + COLOR + ";color:#fff;border:none;cursor:pointer;font-size:28px;z-index:999998;",
    "box-shadow:0 4px 14px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center;}",
    "#gabby-panel{position:fixed;bottom:92px;right:20px;width:340px;max-width:calc(100vw - 40px);",
    "height:460px;max-height:calc(100vh - 120px);background:#fff;border-radius:14px;z-index:999999;",
    "box-shadow:0 8px 30px rgba(0,0,0,.22);display:none;flex-direction:column;overflow:hidden;",
    "font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}",
    "#gabby-head{background:" + COLOR + ";color:#fff;padding:12px 14px;font-weight:600;font-size:15px;}",
    "#gabby-msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;}",
    ".gabby-msg{max-width:82%;padding:8px 12px;border-radius:14px;font-size:14px;line-height:1.4;}",
    ".gabby-user{align-self:flex-end;background:" + COLOR + ";color:#fff;border-bottom-right-radius:4px;}",
    ".gabby-bot{align-self:flex-start;background:#f1f1f4;color:#222;border-bottom-left-radius:4px;}",
    "#gabby-form{display:flex;border-top:1px solid #e5e5e5;}",
    "#gabby-input{flex:1;border:none;padding:12px;font-size:14px;outline:none;}",
    "#gabby-send{background:" + COLOR + ";color:#fff;border:none;padding:0 16px;cursor:pointer;font-size:14px;}",
    "#gabby-chips{display:flex;flex-wrap:wrap;gap:6px;padding:0 12px 10px;}",
    ".gabby-chip{border:1px solid " + COLOR + ";color:" + COLOR + ";background:#fff;border-radius:16px;",
    "padding:5px 12px;font-size:12.5px;cursor:pointer;}",
    ".gabby-chip:hover{background:" + COLOR + ";color:#fff;}"
  ].join("\n");
  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  // --- DOM ------------------------------------------------------------------
  var bubble = document.createElement("button");
  bubble.id = "gabby-bubble";
  bubble.innerHTML = "&#128172;";
  bubble.setAttribute("aria-label", "Chat with us");
  document.body.appendChild(bubble);

  var panel = document.createElement("div");
  panel.id = "gabby-panel";
  panel.innerHTML =
    '<div id="gabby-head">' + AGENT_NAME + "</div>" +
    '<div id="gabby-msgs"></div>' +
    '<div id="gabby-chips"></div>' +
    '<form id="gabby-form"><input id="gabby-input" placeholder="Type a message&hellip;" autocomplete="off"/>' +
    '<button id="gabby-send" type="submit">Send</button></form>';
  document.body.appendChild(panel);

  var msgs = panel.querySelector("#gabby-msgs");
  var form = panel.querySelector("#gabby-form");
  var input = panel.querySelector("#gabby-input");
  var chipsBox = panel.querySelector("#gabby-chips");

  function sendText(text) {
    text = (text || "").trim();
    if (!text) return;
    addMsg(text, "gabby-user");
    chipsBox.style.display = "none";
    fetch(API_BASE + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ public_key: PUBLIC_KEY, session_token: sessionToken, message: text })
    })
      .then(function (r) { return r.json(); })
      .then(function (data) { addMsg(data.reply || "Sorry, something went wrong.", "gabby-bot"); })
      .catch(function () { addMsg("Sorry, I couldn't reach the server.", "gabby-bot"); });
  }

  function renderChips() {
    chipsBox.innerHTML = "";
    if (!SUGGESTIONS || !SUGGESTIONS.length) { chipsBox.style.display = "none"; return; }
    chipsBox.style.display = "flex";
    SUGGESTIONS.forEach(function (q) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "gabby-chip";
      b.textContent = q;
      b.onclick = function () { sendText(q); };
      chipsBox.appendChild(b);
    });
  }

  function addMsg(text, cls) {
    var d = document.createElement("div");
    d.className = "gabby-msg " + cls;
    d.textContent = text;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  }

  bubble.onclick = function () {
    var open = panel.style.display === "flex";
    panel.style.display = open ? "none" : "flex";
    if (!open && !msgs.children.length) {
      addMsg(GREETING, "gabby-bot");
      renderChips();
    }
    if (!open) input.focus();
  };

  form.onsubmit = function (e) {
    e.preventDefault();
    var text = input.value.trim();
    input.value = "";
    sendText(text);
  };
})();
