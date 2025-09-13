const messagesEl = document.getElementById('messages');
const form = document.getElementById('composer');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const topkEl = document.getElementById('topk');
const clearBtn = document.getElementById('clear');
const llmEl = document.getElementById('llm');
const sourcesToggleEl = document.getElementById('toggle-sources');
const sourcesPanelEl = document.getElementById('sources');
const sourcesListEl = document.getElementById('sources-list');

const state = {
  messages: [], // { role: 'user'|'assistant', content: string, contexts?: [] }
};

function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c])); }

// Minimal Markdown → HTML converter with sanitization
function mdToHtml(md) {
  if (!md) return '';
  // Normalize newlines
  md = md.replace(/\r\n?/g, '\n');

  // Code blocks ```lang\n...```
  md = md.replace(/```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g, (m, lang, code) => {
    const safe = escapeHtml(code);
    const cls = lang ? ` class="lang-${escapeHtml(lang)}"` : '';
    return `<pre><code${cls}>${safe}</code></pre>`;
  });

  // Headings
  md = md.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>')
         .replace(/^##\s+(.+)$/gm, '<h2>$1</h2>')
         .replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');

  // Bold and italic (basic)
  md = md.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
         .replace(/__([^_]+)__/g, '<strong>$1</strong>')
         .replace(/\*([^*]+)\*/g, '<em>$1</em>')
         .replace(/_([^_]+)_/g, '<em>$1</em>');

  // Inline code
  md = md.replace(/`([^`]+)`/g, (m, code) => `<code>${escapeHtml(code)}</code>`);

  // Links [text](url)
  md = md.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, text, url) => {
    const t = escapeHtml(text);
    const u = escapeHtml(url);
    return `<a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`;
  });

  // Lists (very simple, contiguous lines starting - or * or 1.)
  md = md.replace(/(?:^|\n)([-*] .*(?:\n[-*] .*)+)/g, (m) => {
    const items = m.trim().split(/\n/).map(l => l.replace(/^[-*]\s+/, '').trim());
    return '\n<ul>' + items.map(i => `<li>${i}</li>`).join('') + '</ul>';
  });
  md = md.replace(/(?:^|\n)((?:\d+\. .*(?:\n\d+\. .*)+))/g, (m, grp) => {
    const items = grp.trim().split(/\n/).map(l => l.replace(/^\d+\.\s+/, '').trim());
    return '\n<ol>' + items.map(i => `<li>${i}</li>`).join('') + '</ol>';
  });

  // Paragraphs: wrap loose lines not already HTML blocks
  const lines = md.split(/\n{2,}/);
  md = lines.map(block => {
    const trimmed = block.trim();
    if (!trimmed) return '';
    // If starts with a block tag, keep as-is
    if (/^<(h\d|ul|ol|li|pre|blockquote|p|table|code)/i.test(trimmed)) return trimmed;
    return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
  }).join('\n');

  return sanitizeHtml(md);
}

function sanitizeHtml(html) {
  const allowed = new Set(['A','P','BR','STRONG','EM','CODE','PRE','H1','H2','H3','UL','OL','LI']);
  const temp = document.createElement('div');
  temp.innerHTML = html;
  const walk = (node) => {
    const children = Array.from(node.childNodes);
    for (const child of children) {
      if (child.nodeType === 1) { // element
        if (!allowed.has(child.tagName)) {
          // Replace disallowed element with its textContent
          const text = document.createTextNode(child.textContent || '');
          node.replaceChild(text, child);
          continue;
        }
        // Clean attributes
        for (const attr of Array.from(child.attributes)) {
          const name = attr.name.toLowerCase();
          if (child.tagName === 'A' && (name === 'href' || name === 'target' || name === 'rel')) continue;
          if (child.tagName === 'CODE' && name === 'class') continue;
          child.removeAttribute(attr.name);
        }
        walk(child);
      } else if (child.nodeType === 8) { // comment
        node.removeChild(child);
      }
    }
  };
  walk(temp);
  return temp.innerHTML;
}

function renderSources(contexts = []) {
  if (!sourcesListEl) return;
  sourcesListEl.innerHTML = '';
  if (!contexts || !contexts.length) {
    const empty = document.createElement('div');
    empty.className = 'source-card';
    empty.textContent = 'No sources for this answer.';
    sourcesListEl.appendChild(empty);
    return;
  }
  contexts.forEach((hit, i) => {
    const card = document.createElement('div');
    card.className = 'source-card';
    const head = document.createElement('div');
    head.className = 'source-head';
    const id = hit.id ?? '';
    const dist = (hit.distance ?? '').toString().slice(0, 6);
    head.textContent = `Context ${i+1} • id=${id} • d=${dist}`;
    const body = document.createElement('div');
    body.className = 'source-body';
    body.textContent = hit.document || '';
    card.appendChild(head);
    card.appendChild(body);
    sourcesListEl.appendChild(card);
  });
}

function renderMessage(msg) {
  const row = document.createElement('div');
  row.className = `message ${msg.role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = msg.role === 'user' ? 'U' : 'A';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  // Content container for typewriter effect
  const content = document.createElement('div');
  content.className = 'content';
  // Render markdown to sanitized HTML for assistant; plain text for user
  if (msg.role === 'assistant') {
    content.innerHTML = mdToHtml(msg.content || '');
  } else {
    content.innerHTML = escapeHtml(msg.content || '').replace(/\n/g, '<br>');
  }
  bubble.appendChild(content);

  // Tools: copy
  const tools = document.createElement('div');
  tools.className = 'tools';
  const copyBtn = document.createElement('button');
  copyBtn.className = 'btn';
  copyBtn.textContent = 'Copy';
  copyBtn.addEventListener('click', () => navigator.clipboard.writeText(msg.content || ''));
  tools.appendChild(copyBtn);
  // Per-message Sources toggle (assistant only)
  if (msg.role === 'assistant') {
    const srcBtn = document.createElement('button');
    srcBtn.className = 'btn';
    srcBtn.textContent = 'Sources';
    srcBtn.addEventListener('click', () => {
      if (msg.contexts) renderSources(msg.contexts);
      if (sourcesPanelEl && sourcesToggleEl) {
        sourcesToggleEl.checked = true;
        sourcesPanelEl.style.display = '';
      }
    });
    tools.appendChild(srcBtn);
  }
  bubble.appendChild(tools);

  // Contexts are displayed in the right sidebar, not within the message bubble

  if (msg.role === 'user') {
    // Right side: show bubble before avatar
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    // Left side: show avatar then bubble
    row.appendChild(avatar);
    row.appendChild(bubble);
  }
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return content; // return content el for typing animation
}

function addTyping() {
  const row = document.createElement('div');
  row.className = 'message assistant';
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = 'A';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const dotWrap = document.createElement('div');
  dotWrap.className = 'typing';
  dotWrap.innerHTML = '<span></span><span></span><span></span>';
  bubble.appendChild(dotWrap);
  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

function removeEl(el) { if (el && el.parentNode) el.parentNode.removeChild(el); }

async function sendPrompt(q) {
  const res = await fetch('/api/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: q,
      top_k: Number(topkEl.value || 5),
      llm: llmEl ? (llmEl.value || '').toLowerCase() : undefined,
    })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.error || 'Request failed');
  return data;
}

async function handleSend(ev) {
  ev && ev.preventDefault();
  const q = (promptEl.value || '').trim();
  if (!q) return;
  sendBtn.disabled = true;

  // Render user
  state.messages.push({ role: 'user', content: q });
  renderMessage({ role: 'user', content: q });
  promptEl.value = '';
  autoResize(promptEl);

  // Typing indicator
  const typingRow = addTyping();
  try {
    const data = await sendPrompt(q);
    removeEl(typingRow);
    const answer = data.answer || '(no answer)';
    const hasMd = /```|\*\*|\[[^\]]+\]\([^\)]+\)|^#|<\w+/.test(answer);
    const contentEl = renderMessage({ role: 'assistant', content: hasMd ? answer : '', contexts: data.contexts || [] });
    if (hasMd) {
      contentEl.innerHTML = mdToHtml(answer);
    } else {
      await typewriter(contentEl, answer);
    }
    state.messages.push({ role: 'assistant', content: answer , contexts: data.contexts || []});
    // Show latest sources in the sidebar
    renderSources(data.contexts || []);
    if (sourcesPanelEl && sourcesToggleEl && sourcesToggleEl.checked) {
      sourcesPanelEl.style.display = '';
    }
  } catch (err) {
    removeEl(typingRow);
    renderMessage({ role: 'assistant', content: 'Error: ' + (err.message || err) });
  } finally {
    sendBtn.disabled = false;
  }
}

function autoResize(ta) {
  ta.style.height = 'auto';
  ta.style.height = Math.min(180, Math.max(40, ta.scrollHeight)) + 'px';
}
promptEl.addEventListener('input', () => autoResize(promptEl));
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend(e);
  }
});
form.addEventListener('submit', handleSend);

clearBtn.addEventListener('click', () => {
  state.messages = [];
  messagesEl.innerHTML = '';
  renderMessage({ role: 'assistant', content: 'Hi! Ask me anything.' });
});

// Global Sources toggle shows/hides the sidebar
if (sourcesToggleEl && sourcesPanelEl) {
  sourcesPanelEl.style.display = sourcesToggleEl.checked ? '' : 'none';
  sourcesToggleEl.addEventListener('change', () => {
    sourcesPanelEl.style.display = sourcesToggleEl.checked ? '' : 'none';
  });
}

async function typewriter(container, text) {
  const delay = (ms) => new Promise(r => setTimeout(r, ms));
  const escaped = escapeHtml(text).replace(/\n/g, '<br>');
  container.innerHTML = '';
  let i = 0;
  while (i < escaped.length) {
    container.innerHTML += escaped.slice(i, i + 3); // chunk few chars
    i += 3;
    messagesEl.scrollTop = messagesEl.scrollHeight;
    await delay(12);
  }
}

// Initial bot message
renderMessage({ role: 'assistant', content: 'Hello! I can answer questions using your indexed data. Type a question below.' });
