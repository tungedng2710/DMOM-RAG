const shellEl = document.getElementById('app-shell');
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const formEl = document.getElementById('composer');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const topkEl = document.getElementById('topk');
const clearBtn = document.getElementById('clear');
const sourcesToggleEl = document.getElementById('toggle-sources');
const sourcesListEl = document.getElementById('sources-list');
const sourceCountEl = document.getElementById('source-count');
const mobileSourcesBtn = document.getElementById('mobile-sources');
const closeSourcesBtn = document.getElementById('close-sources');
const panelOverlayEl = document.getElementById('panel-overlay');

const state = { messages: [], controller: null };
let clarificationEl = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);
}

function isSafeUrl(url) {
  try {
    const parsed = new URL(url, window.location.origin);
    return ['http:', 'https:', 'mailto:'].includes(parsed.protocol);
  } catch (_) {
    return false;
  }
}

function sanitizeHtml(html) {
  const allowed = new Set(['A', 'P', 'BR', 'STRONG', 'EM', 'CODE', 'PRE', 'H1', 'H2', 'H3', 'UL', 'OL', 'LI', 'SPAN']);
  const container = document.createElement('div');
  container.innerHTML = html;

  function walk(node) {
    Array.from(node.childNodes).forEach((child) => {
      if (child.nodeType === Node.COMMENT_NODE) {
        child.remove();
        return;
      }
      if (child.nodeType !== Node.ELEMENT_NODE) return;
      if (!allowed.has(child.tagName)) {
        child.replaceWith(document.createTextNode(child.textContent || ''));
        return;
      }
      Array.from(child.attributes).forEach((attribute) => {
        const keepLinkAttribute = child.tagName === 'A' && ['href', 'target', 'rel'].includes(attribute.name);
        const keepCodeClass = child.tagName === 'CODE' && attribute.name === 'class';
        const keepCitationClass = child.tagName === 'SPAN' && attribute.name === 'class' && attribute.value === 'citation';
        if (!keepLinkAttribute && !keepCodeClass && !keepCitationClass) child.removeAttribute(attribute.name);
      });
      if (child.tagName === 'A' && !isSafeUrl(child.getAttribute('href') || '')) {
        child.removeAttribute('href');
        child.removeAttribute('target');
        child.removeAttribute('rel');
      }
      walk(child);
    });
  }

  walk(container);
  return container.innerHTML;
}

function mdToHtml(markdown) {
  if (!markdown) return '';
  const codeBlocks = [];
  let text = String(markdown).replace(/\r\n?/g, '\n');
  text = text.replace(/```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g, (_, language, code) => {
    const token = `@@DMOM_CODE_${codeBlocks.length}@@`;
    const className = language ? ` class="lang-${escapeHtml(language)}"` : '';
    codeBlocks.push(`<pre><code${className}>${escapeHtml(code)}</code></pre>`);
    return token;
  });

  text = escapeHtml(text)
    .replace(/^###\s+(.+)$/gm, '<h3>$1</h3>')
    .replace(/^##\s+(.+)$/gm, '<h2>$1</h2>')
    .replace(/^#\s+(.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+|mailto:[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\[(\d+)\]/g, '<span class="citation">[$1]</span>');

  text = text.replace(/(?:^|\n)((?:[-*]\s+.*(?:\n|$))+)/g, (_, group) => {
    const items = group.trim().split('\n').map((line) => line.replace(/^[-*]\s+/, ''));
    return `\n<ul>${items.map((item) => `<li>${item}</li>`).join('')}</ul>\n`;
  });
  text = text.replace(/(?:^|\n)((?:\d+\.\s+.*(?:\n|$))+)/g, (_, group) => {
    const items = group.trim().split('\n').map((line) => line.replace(/^\d+\.\s+/, ''));
    return `\n<ol>${items.map((item) => `<li>${item}</li>`).join('')}</ol>\n`;
  });

  text = text.split(/\n{2,}/).map((block) => {
    const trimmed = block.trim();
    if (!trimmed) return '';
    if (/^(<h\d|<ul|<ol|<pre|@@DMOM_CODE_)/i.test(trimmed)) return trimmed;
    return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
  }).join('\n');

  codeBlocks.forEach((block, index) => {
    text = text.replace(`@@DMOM_CODE_${index}@@`, block);
  });
  return sanitizeHtml(text);
}

function scrollToLatest() {
  requestAnimationFrame(() => { messagesEl.scrollTop = messagesEl.scrollHeight; });
}

function copyIcon() {
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>';
}

function renderMessage(message) {
  welcomeEl.classList.add('hidden');
  const row = document.createElement('article');
  row.className = `message ${message.role}${message.error ? ' error' : ''}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = message.role === 'user' ? 'BẠN' : 'AI';

  const body = document.createElement('div');
  body.className = 'message-body';
  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.innerHTML = message.role === 'user'
    ? '<span>Bạn</span>'
    : '<span>DMOM Assistant</span><span class="model-mini">GLM 5.3</span>';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const content = document.createElement('div');
  content.className = 'content';
  if (message.role === 'assistant') content.innerHTML = mdToHtml(message.content || '');
  else content.innerHTML = escapeHtml(message.content || '').replace(/\n/g, '<br>');
  bubble.appendChild(content);
  body.append(meta, bubble);

  if (message.role === 'assistant' && !message.error) {
    const copy = document.createElement('button');
    copy.className = 'copy-button';
    copy.type = 'button';
    copy.innerHTML = `${copyIcon()}<span>Sao chép</span>`;
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(message.content || '');
        copy.querySelector('span').textContent = 'Đã sao chép';
        setTimeout(() => { copy.querySelector('span').textContent = 'Sao chép'; }, 1400);
      } catch (_) {
        copy.querySelector('span').textContent = 'Không thể sao chép';
      }
    });
    body.appendChild(copy);
  }

  if (message.role === 'user') row.append(body, avatar);
  else row.append(avatar, body);
  messagesEl.appendChild(row);
  scrollToLatest();
  return row;
}

function addTyping() {
  const row = document.createElement('article');
  row.className = 'message assistant';
  row.innerHTML = `
    <div class="avatar">AI</div>
    <div class="message-body">
      <div class="message-meta"><span>DMOM Assistant</span><span class="model-mini">GLM 5.3</span></div>
      <div class="bubble"><div class="typing" aria-label="Đang tạo câu trả lời"><span></span><span></span><span></span></div></div>
    </div>`;
  messagesEl.appendChild(row);
  scrollToLatest();
  return row;
}

function renderSources(contexts = []) {
  sourcesListEl.innerHTML = '';
  sourceCountEl.textContent = String(contexts.length);
  if (!contexts.length) {
    sourcesListEl.innerHTML = `
      <div class="sources-empty">
        <span><svg viewBox="0 0 32 32" aria-hidden="true"><path d="M6 5h16a3 3 0 0 1 3 3v19H9a3 3 0 0 1-3-3V5Zm0 19a3 3 0 0 1 3-3h16M11 11h9M11 15h7" /></svg></span>
        <strong>Chưa có nguồn được truy xuất</strong>
        <p>Nguồn liên quan sẽ xuất hiện tại đây sau khi bạn đặt câu hỏi.</p>
      </div>`;
    return;
  }

  contexts.forEach((hit, index) => {
    const card = document.createElement('article');
    card.className = 'source-card';
    const distance = Number(hit.distance);
    const score = Number.isFinite(distance) ? distance.toFixed(3) : '—';
    const sourceId = hit.id ? `ID ${hit.id}` : 'Dữ liệu DMOM';
    card.innerHTML = `
      <div class="source-head">
        <span class="source-number">${index + 1}</span>
        <span class="source-title" title="${escapeHtml(sourceId)}">${escapeHtml(sourceId)}</span>
        <span class="source-score">d ${score}</span>
      </div>
      <div class="source-body"></div>
      <button class="source-toggle" type="button">Xem thêm</button>`;
    card.querySelector('.source-body').textContent = hit.document || '';
    const toggle = card.querySelector('.source-toggle');
    toggle.addEventListener('click', () => {
      const expanded = card.classList.toggle('expanded');
      toggle.textContent = expanded ? 'Thu gọn' : 'Xem thêm';
    });
    sourcesListEl.appendChild(card);
  });
}

async function sendPrompt(question, choice, signal) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({ message: question, top_k: Number(topkEl.value || 5), ...(choice !== undefined ? { clarification_choice: choice } : {}) })
  });
  let data = {};
  try { data = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) throw new Error(data.detail || data.error || 'Không thể kết nối đến hệ thống.');
  return data;
}

function removeClarification() {
  if (clarificationEl) clarificationEl.remove();
  clarificationEl = null;
}

function renderClarification(menu, question) {
  removeClarification();
  const panel = document.createElement('section');
  panel.className = 'clarification-card';
  panel.setAttribute('aria-labelledby', 'clarification-title');
  const eyebrow = document.createElement('p');
  eyebrow.className = 'overline';
  eyebrow.textContent = 'CÙNG LÀM RÕ';
  const title = document.createElement('h2');
  title.id = 'clarification-title';
  title.tabIndex = -1;
  title.textContent = menu.title;
  const description = document.createElement('p');
  description.className = 'clarification-description';
  description.textContent = menu.description;
  const options = document.createElement('div');
  options.className = 'clarification-options';
  menu.options.forEach((option) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = option.label;
    button.addEventListener('click', () => handleSend(null, question, option.id, option.label));
    options.appendChild(button);
  });
  const custom = document.createElement('button');
  custom.type = 'button';
  custom.className = 'clarification-custom';
  custom.textContent = 'Tự mô tả thêm →';
  custom.addEventListener('click', () => {
    promptEl.value = `${question} — `;
    autoResize();
    promptEl.focus();
    scrollToLatest();
  });
  panel.append(eyebrow, title, description, options, custom);
  messagesEl.appendChild(panel);
  clarificationEl = panel;
  renderSources([]);
  scrollToLatest();
  requestAnimationFrame(() => title.focus({ preventScroll: true }));
}

async function handleSend(event, suggestedPrompt, choice, displayText) {
  if (event) event.preventDefault();
  const question = String(suggestedPrompt || promptEl.value || '').trim();
  if (!question || sendBtn.disabled) return;

  removeClarification();
  sendBtn.disabled = true;
  const controller = new AbortController();
  state.controller = controller;
  state.messages.push({ role: 'user', content: displayText || question });
  renderMessage({ role: 'user', content: displayText || question });
  promptEl.value = '';
  autoResize();
  const typingRow = addTyping();

  try {
    const data = await sendPrompt(question, choice, controller.signal);
    if (controller.signal.aborted) return;
    typingRow.remove();
    if (data.type === 'clarification') {
      renderClarification(data.clarification, question);
      return;
    }
    const answer = data.answer || 'Tôi chưa tìm thấy câu trả lời phù hợp trong kho tri thức.';
    state.messages.push({ role: 'assistant', content: answer, contexts: data.contexts || [] });
    renderMessage({ role: 'assistant', content: answer });
    renderSources(data.contexts || []);
  } catch (error) {
    typingRow.remove();
    if (controller.signal.aborted) return;
    renderMessage({
      role: 'assistant',
      error: true,
      content: `Đã xảy ra lỗi khi xử lý câu hỏi. ${error.message || 'Vui lòng thử lại sau.'}`
    });
  } finally {
    if (state.controller === controller) {
      state.controller = null;
      sendBtn.disabled = false;
      if (!clarificationEl) promptEl.focus();
    }
  }
}

function autoResize() {
  promptEl.style.height = 'auto';
  promptEl.style.height = `${Math.min(150, Math.max(38, promptEl.scrollHeight))}px`;
}

function openSources() {
  shellEl.classList.add('panel-open');
  mobileSourcesBtn.setAttribute('aria-expanded', 'true');
}

function closeSources() {
  shellEl.classList.remove('panel-open');
  mobileSourcesBtn.setAttribute('aria-expanded', 'false');
}

formEl.addEventListener('submit', handleSend);
promptEl.addEventListener('input', autoResize);
promptEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    handleSend(event);
  }
});

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => handleSend(null, button.dataset.prompt));
});

function clearConversation() {
  if (state.controller) state.controller.abort();
  state.controller = null;
  sendBtn.disabled = false;
  removeClarification();
  state.messages = [];
  messagesEl.querySelectorAll('.message').forEach((message) => message.remove());
  welcomeEl.classList.remove('hidden');
  renderSources([]);
  promptEl.value = '';
  autoResize();
  promptEl.focus();
}
clearBtn.addEventListener('click', clearConversation);
document.getElementById('mobile-clear').addEventListener('click', clearConversation);

sourcesToggleEl.addEventListener('change', () => {
  shellEl.classList.toggle('sources-hidden', !sourcesToggleEl.checked);
});
mobileSourcesBtn.addEventListener('click', openSources);
closeSourcesBtn.addEventListener('click', closeSources);
panelOverlayEl.addEventListener('click', closeSources);
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeSources(); });

renderSources([]);
