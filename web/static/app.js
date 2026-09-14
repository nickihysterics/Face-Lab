'use strict';
const $ = (id) => document.getElementById(id);
const form = $('predictForm');
let selectedFile = null;
let resultData = null;
let controller = null;
let objectUrl = null;
let generation = 0;

function setError(message = '') {
  $('errorBox').textContent = message;
  $('errorBox').hidden = !message;
}
function busy(value) {
  $('demoBtn').disabled = value;
  $('runBtn').disabled = value;
  $('canvas').setAttribute('aria-busy', String(value));
}
function clearResult() {
  resultData = null;
  $('resultImage').hidden = true;
  $('resultImage').removeAttribute('src');
  $('placeholder').hidden = false;
  $('results').replaceChildren();
  $('demoNotice').hidden = true;
  $('downloadBtn').hidden = true;
  $('resultMode').textContent = 'ОЖИДАНИЕ';
  $('faceCount').textContent = '— ОБЪЕКТОВ';
}
function selectFile(file) {
  generation += 1;
  if (controller) controller.abort();
  busy(false);
  clearResult();
  setError();
  selectedFile = file || null;
  $('fileName').textContent = file ? file.name : 'Файл ещё не выбран';
  $('statusText').textContent = file ? 'Изображение выбрано' : 'Рабочая область готова';
  if (file && file.size > 8 * 1024 * 1024) {
    selectedFile = null;
    setError('Выберите изображение не больше 8 МБ.');
  }
}
function showResult(data) {
  resultData = data;
  const isDemo = data.mode === 'demo';
  $('resultMode').textContent = isDemo ? 'ДЕМОНСТРАЦИЯ' : 'ИНФЕРЕНС';
  $('demoNotice').hidden = !isDemo;
  $('placeholder').hidden = true;
  $('resultImage').src = `data:image/${data.image_format};base64,${data.image_base64}`;
  $('resultImage').hidden = false;
  $('statusText').textContent = isDemo ? 'Синтетический пример · заданная разметка' : data.detections.length ? 'Обработка завершена' : 'Лица не обнаружены. Попробуйте другой снимок или порог.';
  $('faceCount').textContent = `${data.detections.length} ОБЪЕКТОВ`;
  $('results').replaceChildren();
  data.detections.forEach((detection, index) => {
    const card = document.createElement('article');
    card.className = `result-card${detection.label === 'unknown' ? ' unknown' : ''}`;
    const top = document.createElement('div');
    top.className = 'card-top';
    const number = document.createElement('span');
    number.textContent = `ОБЪЕКТ ${String(index + 1).padStart(2, '0')}`;
    const score = document.createElement('span');
    score.textContent = Number(detection.conf).toFixed(2);
    top.append(number, score);
    const title = document.createElement('h3');
    title.textContent = detection.label;
    const meter = document.createElement('div');
    meter.className = 'score-line';
    const fill = document.createElement('span');
    fill.style.width = `${Math.max(0, Math.min(1, Number(detection.conf))) * 100}%`;
    meter.append(fill);
    const details = document.createElement('p');
    details.textContent = `Рамка: ${detection.x1}, ${detection.y1} — ${detection.x2}, ${detection.y2}`;
    card.append(top, title, meter, details);
    $('results').append(card);
  });
  $('downloadBtn').hidden = false;
}
async function request(url, options = {}) {
  const current = ++generation;
  if (controller) controller.abort();
  const requestController = new AbortController();
  controller = requestController;
  clearResult();
  setError();
  busy(true);
  $('statusText').textContent = 'Выполняем запрос…';
  const timer = setTimeout(() => requestController.abort(), 120000);
  try {
    const response = await fetch(url, {...options, signal: requestController.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : `Ошибка запроса: ${response.status}`);
    if (current === generation) showResult(data);
  } catch (error) {
    if (current !== generation) return;
    setError(error.name === 'AbortError' ? 'Запрос отменён или превысил 120 секунд. Сервер может ещё завершать инференс.' : error.message || 'Нет связи с сервером.');
    $('statusText').textContent = 'Запрос не завершён';
  } finally {
    clearTimeout(timer);
    if (current === generation) busy(false);
  }
}
$('demoBtn').addEventListener('click', () => request('/api/demo/'));
$('imageFile').addEventListener('change', (event) => selectFile(event.target.files[0]));
form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!selectedFile) return setError('Сначала выберите изображение.');
  const body = new FormData();
  body.append('file', selectedFile);
  body.append('conf', $('conf').value);
  body.append('unknown_threshold', $('unknown').value);
  request('/api/predict/', {method: 'POST', body});
});
for (const [input, output] of [['conf', 'confValue'], ['unknown', 'unknownValue']]) {
  $(input).addEventListener('input', () => $(output).textContent = Number($(input).value).toFixed(2));
}
for (const event of ['dragenter', 'dragover']) $('dropZone').addEventListener(event, (e) => { e.preventDefault(); $('dropZone').classList.add('dragging'); });
for (const event of ['dragleave', 'drop']) $('dropZone').addEventListener(event, (e) => { e.preventDefault(); $('dropZone').classList.remove('dragging'); });
$('dropZone').addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));
$('downloadBtn').addEventListener('click', () => {
  if (!resultData) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  const {image_base64, image_format, ...json} = resultData;
  objectUrl = URL.createObjectURL(new Blob([JSON.stringify(json, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = `face-lab-${resultData.mode}.json`;
  link.click();
});
fetch('/api/status/').then(response => { if (!response.ok) throw new Error(); return response.json(); }).then(status => {
  $('modelStatus').textContent = status.inference_configured ? '● Модели подключены' : '● Демо доступно · модели не настроены';
  if (!status.inference_configured) $('requirements').textContent = 'Для обработки своего снимка подключите модели и ML-зависимости по README. Пример доступен сразу.';
}).catch(() => { $('modelStatus').textContent = 'Не удалось проверить состояние сервера'; });
