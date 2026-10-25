'use strict';
const $=s=>document.querySelector(s);
const terminal=new Set(['completed','failed','cancelled','interrupted']);
const labels={count_places:'Считаю места',find_places:'Ищу места',nearest_place:'Сравниваю расстояния',place_details:'Открываю карточку',top_brands:'Сравниваю сети',semantic_search:'Ищу по смыслу'};
const errors={invalid_credentials:'Неверный логин или пароль.',username_taken:'Этот логин уже занят.',invalid_request:'Проверьте поля формы. Пароль — от 12 символов, логин — от 3 латинских букв, цифр или _.',too_many_attempts:'Слишком много попыток. Попробуйте через минуту.',csrf_failed:'Сессия изменилась. Обновите страницу.',server_capacity:'Все места в очереди заняты. Попробуйте чуть позже.',user_already_running:'Дождитесь завершения текущего запроса.',provider_error:'Провайдер модели не смог ответить. Попробуйте позже.',provider_rate_limit:'Провайдер ограничил запросы. Попробуйте позже.',provider_or_runtime_error:'Не удалось получить ответ модели. Попробуйте позже.',model_not_configured:'Модель ещё не настроена.',run_timeout:'Время ожидания истекло. Попробуйте уточнить вопрос.',repeated_invalid_arguments:'Модель повторила неверный вызов инструмента. Уточните вопрос.',model_call_limit:'Достигнут лимит обращений к модели.',invalid_answer:'Модель не вернула проверяемый ответ.',server_restarted:'Сервер был перезапущен. Отправьте вопрос ещё раз.',user_cancelled:'Запрос остановлен.',server_stopping:'Сервер остановлен.',context_limit:'Текущий запрос превысил лимит контекста.',internal_error:'Не удалось выполнить запрос. Попробуйте позже.'};
function csrf(){return document.cookie.split('; ').find(x=>x.startsWith('mvp_csrf='))?.split('=')[1]||'';}
async function api(path,options={}){
  const response=await fetch(path,{credentials:'same-origin',...options,headers:{'Content-Type':'application/json','X-CSRF-Token':csrf(),...options.headers}});
  const data=await response.json().catch(()=>({}));
  if(!response.ok){if(response.status===401&&!path.includes('/api/auth/'))location.assign('/');throw new Error(errors[data.detail]||'Не удалось выполнить действие. Попробуйте ещё раз.');}return data;
}
function node(tag,text,className){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
if(document.querySelector('[data-page="auth"]')){
  let registering=false;
  $('#auth-toggle').addEventListener('click',()=>{registering=!registering;$('#auth-title').textContent=registering?'Начнём знакомство':'С возвращением';$('#auth-subtitle').textContent=registering?'Создайте аккаунт и сохраните свои диалоги.':'Войдите, чтобы продолжить исследовать город.';$('#auth-submit').textContent=registering?'Создать аккаунт ↗':'Войти ↗';$('#switch-text').textContent=registering?'Уже есть аккаунт?':'Первый раз здесь?';$('#auth-toggle').textContent=registering?'Войти':'Создать аккаунт';$('#password').autocomplete=registering?'new-password':'current-password';$('#auth-error').textContent='';});
  $('#auth-form').addEventListener('submit',async event=>{event.preventDefault();$('#auth-error').textContent='';$('#auth-submit').disabled=true;try{await api('/api/auth/'+(registering?'register':'login'),{method:'POST',body:JSON.stringify({username:$('#username').value,password:$('#password').value})});$('#password').value='';location.assign('/chat');}catch(error){$('#auth-error').textContent=error.message;}finally{$('#auth-submit').disabled=false;}});
}
if(document.querySelector('[data-page="chat"]')){
  let chatId=null,activeRun=null,stream=null,pending=null;
  const seen=new Set();
  function scroll(){const el=$('#conversation');el.scrollTop=el.scrollHeight;}
  function closeStream(){if(stream)stream.close();stream=null;}
  function busy(value){$('#send-question').disabled=value;$('#question').disabled=value;$('#cancel-run').hidden=!value;}
  function message(prompt){$('#welcome')?.remove();const row=node('article',undefined,'message user-message');row.append(node('div','ВЫ','message-label'),node('p',prompt));$('#conversation').append(row);scroll();}
  function showRun(run){
    if(seen.has(run.id))return;seen.add(run.id);
    const row=node('article',undefined,'message assistant-message');row.dataset.runId=run.id;row.append(node('div','РЯДОМ','message-label'));
    if(run.answer){const a=run.answer;
      if(a.type==='count')row.append(node('p',`Найдено мест: ${a.count.toLocaleString('ru-RU')}`,'count-answer'));
      else if(a.type==='brands')row.append(node('p',a.brands.join(' · ')));
      else if(a.type==='unavailable')row.append(node('p',a.reason==='not_found'?'Подходящих мест не нашлось. Попробуйте другое описание.':a.reason==='unsupported'?'Этот запрос пока не поддерживается. Спросите о местах в городе.':'Нужны дополнительные данные. Укажите вид места, улицу или координаты.'));
      for(const place of a.places||[]){const card=node('div',undefined,'place-card');card.append(node('h3',place.name||place.brand||'Без названия'));const address=[place.street,place.housenumber].filter(Boolean).join(', ');card.append(node('p',address||'Адрес не указан в OSM'));const info=[place.cuisine&&`Кухня: ${place.cuisine}`,place.opening_hours&&`Часы: ${place.opening_hours}`,place.distance_m!==undefined&&`${Math.round(place.distance_m)} м по прямой`].filter(Boolean);for(const text of info)card.append(node('small',text));if(place.similarity!==undefined)card.append(node('small',`Смысловое сходство: ${Math.round(place.similarity*100)}%`));row.append(card);}
      row.append(node('small',a.note,'answer-note'));
    }else row.append(node('p',errors[run.error_code]||'Запрос не завершён. Отправьте его ещё раз.','error'));
    const usage=(run.input_tokens===null||run.output_tokens===null)?'Токены: провайдер не передал полные данные':`Токены: ${run.input_tokens} вход / ${run.output_tokens} выход`;
    row.append(node('small',`${usage} · обращений к модели: ${run.model_calls}`,'usage'));$('#conversation').append(row);scroll();
  }
  async function list(){const rows=await api('/api/chats');$('#chat-list').replaceChildren();for(const chat of rows){const button=node('button',chat.title,chat.id===chatId?'selected':'');button.addEventListener('click',()=>openChat(chat.id));$('#chat-list').append(button);}}
  async function finish(id){const run=await api('/api/runs/'+id);if(!terminal.has(run.status))return;closeStream();showRun(run);activeRun=null;busy(false);$('#run-status').textContent='';await list();}
  function listen(id){closeStream();activeRun=id;busy(true);stream=new EventSource('/api/runs/'+id+'/events');
    for(const kind of ['queued','started','agent_step','tool_started','tool_finished','tool_rejected','answer_retry','usage','completed','failed','cancelled','interrupted','auth_expired','stream_error'])stream.addEventListener(kind,event=>{const data=JSON.parse(event.data);if(terminal.has(kind)){finish(id).catch(error=>$('#chat-error').textContent=error.message);return;}if(kind==='auth_expired'){closeStream();location.assign('/');return;}if(kind==='queued')$('#run-status').textContent='Запрос в очереди…';if(kind==='started')$('#run-status').textContent='Начинаю поиск…';if(kind==='agent_step')$('#run-status').textContent=`Агент: шаг ${data.step}`;if(kind==='tool_started')$('#run-status').textContent=(labels[data.tool]||'Работаю с инструментом')+'…';if(kind==='tool_finished')$('#run-status').textContent=data.ok?'Данные получены. Проверяю ответ…':'Источник не ответил. Проверяю возможности…';if(kind==='answer_retry')$('#run-status').textContent='Проверяю формат и факты ответа…';});
    stream.onerror=()=>{$('#run-status').textContent='Восстанавливаю соединение…';finish(id).catch(()=>{});};
  }
  async function openChat(id){closeStream();chatId=id;localStorage.setItem('ryadom_chat',id);seen.clear();$('#conversation').replaceChildren();$('#chat-error').textContent='';const data=await api('/api/chats/'+id);activeRun=null;for(const run of data.runs){message(run.prompt);if(terminal.has(run.status))showRun(run);else{activeRun=run.id;listen(run.id);}}busy(Boolean(activeRun));await list();}
  async function newChat(){if(activeRun){$('#chat-error').textContent='Сначала завершите или остановите текущий запрос.';return;}const chat=await api('/api/chats',{method:'POST'});await openChat(chat.id);$('#question').focus();}
  $('#new-chat').addEventListener('click',()=>newChat().catch(error=>$('#chat-error').textContent=error.message));
  $('#question-form').addEventListener('submit',async event=>{event.preventDefault();if(activeRun)return;const text=$('#question').value.trim();if(!text)return;$('#chat-error').textContent='';busy(true);try{if(!chatId){chatId=(await api('/api/chats',{method:'POST'})).id;localStorage.setItem('ryadom_chat',chatId);}if(!pending||pending.text!==text||pending.chat!==chatId)pending={text,chat:chatId,key:crypto.randomUUID()};const run=await api('/api/chats/'+chatId+'/messages',{method:'POST',body:JSON.stringify({text,request_key:pending.key})});pending=null;message(text);$('#question').value='';listen(run.id);}catch(error){$('#chat-error').textContent=error.message;busy(false);}});
  $('#question').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();$('#question-form').requestSubmit();}});
  $('#cancel-run').addEventListener('click',async()=>{try{if(activeRun)await api('/api/runs/'+activeRun+'/cancel',{method:'POST'});}catch(error){$('#chat-error').textContent=error.message;}});
  for(const button of document.querySelectorAll('[data-question]'))button.addEventListener('click',()=>{$('#question').value=button.dataset.question;$('#question').focus();});
  $('#logout').addEventListener('click',async()=>{try{await api('/api/auth/logout',{method:'POST'});closeStream();localStorage.removeItem('ryadom_chat');location.assign('/');}catch(error){$('#chat-error').textContent=error.message;}});
  (async()=>{const me=await api('/api/auth/me');$('#account-name').textContent=me.username;const chats=await api('/api/chats');const saved=localStorage.getItem('ryadom_chat');if(chats.some(c=>c.id===saved))await openChat(saved);else await list();async function status(){try{const data=await api('/api/status');$('#service-status').textContent=!data.model_configured?'Модель не настроена. Проверьте локальную конфигурацию.':data.search==='preparing'?'Готовлю локальный смысловой индекс. Обычный поиск уже доступен.':data.search==='failed'?'Смысловой индекс недоступен. Проверьте загрузку локальной модели.':'';}catch{}}await status();setInterval(status,15000);})().catch(error=>$('#chat-error').textContent=error.message);
  window.addEventListener('pagehide',closeStream);
}
