async def test_auth_chat_redirect_assets_and_csp(client):
    response=await client.get('/')
    assert response.status_code==200 and 'Место найдётся' in response.text
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    assert (await client.get('/chat')).status_code==303
    assert (await client.get('/static/app.js')).status_code==200
    assert (await client.get('/static/styles.css')).status_code==200
    assert (await client.get('/health')).json()=={'status':'ok'}


def test_browser_registration_question_history_mobile(settings,tmp_path):
    import socket,time,threading
    from pathlib import Path
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    from app.main import create_app
    from conftest import FakeRuntime,FakeEmbedder
    runtime=FakeRuntime()
    app=create_app(settings,runtime_factory=lambda *args:runtime,embedder_factory=FakeEmbedder,autobuild=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,access_log=False,log_level='critical'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    for _ in range(200):
        if server.started:break
        time.sleep(.05)
    assert server.started
    images=Path(__file__).resolve().parents[2]/'private/mvp/screenshots';images.mkdir(parents=True,exist_ok=True)
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,downloads_path=str(tmp_path/'downloads'))
            page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[];page.on('pageerror',lambda error:errors.append(type(error).__name__))
            page.goto(f'http://127.0.0.1:{port}/')
            page.screenshot(path=str(images/'auth-desktop.png'),full_page=True)
            page.locator('#auth-toggle').click();page.locator('#username').fill('browser_tester');page.locator('#password').fill('test-only-password-2026');page.locator('#auth-submit').click()
            expect(page).to_have_url(f'http://127.0.0.1:{port}/chat')
            page.screenshot(path=str(images/'chat-desktop.png'),full_page=True)
            page.locator('#question').fill('Сколько всего банков в городе?');page.locator('#send-question').click()
            expect(page.locator('.count-answer')).to_have_text('Найдено мест: 42',timeout=15000)
            page.reload();expect(page.locator('.count-answer')).to_have_text('Найдено мест: 42')
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(images/'chat-mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.locator('#logout').click();expect(page).to_have_url(f'http://127.0.0.1:{port}/')
            assert runtime.calls==1 and not errors
            browser.close()
    finally:
        server.should_exit=True;thread.join(timeout=10)
