import base64
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import agy_facade as facade


SYNTHETIC_KEY = 'test-only'
AUTH = {'Authorization': 'Bearer ' + SYNTHETIC_KEY}
CHAT = {'messages': [{'role': 'user', 'content': 'ping'}]}
PNG_HEADER = b'\x89PNG\r\n\x1a\n' + b'test-not-an-image'


class MockWorker:
    def run_turn(self, prompt, media, on_delta, timeout, **kwargs):
        if '[user]\nping' not in prompt:
            raise AssertionError('missing user turn')
        if on_delta:
            on_delta('mock ')
            on_delta('reply')
        return 'mock reply', {'input_tokens': 1, 'output_tokens': 2}

    def close(self):
        pass


class FacadeTest(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.image = Path(tmp) / 'img_1234567812345678.png'
        self.image.write_bytes(PNG_HEADER)
        self.enterContext(patch.multiple(facade, IMAGE_DIR=tmp, KEYS_PATH=str(Path(tmp) / 'keys.json'), PACER=facade.Pacer()))
        self.enterContext(patch.dict(facade.CFG, dict(facade.DEFAULT_CONFIG, facade_key=SYNTHETIC_KEY, min_turn_gap_sec=0, turn_gap_jitter_sec=0, max_request_bytes=1024), clear=True))
        self.enterContext(patch.object(facade.CATALOG, 'get', return_value=[facade.CFG['default_model']]))
        self.enterContext(patch.object(facade.POOL, 'acquire', return_value=MockWorker()))
        self.enterContext(patch.object(facade.POOL, 'release_dead'))
        self.enterContext(patch.object(facade, 'log'))
        self.server = self.enterContext(facade.SafeThreadingHTTPServer(('127.0.0.1', 0), facade.Handler))
        thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, data=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            conn.request(method, path, body=None if data is None else json.dumps(data).encode(), headers=AUTH if headers is None else headers)
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    def test_auth_and_catalog(self):
        self.assertEqual(self.request('GET', '/v1/models', headers={})[0], 401)
        status, body = self.request('GET', '/v1/models', headers={'x-api-key': SYNTHETIC_KEY})
        self.assertEqual(status, 200)
        self.assertEqual([m['id'] for m in json.loads(body)['data']], [facade.CFG['default_model']])
        with patch.dict(facade.CFG, {'facade_key': ''}):
            self.assertEqual(self.request('GET', '/v1/models')[0], 401)

    def test_chat_json(self):
        status, body = self.request('POST', '/v1/chat/completions', CHAT)
        self.assertEqual(status, 200)
        message = json.loads(body)
        self.assertEqual(message['choices'][0]['message']['content'], 'mock reply')
        self.assertEqual(message['usage']['total_tokens'], 3)

    def test_chat_sse(self):
        status, body = self.request('POST', '/v1/chat/completions', dict(CHAT, stream=True))
        self.assertEqual(status, 200)
        frames = [line[6:] for line in body.splitlines() if line.startswith(b'data: ')]
        self.assertEqual(frames[-1], b'[DONE]')
        choices = [json.loads(x)['choices'][0] for x in frames[:-1]]
        self.assertEqual(''.join(c['delta'].get('content', '') for c in choices), 'mock reply')
        self.assertEqual(choices[-1]['finish_reason'], 'stop')

    def test_request_validation(self):
        self.assertEqual(self.request('POST', '/v1/chat/completions', ['not an object'])[0], 400)
        self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': ['not a message']})[0], 400)
        too_large = {'messages': [{'role': 'user', 'content': 'a' * 1100}]}
        self.assertEqual(self.request('POST', '/v1/chat/completions', too_large)[0], 413)

    def test_image_routes(self):
        route = '/v1/images/' + self.image.name
        self.assertEqual(self.request('GET', route, headers={})[0], 401)
        self.assertEqual(self.request('GET', route), (200, PNG_HEADER))
        self.assertEqual(self.request('POST', '/v1/images/generations', {'prompt': 'test image'})[0], 404)
        with patch.dict(facade.CFG, {'enable_image_generation': True}), patch.object(facade, 'generate_image_via_agy', return_value=(self.image, self.image.name)):
            status, body = self.request('POST', '/v1/images/generations', {'prompt': 'test image', 'response_format': 'b64_json'})
            self.assertEqual(status, 200)
            self.assertEqual(base64.b64decode(json.loads(body)['data'][0]['b64_json']), PNG_HEADER)

    def test_inline_media_only(self):
        uri = 'data:image/png;base64,' + base64.b64encode(PNG_HEADER).decode()
        req = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'ping'}, {'type': 'image_url', 'image_url': {'url': uri}}]}]}
        prompt, media = facade.build_prompt_and_media(req, req['messages'])
        self.assertIn('ping', prompt)
        self.assertEqual(media, [('image/png', PNG_HEADER)])
        self.assertEqual(facade.parse_media_value(str(self.image)), (None, None))
        self.assertEqual(facade.parse_media_value('https://example.invalid/picture.png'), (None, None))
        self.assertEqual(facade.parse_media_value(self.image.as_uri()), (None, None))


if __name__ == '__main__':
    unittest.main()
