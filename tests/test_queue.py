import base64
import http.client
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from unittest.mock import patch

import agy_facade as facade
from test_facade import AUTH, PNG_HEADER, SYNTHETIC_KEY


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(facade.CFG, dict(
            facade.DEFAULT_CONFIG, facade_key=SYNTHETIC_KEY, min_turn_gap_sec=0,
            turn_gap_jitter_sec=0, queue_timeout_sec=4), clear=True))
        self.enterContext(patch.object(facade, 'active_keys', return_value={SYNTHETIC_KEY: 'test'}))
        self.enterContext(patch.object(facade.CATALOG, 'get', return_value=[facade.CFG['default_model']]))
        self.enterContext(patch.object(facade, 'PACER', facade.Pacer()))
        self.enterContext(patch.object(facade, 'log'))
        self.acquire = self.enterContext(patch.object(facade.POOL, 'acquire', side_effect=self.make_worker))
        self.enterContext(patch.object(facade.POOL, 'release_dead', side_effect=self.close_worker))
        self.server = self.enterContext(facade.SafeThreadingHTTPServer(('127.0.0.1', 0), facade.Handler))
        self.running = []
        self.open_workers = 0
        self.peak_workers = 0
        self.body = {}
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.aborted = threading.Event()
        thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)
        self.clients = self.enterContext(ThreadPoolExecutor(max_workers=12))
        self.addCleanup(self.release.set)

    def make_worker(self, *args, **kwargs):
        with self.lock:
            self.open_workers += 1
            self.peak_workers = max(self.peak_workers, self.open_workers)
        return self

    def close_worker(self, worker):
        with self.lock:
            self.open_workers -= 1

    def run_turn(self, prompt, media, on_delta, timeout, *, abort_event, **kwargs):
        marker = prompt.split('[user]\n', 1)[1].splitlines()[0]
        with self.lock:
            self.running.append(marker)
            self.body[marker] = media
        if marker in ('hold', 'disconnect', 'fail'):
            if marker == 'disconnect':
                on_delta(marker)
            self.entered.set()
            deadline = time.monotonic() + 5
            while not self.release.wait(0.01):
                if abort_event.is_set():
                    self.aborted.set()
                    raise facade.AgyError('client aborted', 499)
                if time.monotonic() > deadline:
                    raise AssertionError('test worker not released')
        if marker == 'fail':
            raise facade.AgyError('mock backend failure', 502)
        if on_delta:
            on_delta(marker + ':')
            on_delta('done')
        return marker + ':done', {}

    @staticmethod
    def payload(marker, stream=False, blob=None):
        content = marker
        if blob is not None:
            uri = 'data:image/png;base64,' + base64.b64encode(blob).decode()
            content = [{'type': 'text', 'text': marker}, {'type': 'image_url', 'image_url': {'url': uri}}]
        return json.dumps({'messages': [{'role': 'user', 'content': content}], 'stream': stream}).encode()

    def request(self, body, path='/v1/chat/completions'):
        with closing(http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=6)) as conn:
            if body is None:
                conn.request('GET', path, headers=AUTH)
            else:
                conn.request('POST', path, body=body, headers=AUTH)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail(f'timed out waiting: {self.server.turn_queue.status()}')
            time.sleep(0.01)

    def submit(self, marker, stream=False, blob=None):
        return self.clients.submit(self.request, self.payload(marker, stream, blob))

    def check_reply(self, response, marker, stream=False):
        status, headers, body = response
        self.assertEqual(status, 200, body)
        if stream:
            self.assertEqual(headers['Content-Type'], 'text/event-stream')
            frames = [line[6:] for line in body.splitlines() if line.startswith(b'data: ')]
            self.assertEqual(frames[-1], b'[DONE]')
            chunks = [json.loads(frame) for frame in frames[:-1]]
            self.assertEqual(len({chunk['id'] for chunk in chunks}), 1)
            self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')
            text = ''.join(chunk['choices'][0]['delta'].get('content', '') for chunk in chunks)
            self.assertEqual(text, marker + ':done')
            return chunks[0]['id']
        reply = json.loads(body)
        self.assertEqual(reply['choices'][0]['message']['content'], marker + ':done')
        return reply['id']

    def test_fifo_mixed_sse_json_and_eight_media_requests(self):
        hold = self.submit('hold')
        self.assertTrue(self.entered.wait(2))
        pending = []
        for i in range(8):
            marker, stream, blob = f'item-{i}', i % 2 == 0, PNG_HEADER + bytes([i]) * 4096
            pending.append((self.submit(marker, stream, blob), marker, stream, blob))
            self.wait_for(lambda: self.server.turn_queue.status()['waiting'] == i + 1)
        self.assertEqual(self.acquire.call_count, 1, 'waiting requests must not claim workers')
        status, _, body = self.request(None, '/')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['queue']['waiting'], 8)
        self.release.set()
        ids = {self.check_reply(hold.result(3), 'hold')}
        for future, marker, stream, blob in pending:
            ids.add(self.check_reply(future.result(3), marker, stream))
            self.assertEqual(self.body[marker], [('image/png', blob)])
        self.assertEqual(len(ids), 9)
        self.assertEqual(self.running, ['hold'] + [f'item-{i}' for i in range(8)])
        self.assertEqual(self.peak_workers, 1)
        self.wait_for(lambda: not self.server.turn_queue.status()['body_bytes'])

    def test_overflow_429_before_sse_headers(self):
        facade.CFG['max_queued_requests'] = 1
        hold = self.submit('hold')
        self.assertTrue(self.entered.wait(2))
        queued = self.submit('queued')
        self.wait_for(lambda: self.server.turn_queue.status()['waiting'] == 1)
        status, headers, body = self.request(self.payload('overflow', True))
        self.assertEqual(status, 429)
        self.assertIn('Retry-After', headers)
        self.assertEqual(json.loads(body)['error']['type'], 'queue_full')
        self.release.set()
        self.check_reply(hold.result(3), 'hold')
        self.check_reply(queued.result(3), 'queued')
        self.assertNotIn('overflow', self.running)

    def test_wait_timeout_never_runs_expired_request(self):
        facade.CFG['queue_timeout_sec'] = 0.15
        hold = self.submit('hold')
        self.assertTrue(self.entered.wait(2))
        status, headers, body = self.request(self.payload('expired', True))
        self.assertEqual(status, 503)
        self.assertIn('Retry-After', headers)
        self.assertEqual(json.loads(body)['error']['type'], 'queue_timeout')
        self.release.set()
        self.check_reply(hold.result(3), 'hold')
        self.check_reply(self.submit('next').result(3), 'next')
        self.assertEqual(self.running, ['hold', 'next'])

    def test_aggregate_body_budget_and_slow_upload_timeout(self):
        facade.CFG.update(max_inflight_body_bytes=1024, max_request_bytes=1024, body_timeout_sec=0.3)
        with socket.create_connection(('127.0.0.1', self.server.server_port), timeout=3) as conn:
            conn.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n'
                         b'Authorization: Bearer test-only\r\nContent-Length: 1024\r\n\r\n{')
            self.wait_for(lambda: self.server.turn_queue.status()['body_bytes'] == 1024)
            status, headers, body = self.request(self.payload('over-budget'))
            self.assertEqual(status, 429)
            self.assertIn('Retry-After', headers)
            self.assertEqual(json.loads(body)['error']['type'], 'body_budget_full')
            response = http.client.HTTPResponse(conn)
            response.begin()
            self.assertEqual(response.status, 408)
            response.read()
        self.check_reply(self.submit('next').result(3), 'next')
        self.assertEqual(self.running, ['next'])

    def test_disconnected_waiter_is_removed(self):
        hold = self.submit('hold')
        self.assertTrue(self.entered.wait(2))
        body = self.payload('gone', True)
        with socket.create_connection(('127.0.0.1', self.server.server_port), timeout=3) as conn:
            conn.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n'
                         b'Authorization: Bearer test-only\r\nContent-Length: '
                         + str(len(body)).encode() + b'\r\n\r\n' + body)
            self.wait_for(lambda: self.server.turn_queue.status()['waiting'] == 1)
        self.wait_for(lambda: not self.server.turn_queue.status()['waiting'])
        self.release.set()
        self.check_reply(hold.result(3), 'hold')
        self.check_reply(self.submit('next').result(3), 'next')
        self.assertEqual(self.running, ['hold', 'next'])

    def test_active_sse_disconnect_waits_for_worker_cleanup(self):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            conn.request('POST', '/v1/chat/completions', body=self.payload('disconnect', True), headers=AUTH)
            response = conn.getresponse()
            self.assertIn(b'disconnect', response.fp.readline())
            queued = self.submit('next')
            self.wait_for(lambda: self.server.turn_queue.status()['waiting'] == 1)
        finally:
            response.close()
            conn.close()
        self.assertTrue(self.aborted.wait(2))
        self.check_reply(queued.result(3), 'next')
        self.assertEqual(self.peak_workers, 1)

    def test_failure_releases_slot_and_image_generation_shares_fifo(self):
        facade.CFG['enable_image_generation'] = True
        fail = self.submit('fail')
        self.assertTrue(self.entered.wait(2))
        with patch.object(facade, 'generate_image_via_agy', return_value=('unused', 'img_123.png')) as image:
            pending = self.clients.submit(self.request, b'{"prompt":"mock drawing"}', '/v1/images/generations')
            self.wait_for(lambda: self.server.turn_queue.status()['waiting'] == 1)
            image.assert_not_called()
            self.release.set()
            self.assertEqual(fail.result(3)[0], 502)
            self.assertEqual(pending.result(3)[0], 200)
            image.assert_called_once()


class WorkerPipeTest(unittest.TestCase):
    def start_worker(self, script):
        workdir = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(facade, 'WORKDIR', workdir))
        real_popen = subprocess.Popen
        with patch.object(facade.subprocess, 'Popen', side_effect=lambda cmd, **opts:
                          real_popen([sys.executable, '-u', '-c', script], **opts)):
            worker = facade.Worker('mock', None)
        self.addCleanup(worker.close)
        self.assertTrue(worker.wait_ready(2))
        return worker

    def test_cli_pipe_sends_complete_large_request(self):
        worker = self.start_worker('''
import json, sys
print(json.dumps({'event': 'init', 'conversation_id': 'fake-session'}))
text = json.loads(sys.stdin.readline())['message']['content']
print(json.dumps({'event': 'result', 'result': {'status': 'SUCCESS', 'response': str(len(text))}}))
''')
        text, _ = worker.run_turn('a' * 256_000, timeout=2)
        self.assertEqual(text, '256000')

    def test_stalled_cli_input_does_not_hold_slot_forever(self):
        worker = self.start_worker('''
import json, time
print(json.dumps({'event': 'init', 'conversation_id': 'fake-session'}))
time.sleep(10)
''')
        with self.assertRaises(facade.AgyError) as error:
            worker.run_turn('a' * 256_000, timeout=0.15)
        self.assertEqual(error.exception.status, 504)


class PoolTest(unittest.TestCase):
    def test_idle_other_model_does_not_block_switch_with_pool_size_one(self):
        class Worker:
            dead = False
            spawn_error = None

            def __init__(self, model, effort):
                self.model = model
                self.closed = False

            def wait_ready(self, timeout):
                return True

            def close(self):
                self.closed = True

        with patch.object(facade, 'Worker', Worker), patch.dict(facade.CFG, {'pool_max_processes': 1}):
            pool = facade.Pool()
            pool.warm('model-a')
            with pool.cond:
                self.assertTrue(pool.cond.wait_for(lambda: bool(pool.idle.get(('model-a', None))), timeout=2))
                old = pool.idle[('model-a', None)][0]
            current = pool.acquire('model-b', None)
            self.assertTrue(old.closed)
            self.assertEqual(current.model, 'model-b')
            pool.release_dead(current)
            self.assertEqual(pool.total, 0)
