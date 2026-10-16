import io
import json
import unittest
from unittest.mock import MagicMock, patch

from app.modules.guide import chat_model


def sse(delta=None, finish=None):
    return ('data: ' + json.dumps({'choices': [{'delta': delta or {}, 'finish_reason': finish}]},
                                 ensure_ascii=False) + '\n\n').encode('utf-8')


class ChatStreamTest(unittest.TestCase):
    settings = {'base_url': 'https://model.invalid/v1', 'model': 'synthetic', 'api_key': 'synthetic-secret'}

    def opening(self, raw):
        response = io.BytesIO(raw)
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = response
        return opener

    def test_text_arrives_before_completion_and_requests_streaming(self):
        raw = sse({'content': '中文'}) + sse({'content': '链接'}) + sse(finish='stop') + b'data: [DONE]\n\n'
        opener = self.opening(raw)
        with patch.object(chat_model.urllib.request, 'build_opener', return_value=opener):
            stream = chat_model.stream(self.settings, [], [])
            self.assertEqual(next(stream), {'type': 'delta', 'text': '中文'})
            response = opener.open.return_value.__enter__.return_value
            self.assertLess(response.tell(), len(raw))
            self.assertEqual(next(stream)['text'], '链接')
            self.assertEqual(next(stream)['message']['content'], '中文链接')
            request = opener.open.call_args.args[0]
            self.assertTrue(json.loads(request.data)['stream'])
            self.assertEqual(request.get_header('Accept'), 'text/event-stream')
            stream.close()
            opener.open.return_value.__exit__.assert_called_once()

    def test_fragmented_parallel_tool_calls_and_keepalive(self):
        raw = b': ping\n\n' + sse({'tool_calls': [
            {'index': 0, 'id': 'one', 'function': {'name': 'wiki_get', 'arguments': '{"kind":'}},
            {'index': 1, 'id': 'two', 'function': {'name': 'wiki_search', 'arguments': '{"query":'}}]})
        raw += sse({'tool_calls': [
            {'index': 0, 'function': {'arguments': '"characters","id":"1042"}'}},
            {'index': 1, 'function': {'arguments': '"黑羽"}'}}]}) + sse(finish='tool_calls')
        with patch.object(chat_model.urllib.request, 'build_opener', return_value=self.opening(raw)):
            message = chat_model.complete(self.settings, [], [])
        self.assertEqual(message['tool_calls'][0]['function']['name'], 'wiki_get')
        self.assertEqual(json.loads(message['tool_calls'][0]['function']['arguments'])['id'], '1042')
        self.assertEqual(json.loads(message['tool_calls'][1]['function']['arguments'])['query'], '黑羽')

    def test_truncated_malformed_and_oversized_streams_fail_safely(self):
        for raw in [sse({'content': '未完成'}), sse(finish='length'),
                    b'data: {invalid}\n\n', b'data: ' + b'x' * 65537,
                    sse({'tool_calls': [{'index': 4}]}),
                    b'data: {"error":{"message":"synthetic-secret"}}\n\n']:
            with self.subTest(raw=raw[:30]), \
                    patch.object(chat_model.urllib.request, 'build_opener', return_value=self.opening(raw)):
                with self.assertRaises(chat_model.ChatModelError) as error:
                    list(chat_model.stream(self.settings, [], []))
                self.assertNotIn('synthetic-secret', str(error.exception))

    def test_cancel_closes_upstream(self):
        opener = self.opening(sse({'content': '部分'}) + sse(finish='stop'))
        with patch.object(chat_model.urllib.request, 'build_opener', return_value=opener):
            stream = chat_model.stream(self.settings, [], [])
            next(stream)
            stream.close()
            opener.open.return_value.__exit__.assert_called_once()

    def test_keepalives_cannot_extend_total_timeout(self):
        opener = self.opening(b': ping\n\n' + sse(finish='stop'))
        with patch.object(chat_model.urllib.request, 'build_opener', return_value=opener), \
                patch.object(chat_model.time, 'monotonic', side_effect=[0, 121]):
            with self.assertRaisesRegex(chat_model.ChatModelError, '用时较长'):
                list(chat_model.stream(self.settings, [], []))
        opener.open.return_value.__exit__.assert_called_once()
