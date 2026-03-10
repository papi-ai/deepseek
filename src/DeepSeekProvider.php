<?php

/*
 * This file is part of PapiAI,
 * A simple but powerful PHP library for building AI agents.
 *
 * (c) Marcello Duarte <marcello.duarte@gmail.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

declare(strict_types=1);

namespace PapiAI\DeepSeek;

use Generator;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use RuntimeException;

/**
 * DeepSeek API provider for PapiAI.
 *
 * Bridges PapiAI's core types (Message, Response, ToolCall) with DeepSeek's OpenAI-compatible
 * API, handling format conversion in both directions. Supports chat completions, streaming,
 * tool calling, and structured JSON output.
 *
 * Authentication is via Bearer token in the Authorization header. All HTTP is done with ext-curl
 * directly, with no HTTP abstraction layer.
 *
 * Supported models:
 *   - deepseek-chat (general purpose)
 *   - deepseek-reasoner (reasoning)
 *
 * @see https://api-docs.deepseek.com/
 */
class DeepSeekProvider implements ProviderInterface
{
    private const API_URL = 'https://api.deepseek.com/chat/completions';

    public const MODEL_DEEPSEEK_CHAT = 'deepseek-chat';
    public const MODEL_DEEPSEEK_REASONER = 'deepseek-reasoner';

    /**
     * Create a new DeepSeek provider instance.
     *
     * @param string $apiKey           DeepSeek API key for Bearer token authentication
     * @param string $defaultModel     Model to use when not specified in options
     * @param int    $defaultMaxTokens Maximum output tokens when not specified in options
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_DEEPSEEK_CHAT,
        private readonly int $defaultMaxTokens = 4096,
    ) {
    }

    /**
     * Send a chat completion request to the DeepSeek API.
     *
     * Converts PapiAI Messages to OpenAI-compatible format, sends the request,
     * and parses the response back into a core Response object.
     *
     * @param array<Message> $messages Conversation history as PapiAI Message objects
     * @param array{
     *     model?: string,
     *     tools?: array,
     *     maxTokens?: int,
     *     temperature?: float,
     *     stopSequences?: array<string>,
     *     outputSchema?: array,
     * } $options Request options (model, tools, maxTokens, temperature, etc.)
     *
     * @return Response Parsed response containing text, tool calls, usage, and stop reason
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return Response::fromOpenAI($response, $messages);
    }

    /**
     * Stream a chat completion from the DeepSeek API using server-sent events.
     *
     * Yields StreamChunk objects as partial responses arrive. The final chunk
     * has isComplete set to true.
     *
     * @param array<Message> $messages Conversation history as PapiAI Message objects
     *
     * @return iterable<StreamChunk> Stream of partial response chunks
     *
     * @throws RuntimeException When the cURL request itself fails
     */
    public function stream(array $messages, array $options = []): iterable
    {
        $payload = $this->buildPayload($messages, $options);
        $payload['stream'] = true;

        foreach ($this->streamRequest($payload) as $event) {
            $delta = $event['choices'][0]['delta'] ?? [];
            if (isset($delta['content'])) {
                yield new StreamChunk($delta['content']);
            }
            if (($event['choices'][0]['finish_reason'] ?? null) !== null) {
                yield new StreamChunk('', isComplete: true);
            }
        }
    }

    /**
     * Whether this provider supports tool calling.
     *
     * @return bool Always true; DeepSeek supports function calling
     */
    public function supportsTool(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports vision (multimodal image input).
     *
     * @return bool Always false; DeepSeek does not support vision
     */
    public function supportsVision(): bool
    {
        return false;
    }

    /**
     * Whether this provider supports structured JSON output.
     *
     * @return bool Always true; DeepSeek supports JSON schema response format
     */
    public function supportsStructuredOutput(): bool
    {
        return true;
    }

    /**
     * Get the provider identifier.
     *
     * @return string The provider name "deepseek"
     */
    public function getName(): string
    {
        return 'deepseek';
    }

    /**
     * Build the API request payload.
     */
    private function buildPayload(array $messages, array $options): array
    {
        $apiMessages = [];

        foreach ($messages as $message) {
            if ($message instanceof Message) {
                $apiMessages[] = $this->convertMessage($message);
            }
        }

        $payload = [
            'model' => $options['model'] ?? $this->defaultModel,
            'messages' => $apiMessages,
        ];

        if (isset($options['maxTokens'])) {
            $payload['max_tokens'] = $options['maxTokens'];
        }

        if (isset($options['temperature'])) {
            $payload['temperature'] = $options['temperature'];
        }

        if (isset($options['stopSequences'])) {
            $payload['stop'] = $options['stopSequences'];
        }

        // Handle structured output / JSON mode
        if (isset($options['outputSchema'])) {
            $payload['response_format'] = [
                'type' => 'json_schema',
                'json_schema' => [
                    'name' => 'response',
                    'schema' => $options['outputSchema'],
                ],
            ];
        }

        // Handle tools
        if (isset($options['tools']) && !empty($options['tools'])) {
            $payload['tools'] = $this->convertTools($options['tools']);
        }

        return $payload;
    }

    /**
     * Convert a Message to OpenAI-compatible API format.
     */
    private function convertMessage(Message $message): array
    {
        $apiMessage = [
            'role' => $this->convertRole($message->role),
        ];

        if ($message->isTool()) {
            $apiMessage['role'] = 'tool';
            $apiMessage['content'] = $message->content;
            $apiMessage['tool_call_id'] = $message->toolCallId;
        } elseif ($message->hasToolCalls()) {
            $apiMessage['content'] = $message->getText() ?: null;
            $apiMessage['tool_calls'] = array_map(function (ToolCall $tc) {
                return [
                    'id' => $tc->id,
                    'type' => 'function',
                    'function' => [
                        'name' => $tc->name,
                        'arguments' => json_encode($tc->arguments),
                    ],
                ];
            }, $message->toolCalls);
        } elseif (is_array($message->content)) {
            $apiMessage['content'] = $this->convertMultimodalContent($message->content);
        } else {
            $apiMessage['content'] = $message->content;
        }

        return $apiMessage;
    }

    /**
     * Convert multimodal content to OpenAI-compatible format.
     */
    private function convertMultimodalContent(array $content): array
    {
        $parts = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $parts[] = ['type' => 'text', 'text' => $part['text']];
            } elseif ($part['type'] === 'image') {
                $source = $part['source'];
                if ($source['type'] === 'url') {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => ['url' => $source['url']],
                    ];
                } else {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => [
                            'url' => "data:{$source['media_type']};base64,{$source['data']}",
                        ],
                    ];
                }
            }
        }

        return $parts;
    }

    /**
     * Convert tools from PapiAI format to OpenAI-compatible format.
     */
    private function convertTools(array $tools): array
    {
        $openaiTools = [];

        foreach ($tools as $tool) {
            if (is_array($tool)) {
                $openaiTools[] = [
                    'type' => 'function',
                    'function' => [
                        'name' => $tool['name'],
                        'description' => $tool['description'],
                        'parameters' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
                    ],
                ];
            }
        }

        return $openaiTools;
    }

    /**
     * Convert Role to OpenAI-compatible role string.
     */
    private function convertRole(Role $role): string
    {
        return match ($role) {
            Role::System => 'system',
            Role::User => 'user',
            Role::Assistant => 'assistant',
            Role::Tool => 'tool',
        };
    }

    /**
     * Send a synchronous POST request to the DeepSeek API.
     *
     * @param array $payload JSON-encodable request body
     *
     * @return array Decoded JSON response body
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    protected function request(array $payload): array
    {
        $ch = curl_init(self::API_URL);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("DeepSeek API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $this->throwForStatusCode($httpCode, $data);
        }

        return $data;
    }

    /**
     * Throw the appropriate exception based on HTTP status code.
     *
     * Maps HTTP 401 to AuthenticationException, 429 to RateLimitException,
     * and all other error codes to ProviderException.
     *
     * @param int        $httpCode HTTP response status code
     * @param array|null $data     Decoded JSON response body, if available
     *
     * @throws AuthenticationException When HTTP 401 (invalid API key)
     * @throws RateLimitException      When HTTP 429 (rate limit exceeded)
     * @throws ProviderException       For all other HTTP error codes
     */
    protected function throwForStatusCode(int $httpCode, ?array $data): never
    {
        $errorMessage = $data['error']['message'] ?? 'Unknown error';

        if ($httpCode === 401) {
            throw new AuthenticationException(
                $this->getName(),
                $httpCode,
                $data,
            );
        }

        if ($httpCode === 429) {
            throw new RateLimitException(
                $this->getName(),
                statusCode: $httpCode,
                responseBody: $data,
            );
        }

        throw new ProviderException(
            "DeepSeek API error ({$httpCode}): {$errorMessage}",
            $this->getName(),
            $httpCode,
            $data,
        );
    }

    /**
     * Send a streaming POST request to the DeepSeek API.
     *
     * Buffers the full SSE response, then parses and yields each event as a decoded array.
     *
     * @param array $payload JSON-encodable request body (must include stream: true)
     *
     * @return Generator<array> Decoded JSON events from the SSE stream
     */
    protected function streamRequest(array $payload): Generator
    {
        $ch = curl_init(self::API_URL);

        $buffer = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);

        curl_exec($ch);
        curl_close($ch);

        // Parse SSE events
        $lines = explode("\n", $buffer);
        foreach ($lines as $line) {
            $line = trim($line);
            if (str_starts_with($line, 'data: ')) {
                $json = substr($line, 6);
                if ($json === '[DONE]') {
                    break;
                }
                $event = json_decode($json, true);
                if ($event !== null) {
                    yield $event;
                }
            }
        }
    }
}
