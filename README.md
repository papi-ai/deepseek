# PapiAI DeepSeek Provider

[![Tests](https://github.com/papi-ai/deepseek/workflows/CI/badge.svg)](https://github.com/papi-ai/deepseek/actions?query=workflow%3ACI)

DeepSeek provider for [PapiAI](https://github.com/papi-ai/papi-core) - A simple but powerful PHP library for building AI agents.

## Installation

```bash
composer require papi-ai/deepseek
```

## Usage

```php
use PapiAI\Core\Agent;
use PapiAI\DeepSeek\DeepSeekProvider;

$provider = new DeepSeekProvider(
    apiKey: $_ENV['DEEPSEEK_API_KEY'],
);

$agent = new Agent(
    provider: $provider,
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Available Models

```php
DeepSeekProvider::MODEL_DEEPSEEK_CHAT      // 'deepseek-chat' (default)
DeepSeekProvider::MODEL_DEEPSEEK_REASONER  // 'deepseek-reasoner' (reasoning)
```

## Features

- Tool/function calling
- Structured output (JSON mode)
- Streaming support

## License

MIT
