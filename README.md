# PapiAI DeepSeek Provider

[![CI](https://github.com/papi-ai/deepseek/workflows/CI/badge.svg)](https://github.com/papi-ai/deepseek/actions?query=workflow%3ACI) [![Latest Version](https://img.shields.io/packagist/v/papi-ai/deepseek.svg)](https://packagist.org/packages/papi-ai/deepseek) [![Total Downloads](https://img.shields.io/packagist/dt/papi-ai/deepseek.svg)](https://packagist.org/packages/papi-ai/deepseek) [![PHP Version](https://img.shields.io/packagist/php-v/papi-ai/deepseek.svg)](https://packagist.org/packages/papi-ai/deepseek) [![License](https://img.shields.io/packagist/l/papi-ai/deepseek.svg)](https://packagist.org/packages/papi-ai/deepseek)

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
DeepSeekProvider::MODEL_DEEPSEEK_V4_FLASH  // 'deepseek-v4-flash' (default)
DeepSeekProvider::MODEL_DEEPSEEK_V4_PRO    // 'deepseek-v4-pro'
```

The `MODEL_DEEPSEEK_CHAT` and `MODEL_DEEPSEEK_REASONER` constants are still shipped but deprecated: both were discontinued on 24 July 2026 and requests using them fail. The V4 family replaces the old chat and reasoner split, with the neutral `effort` option controlling how hard the model thinks.


## Features

- Tool/function calling
- Structured output (JSON mode)
- Streaming support

## License

MIT
