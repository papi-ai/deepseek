# DeepSeek

DeepSeek provider for PapiAI.

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

## Models

```php
DeepSeekProvider::MODEL_DEEPSEEK_V4_FLASH  // 'deepseek-v4-flash' (default)
DeepSeekProvider::MODEL_DEEPSEEK_V4_PRO    // 'deepseek-v4-pro'
```

The `MODEL_DEEPSEEK_CHAT` and `MODEL_DEEPSEEK_REASONER` constants are still shipped but deprecated: both were discontinued on 24 July 2026 and requests using them fail. The V4 family replaces the old chat and reasoner split, with the neutral `effort` option controlling how hard the model thinks.


## Capabilities

| Capability | Supported |
|---|---|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes |
| Structured output | Yes |

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
