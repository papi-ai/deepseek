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
DeepSeekProvider::MODEL_DEEPSEEK_CHAT      // 'deepseek-chat' (default)
DeepSeekProvider::MODEL_DEEPSEEK_REASONER  // 'deepseek-reasoner' (reasoning)
```

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
