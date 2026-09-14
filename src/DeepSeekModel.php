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

use PapiAI\Core\Effort;

/**
 * Every DeepSeek model this package knows, and which effort levels each accepts.
 *
 * The enum is the source of truth: the `MODEL_*` constants on DeepSeekProvider alias its values.
 * It lets a watchdog enumerate what we ship instead of parsing source, and each case knows
 * whether it has been retired and what replaces it.
 *
 * An ID we have not heard of is not an error: `tryFrom()` returns null and the provider falls
 * back to reading the name, assuming newer rather than older.
 *
 * @see https://api-docs.deepseek.com/quick_start/pricing
 */
enum DeepSeekModel: string
{
    /** The rolling name for the current Flash generation. Follows the generation forward. */
    case Flash = 'deepseek-flash';
    /**
     * Live but unstable: DeepSeek's news page says Pro routes to Flash from 14 September 2026
     * until V4.1-Pro ships, and their pricing page says service continues unchanged.
     */
    case V4Pro = 'deepseek-v4-pro';
    /** @deprecated Retired 10 September 2026; routed to Flash "temporarily". Use Flash. */
    case V4Flash = 'deepseek-v4-flash';
    /** @deprecated Discontinued 24 July 2026; requests fail. Use Flash. */
    case Chat = 'deepseek-chat';
    /** @deprecated Discontinued 24 July 2026; requests fail. Use Flash. */
    case Reasoner = 'deepseek-reasoner';

    /**
     * The effort levels this model accepts, in the neutral vocabulary.
     *
     * DeepSeek has three of its own (low, high, max) and Pro does not honour low.
     *
     * @return non-empty-list<Effort>
     */
    public function effortLevels(): array
    {
        return match ($this) {
            self::V4Pro => [Effort::High, Effort::Maximum],
            default => [Effort::Low, Effort::High, Effort::Maximum],
        };
    }

    /**
     * Whether DeepSeek has retired this model.
     */
    public function isDeprecated(): bool
    {
        return match ($this) {
            self::V4Flash, self::Chat, self::Reasoner => true,
            default => false,
        };
    }

    /**
     * The published retirement date, ISO formatted.
     */
    public function retiredOn(): ?string
    {
        return match ($this) {
            self::V4Flash => '2026-09-10',
            self::Chat, self::Reasoner => '2026-07-24',
            default => null,
        };
    }

    /**
     * What to use instead, for retired models.
     */
    public function replacement(): ?self
    {
        return match ($this) {
            self::V4Flash, self::Chat, self::Reasoner => self::Flash,
            default => null,
        };
    }
}
