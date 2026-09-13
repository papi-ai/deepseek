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

use PapiAI\Core\Effort;
use PapiAI\DeepSeek\DeepSeekModel;
use PapiAI\DeepSeek\DeepSeekProvider;

describe('DeepSeekModel', function () {
    it('is the source of truth the old constants alias', function () {
        expect(DeepSeekProvider::MODEL_DEEPSEEK_FLASH)->toBe(DeepSeekModel::Flash->value);
        expect(DeepSeekProvider::MODEL_DEEPSEEK_V4_FLASH)->toBe(DeepSeekModel::V4Flash->value);
    });

    it('ships unique IDs', function () {
        $ids = array_map(fn (DeepSeekModel $m) => $m->value, DeepSeekModel::cases());

        expect($ids)->toBe(array_unique($ids));
    });

    it('returns null for an ID it has not heard of, rather than throwing', function () {
        expect(DeepSeekModel::tryFrom('deepseek-v5'))->toBeNull();
    });

    it('keeps Pro off the low level it does not honour', function () {
        expect(DeepSeekModel::V4Pro->effortLevels())->not->toContain(Effort::Low);
        expect(DeepSeekModel::Flash->effortLevels())->toContain(Effort::Low);
    });

    it('knows which models are retired, when, and what replaces them', function () {
        expect(DeepSeekModel::V4Flash->isDeprecated())->toBeTrue();
        expect(DeepSeekModel::V4Flash->retiredOn())->toBe('2026-09-10');
        expect(DeepSeekModel::Chat->replacement())->toBe(DeepSeekModel::Flash);
        expect(DeepSeekModel::Flash->isDeprecated())->toBeFalse();
    });
});
