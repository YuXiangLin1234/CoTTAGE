import json
import math
import os
import re

from swift.plugin import ORM, orms

SCORE_RE = re.compile(r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
SCORE_MIN, SCORE_MAX = 1.0, 5.0
EPS = 1e-6


def extract_score(text):
    match = SCORE_RE.search(text or "")
    return float(match.group(1)) if match else None


def _num_generations(kwargs, default=8):
    env = os.environ.get("NUM_GENERATIONS")
    if env:
        return int(env)
    solutions = kwargs.get("solution") or []
    count = 1
    while count < len(solutions) and solutions[count] == solutions[0]:
        count += 1
    return count if count > 1 else default


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bhattacharyya(p, q):
    return math.sqrt(max(p * q, 0.0)) + math.sqrt(max((1.0 - p) * (1.0 - q), 0.0))


def _preference_target(gt_a, gt_b):
    # CoTTAGE metadata is already on the 1-5 scale.  
    # Applying that conversion here would rescale twice.
    threshold = max(float(os.environ.get("FIDELITY_TIE_THRESHOLD", "0.0")), 0.0)
    if gt_a - gt_b > threshold:
        return 1.0
    if gt_b - gt_a > threshold:
        return 0.0
    return 0.5


class FormatReward(ORM):
    def __call__(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            score = extract_score(completion)
            valid = (
                THINK_RE.search(completion or "") is not None
                and score is not None
                and SCORE_MIN <= score <= SCORE_MAX
            )
            rewards.append(1.0 if valid else 0.0)
        return rewards


class PreferenceReward(ORM):
    """Compare across the global batch, excluding self and assigned partner."""

    def __call__(self, completions, **kwargs):
        import torch.distributed as dist

        num_generations = _num_generations(kwargs)
        local = (list(completions), list(kwargs.get("solution") or []), num_generations)
        rank = 0
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]

        groups = []
        offset = 0
        dimensions = set()
        for rank_completions, solutions, group_size in gathered:
            if group_size != num_generations or group_size <= 0:
                raise ValueError("Preference reward requires the same positive NUM_GENERATIONS on every rank")
            if len(solutions) != len(rank_completions) or len(rank_completions) % group_size:
                raise ValueError("Preference reward requires complete GRPO groups and one solution per completion")
            for start in range(0, len(rank_completions), group_size):
                solution = solutions[start]
                solution = json.loads(solution) if isinstance(solution, str) else solution
                if solution.get("gt_score") is None:
                    raise KeyError("Preference reward requires gt_score for every audio group")
                dimensions.add(solution.get("dimension"))
                scores = [extract_score(c) for c in rank_completions[start:start + group_size]]
                valid = [s for s in scores if s is not None]
                mean = sum(valid) / len(valid) if valid else None
                variance = sum((s - mean) ** 2 for s in valid) / len(valid) if valid else None
                groups.append((offset + start, scores, solution, mean, variance))
            offset += len(rank_completions)
        if len(dimensions) > 1:
            raise ValueError("Preference reward requires a single dimension (REL or OVL) in the global batch")

        partners = {}
        pairs = {}
        for index, (_, _, solution, _, _) in enumerate(groups):
            pair_id, position = solution.get("pair_id"), solution.get("position")
            if pair_id is not None and position in ("a", "b"):
                pairs.setdefault(pair_id, {})[position] = index
        for sides in pairs.values():
            if "a" in sides and "b" in sides:
                partners[sides["a"]] = sides["b"]
                partners[sides["b"]] = sides["a"]

        rewards = [0.0] * offset
        for index, (start, scores, solution, mean, variance) in enumerate(groups):
            if mean is None:
                continue
            others = [
                group for other_index, group in enumerate(groups)
                if other_index != index and other_index != partners.get(index)
                and group[3] is not None
            ]
            if not others:
                continue
            for sample_index, score in enumerate(scores):
                if score is None:
                    continue
                total = 0.0
                for _, _, other_solution, other_mean, other_variance in others:
                    target = _preference_target(float(solution["gt_score"]), float(other_solution["gt_score"]))
                    spread = math.sqrt(variance + other_variance + EPS)
                    win_prob = _normal_cdf((score - other_mean) / spread)
                    total += _bhattacharyya(target, win_prob)
                rewards[start + sample_index] = total / len(others)

        # Prefix sums also handle unequal local batch sizes correctly.
        local_start = sum(len(payload[0]) for payload in gathered[:rank])
        return rewards[local_start:local_start + len(local[0])]


orms["cottage_format"] = FormatReward
orms["cottage_preference"] = PreferenceReward
