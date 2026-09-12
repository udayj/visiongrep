"""Local comparison labels; timing analysis is shared with cloud runs."""

def enabled(profile: dict) -> bool:
    return profile["name"] == "local-quick" and not profile.get("cloud", False)


def candidate_decision(
    comparisons: dict, resources: dict, scenarios: list[str]
) -> tuple[str, list[str]]:
    """A local shortlist decision never authorizes or substitutes for cloud runs."""
    required_resources = {
        name + ":" + field
        for name in scenarios
        for field in ("peak_rss_bytes", "index_bytes")
    }
    if set(comparisons) != set(scenarios) or not required_resources <= resources.keys():
        return "inconclusive", ["incomplete local timing or resource comparisons"]
    checks = comparisons | resources
    regressions = [name for name, item in checks.items() if item["interval"][1] < -0.05]
    if regressions:
        return "does_not_qualify", [
            "material local regression: " + name for name in regressions
        ]
    if any(item["interval"][0] < -0.05 for item in checks.values()):
        return "inconclusive", [
            "cannot exclude a material regression in the local suite"
        ]
    improved = [
        name
        for name, item in comparisons.items()
        if item["improvement"] >= 0.05 and item["interval"][0] > 0
    ]
    if improved:
        return "promising", [
            "supported local improvement: " + name for name in improved
        ] + [
            "local suite only; eligible for consideration for cloud validation, not qualified"
        ]
    if all(item["interval"][1] < 0.05 for item in comparisons.values()):
        return "does_not_qualify", [
            "no material improvement supported by the local suite"
        ]
    return "inconclusive", ["local improvement remains uncertain"]
