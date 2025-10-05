# prompts.py
import random

PROMPT_POOL = {
    "drywall_join": [
        "segment the drywall taping or joint area on the wall surface",
        "find and mask the taped joints between drywall sheets",
        "highlight the drywall seams or taping regions accurately",
        "detect and segment the drywall taping areas"
    ],
    "cracks": [
        "segment visible cracks or defects on the wall surface",
        "find and mask wall crack regions clearly",
        "highlight damaged or cracked portions of the wall",
        "detect and segment surface cracks and imperfections"
    ]
}

def get_random_prompt(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in PROMPT_POOL:
        return random.choice(PROMPT_POOL[dataset_name])
    return "segment the region of interest"
