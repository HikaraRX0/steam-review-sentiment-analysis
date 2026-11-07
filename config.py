FPS_GAME_IDS = {
    "CS2": 730,
    "COD_MW4": 4435490,
    "APEX": 1172470,
    "BATTLEFIELD_6": 2807960,
    "DELTA_FORCE": 2507950,
    "TITANFALL_2": 1237970,
    "RAINBOW_SIX_SIEGE": 359550,
}

RPG_GAME_IDS = {
    # Add RPG game names and AppIDs here.
}

# Held-out games for external evaluation; exclude from training and validation.
EXTERNAL_TEST_GAME_IDS = {
    "WARDOGS": 1867240,
    "CYBERPUNK_2077": 1091500,
    "PERSONA_5_ROYAL": 1687950,
}

# The collector accepts games from every genre.
GAME_IDS = {**FPS_GAME_IDS, **RPG_GAME_IDS, **EXTERNAL_TEST_GAME_IDS}
