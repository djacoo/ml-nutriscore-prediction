"""
Country name mappings for data cleaning.

Maps various country name formats to standard English names.
"""

# Country name overrides for standardization
COUNTRY_OVERRIDES = {
    # Handle official names that get split incorrectly
    'plurinational state of': 'Bolivia',
    'bolivarian republic of': 'Venezuela',
    'republic of korea': 'South Korea',

    # North America
    'usa': 'United States',
    'us': 'United States',
    'en:us': 'United States',
    'états-unis': 'United States',

    # European countries
    'uk': 'United Kingdom',
    'en:gb': 'United Kingdom',
    'royaume-uni': 'United Kingdom',

    'de': 'Germany',
    'deutschland': 'Germany',
    'allemagne': 'Germany',

    'fr': 'France',
    'en:fr': 'France',
    'francia': 'France',

    'es': 'Spain',
    'españa': 'Spain',
    'espagne': 'Spain',

    'it': 'Italy',
    'italia': 'Italy',

    'nl': 'Netherlands',
    'nederland': 'Netherlands',
    'holland': 'Netherlands',

    'be': 'Belgium',
    'belgique': 'Belgium',

    'ch': 'Switzerland',
    'suisse': 'Switzerland',
    'schweiz': 'Switzerland',

    'cz': 'Czechia',
    'česko': 'Czechia',

    'pl': 'Poland',
    'polska': 'Poland',

    'se': 'Sweden',
    'sverige': 'Sweden',

    'no': 'Norway',
    'norge': 'Norway',

    # Other regions
    'ru': 'Russian Federation',
    'russia': 'Russian Federation',

    'jp': 'Japan',
    'cn': 'China',

    'br': 'Brazil',
    'brasil': 'Brazil',
}
