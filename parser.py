import argparse
import re
from typing import Optional, Tuple

import pandas as pd
import phonenumbers
from phonenumbers import geocoder

# Define COUNTRY_CODES based on countries.txt
COUNTRY_CODES = {
    '+93': 'Afghanistan', '+355': 'Albania', '+213': 'Algeria', '+376': 'Andorra', '+244': 'Angola',
    '+1-268': 'Antigua and Barbuda', '+54': 'Argentina', '+374': 'Armenia', '+297': 'Aruba', '+61': 'Australia',
    '+43': 'Austria', '+994': 'Azerbaijan', '+1-242': 'Bahamas', '+973': 'Bahrain', '+880': 'Bangladesh',
    '+1-246': 'Barbados', '+375': 'Belarus', '+32': 'Belgium', '+501': 'Belize', '+229': 'Benin', '+975': 'Bhutan',
    '+591': 'Bolivia', '+387': 'Bosnia and Herzegovina', '+267': 'Botswana', '+55': 'Brazil', '+673': 'Brunei',
    '+359': 'Bulgaria', '+226': 'Burkina Faso', '+95': 'Burma', '+257': 'Burundi', '+855': 'Cambodia', '+237': 'Cameroon',
    '+1': 'Canada', '+238': 'Cape Verde', '+236': 'Central African Republic', '+235': 'Chad', '+56': 'Chile',
    '+86': 'China', '+57': 'Colombia', '+269': 'Comoros', '+506': 'Costa Rica', '+225': 'Cote d\'Ivoire',
    '+385': 'Croatia', '+53': 'Cuba', '+599': 'Curacao', '+357': 'Cyprus', '+420': 'Czech Republic',
    '+243': 'Democratic Republic of the Congo', '+45': 'Denmark', '+253': 'Djibouti', '+1-767': 'Dominica',
    '+1-809': 'Dominican Republic', '+670': 'East Timor', '+593': 'Ecuador', '+20': 'Egypt', '+503': 'El Salvador',
    '+240': 'Equatorial Guinea', '+291': 'Eritrea', '+372': 'Estonia', '+251': 'Ethiopia', '+679': 'Fiji',
    '+358': 'Finland', '+33': 'France', '+241': 'Gabon', '+220': 'Gambia', '+995': 'Georgia', '+49': 'Germany',
    '+233': 'Ghana', '+30': 'Greece', '+1-473': 'Grenada', '+502': 'Guatemala', '+224': 'Guinea', '+245': 'Guinea-Bissau',
    '+592': 'Guyana', '+509': 'Haiti', '+39': 'Holy See', '+504': 'Honduras', '+852': 'Hong Kong', '+36': 'Hungary',
    '+354': 'Iceland', '+91': 'India', '+62': 'Indonesia', '+98': 'Iran', '+964': 'Iraq', '+353': 'Ireland',
    '+972': 'Israel', '+39': 'Italy', '+1-876': 'Jamaica', '+81': 'Japan', '+962': 'Jordan', '+7': 'Kazakhstan',
    '+254': 'Kenya', '+686': 'Kiribati', '+383': 'Kosovo', '+965': 'Kuwait', '+996': 'Kyrgyzstan', '+856': 'Laos',
    '+371': 'Latvia', '+961': 'Lebanon', '+266': 'Lesotho', '+231': 'Liberia', '+218': 'Libya', '+423': 'Liechtenstein',
    '+370': 'Lithuania', '+352': 'Luxembourg', '+853': 'Macau', '+389': 'Macedonia', '+261': 'Madagascar',
    '+265': 'Malawi', '+60': 'Malaysia', '+960': 'Maldives', '+223': 'Mali', '+356': 'Malta', '+692': 'Marshall Islands',
    '+222': 'Mauritania', '+230': 'Mauritius', '+52': 'Mexico', '+691': 'Micronesia', '+373': 'Moldova', '+377': 'Monaco',
    '+976': 'Mongolia', '+382': 'Montenegro', '+212': 'Morocco', '+258': 'Mozambique', '+264': 'Namibia', '+674': 'Nauru',
    '+977': 'Nepal', '+31': 'Netherlands', '+599': 'Netherlands Antilles', '+64': 'New Zealand', '+505': 'Nicaragua',
    '+227': 'Niger', '+234': 'Nigeria', '+850': 'North Korea', '+47': 'Norway', '+968': 'Oman', '+92': 'Pakistan',
    '+680': 'Palau', '+970': 'Palestinian Territories', '+507': 'Panama', '+675': 'Papua New Guinea', '+595': 'Paraguay',
    '+51': 'Peru', '+63': 'Philippines', '+48': 'Poland', '+351': 'Portugal', '+974': 'Qatar', '+242': 'Republic of the Congo',
    '+40': 'Romania', '+7': 'Russia', '+250': 'Rwanda', '+1-869': 'Saint Kitts and Nevis', '+1-758': 'Saint Lucia',
    '+1-784': 'Saint Vincent and the Grenadines', '+685': 'Samoa', '+378': 'San Marino', '+239': 'Sao Tome and Principe',
    '+966': 'Saudi Arabia', '+221': 'Senegal', '+381': 'Serbia', '+248': 'Seychelles', '+232': 'Sierra Leone',
    '+65': 'Singapore', '+1-721': 'Sint Maarten', '+421': 'Slovakia', '+386': 'Slovenia', '+677': 'Solomon Islands',
    '+252': 'Somalia', '+27': 'South Africa', '+82': 'South Korea', '+211': 'South Sudan', '+34': 'Spain',
    '+94': 'Sri Lanka', '+249': 'Sudan', '+597': 'Suriname', '+268': 'Swaziland', '+46': 'Sweden', '+41': 'Switzerland',
    '+963': 'Syria', '+886': 'Taiwan', '+992': 'Tajikistan', '+255': 'Tanzania', '+66': 'Thailand', '+670': 'Timor-Leste',
    '+228': 'Togo', '+676': 'Tonga', '+1-868': 'Trinidad and Tobago', '+216': 'Tunisia', '+90': 'Turkey',
    '+993': 'Turkmenistan', '+688': 'Tuvalu', '+256': 'Uganda', '+380': 'Ukraine', '+971': 'United Arab Emirates',
    '+44': 'United Kingdom', '+1': 'United States', '+598': 'Uruguay', '+998': 'Uzbekistan', '+678': 'Vanuatu',
    '+58': 'Venezuela', '+84': 'Vietnam', '+967': 'Yemen', '+260': 'Zambia', '+263': 'Zimbabwe'
}

def parse_phone_number(phone: str) -> Tuple[Optional[str], Optional[str]]:
    if not phone or not isinstance(phone, str):
        return None, None
    try:
        parsed = phonenumbers.parse(phone, None)
        if phonenumbers.is_valid_number(parsed):
            country = geocoder.description_for_number(parsed, "en")
            number = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL).replace(' ', '')
            return country, number
    except:
        pass
    # Fallback
    clean_phone = re.sub(r'[^\d+]', '', phone)
    for code in sorted(COUNTRY_CODES, key=len, reverse=True):
        if clean_phone.startswith(code):
            return COUNTRY_CODES[code], clean_phone[len(code):]
    return None, clean_phone

# Load legal suffixes once at module level
with open("legal.txt", "r", encoding="utf-8") as f:
    LEGAL_SUFFIXES = [line.strip() for line in f if line.strip()]

# Build a regex pattern: \b(suffix1|suffix2|suffix3)\b$ (case-insensitive, end of string)
LEGAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(suffix) for suffix in LEGAL_SUFFIXES) + r")\b\.?$",
    re.IGNORECASE
)

def parse_company_name(company: str) -> Tuple[Optional[str], Optional[str]]:
    if not company or not isinstance(company, str):
        return None, None

    company = company.strip()

    # Try to match legal suffix at the end
    match = LEGAL_PATTERN.search(company)
    if match:
        legal = match.group(1).strip()
        # Remove the matched suffix part from company name
        name = company[:match.start()].strip()
        return name, legal

    # No legal suffix found
    return company, None

def process_file(file_path: str):
    df = pd.read_csv(file_path)
    df_out = pd.DataFrame(index=df.index)
    output_columns = []

    # Iterate over all values in the DataFrame
    for col in df.columns:
        for idx, value in df[col].items():
            if pd.notna(value):
                value_str = str(value)
                # Check if it looks like a phone number
                country, number = parse_phone_number(value_str)
                if country or number:
                    if 'PhoneNumber' not in output_columns:
                        output_columns.extend(['PhoneNumber', 'Country', 'Number'])
                        df_out['PhoneNumber'] = ''
                        df_out['Country'] = ''
                        df_out['Number'] = ''
                    df_out.at[idx, 'PhoneNumber'] = value_str
                    df_out.at[idx, 'Country'] = country or ''
                    df_out.at[idx, 'Number'] = number or ''
                    continue

                # Check if it looks like a company name
                name, legal = parse_company_name(value_str)
                if name != value_str or legal:  # Only add if a split occurred
                    if 'CompanyName' not in output_columns:
                        output_columns.extend(['CompanyName', 'Name', 'Legal'])
                        df_out['CompanyName'] = ''
                        df_out['Name'] = ''
                        df_out['Legal'] = ''
                    df_out.at[idx, 'CompanyName'] = value_str
                    df_out.at[idx, 'Name'] = name or ''
                    df_out.at[idx, 'Legal'] = legal or ''
                    continue

    if output_columns:
        df_out = df_out[output_columns]
        df_out.to_csv('output.csv', index=False)
    else:
        print("No Phone Number or Company Name data detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse and normalize data.')
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    args = parser.parse_args()
    process_file(args.input)